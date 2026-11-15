"""Nemotron 3.5 ASR streaming (cache-aware) の推論コア。

NeMo 標準の streaming 推論ロジック（StreamingSession / モデル設定）を ASRService から
別プロセスで利用できる形に移植したもの。nemo-toolkit はアプリ本体の venv とは分離した
専用 venv（既定 .venv_nemotron。scripts/setup_nemotron_venv.ps1 で作成）にインストールし、
nemotron_asr_server.py がその venv の Python で起動される。
マイク取り込み・argparse・CLI 部分は持たない（音声フレームは ASRService 側から feed する）。

ストリーミング方式は標準スクリプトと同一:
  全音声を前処理して mel を作り、新規フレーム分のチャンクだけを encoder cache 付きで
  conformer_stream_step に渡す逐次手法。直近 window 秒だけ保持して前処理コストを一定化する。
"""

from __future__ import annotations

import base64
import json
import re
import socket
import struct
import sys
import time

import numpy as np

SAMPLE_RATE = 16000

DEFAULT_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# 取りこぼし防止のため前処理に使う直近音声の窓（秒）。正規化統計を安定させる役割。
DEFAULT_WINDOW_SEC = 15.0

# RNNT greedy の1フレームあたり最大トークン数。既定 10 は日本語で小さすぎ取りこぼす。
# 30 で CER 約半減・処理時間不変（実測で 30 が頭打ち）。
DEFAULT_MAX_SYMBOLS = 30

# チャンクサイズ(ms) -> att_context_size の right_context。
# cache-aware FastConformer はエンコーダ1フレーム=80ms、chunk=(right_context+1)*80ms。
# 値は nemotron-3.5-asr-streaming-0.6b の学習済み right_context(0/3/6/13)に対応。
# 二重 import はここだけ意図的に残す。このモジュールは
# (a) 本体アプリから `maai_sds.nemotron_asr` として、
# (b) `nemotron_asr_server.py` から `.venv_nemotron`（maai_sds 未インストール）で
#     トップレベル module として、
# の両方の経路で読まれる。(b) では相対 import が ImportError になる。
try:
    from .nemotron_chunk import CHUNK_TO_RIGHT_CONTEXT
except ImportError:  # noqa: F401  -- (b) 別 venv でスクリプト起動された場合
    from nemotron_chunk import CHUNK_TO_RIGHT_CONTEXT

_LANG_TAG_RE = re.compile(r"\s*<[A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?>\s*")


def _strip_lang_tags(text: str) -> str:
    return _LANG_TAG_RE.sub(" ", text).strip()


class StreamingSession:
    """server.py 準拠の逐次 ASR。直近 window 秒だけ前処理する有界版。

    - 取りこぼし防止: 全音声を前処理して mel を作り、新規フレーム分のチャンクだけを
      encoder cache 付きで渡す（オフライン全文とほぼ一致する手法）。
    - 止まらない: 保持音声を直近 window 秒に制限し、前処理コスト・メモリを一定化。
    """

    def __init__(self, model, strip_lang_tags: bool, window_sec: float) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.device = model.device
        self.strip = strip_lang_tags

        scfg = model.encoder.streaming_cfg
        self.shift = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else int(scfg.shift_size)
        pe = scfg.pre_encode_cache_size
        self.precache = pe[1] if isinstance(pe, list) else int(pe)
        self.drop = int(scfg.drop_extra_pre_encoded)
        self.hop = int(model.cfg.preprocessor.window_stride * SAMPLE_RATE)
        self.chunk_mel_frames = self.shift  # 定常時の1チャンク mel フレーム数
        # 直近窓（mel フレーム）。前処理コストと正規化統計の安定性のトレードオフ。
        self.window_frames = int(window_sec * 1000 / 10)
        self.reset()

    def reset(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.base_frames = 0  # トリムで捨てた mel フレーム数
        self.emitted = 0  # グローバルに処理済みの mel フレーム数
        (
            self.cache_ch,
            self.cache_t,
            self.cache_ch_len,
        ) = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.previous_hypotheses = None
        self.current_text = ""

    def feed(self, audio_f32: np.ndarray) -> str:
        if audio_f32.size:
            self.audio = np.concatenate([self.audio, audio_f32])
        self._trim()
        while True:
            need = (self.emitted - self.base_frames + self.shift + 1) * self.hop
            if len(self.audio) < need:
                break
            self._process(final=False)
        return self.current_text

    def finalize(self) -> str:
        pad = (self.precache + self.shift) * self.hop
        self.audio = np.concatenate([self.audio, np.zeros(pad, dtype=np.float32)])
        self._process(final=True)
        return self.current_text

    def _trim(self) -> None:
        """保持音声を直近窓に制限（O(n^2) と肥大化を防ぐ）。emitted のローカル位置を base_frames で補正。"""
        max_samples = (self.window_frames + self.shift + self.precache + 2) * self.hop
        if len(self.audio) <= max_samples:
            return
        drop_frames = (len(self.audio) - max_samples) // self.hop
        if drop_frames <= 0:
            return
        # まだ処理していないフレームは捨てない
        drop_frames = min(drop_frames, (self.emitted - self.base_frames) - self.precache)
        if drop_frames <= 0:
            return
        self.audio = self.audio[drop_frames * self.hop :]
        self.base_frames += drop_frames

    def _process(self, final: bool) -> None:
        torch = self._torch
        try:
            at = torch.from_numpy(self.audio).unsqueeze(0).to(self.device)
            al = torch.tensor([len(self.audio)], device=self.device)
            with torch.inference_mode():
                mel, _ = self.model.preprocessor(input_signal=at, length=al)
                local_emitted = self.emitted - self.base_frames
                avail = mel.shape[-1] - (0 if final else 1)
                if not final and (avail - local_emitted) < self.shift:
                    return
                if self.emitted == 0:
                    start = 0
                    end = mel.shape[-1] if final else self.shift
                    drop = 0
                else:
                    start = max(0, local_emitted - self.precache)
                    end = mel.shape[-1] if final else local_emitted + self.shift
                    drop = self.drop if start > 0 else 0
                chunk = mel[:, :, start:end]
                if chunk.shape[-1] == 0:
                    return
                clen = torch.tensor([chunk.shape[-1]], device=self.device)
                (
                    _pred,
                    txts,
                    self.cache_ch,
                    self.cache_t,
                    self.cache_ch_len,
                    self.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk,
                    processed_signal_length=clen,
                    cache_last_channel=self.cache_ch,
                    cache_last_time=self.cache_t,
                    cache_last_channel_len=self.cache_ch_len,
                    keep_all_outputs=final,
                    previous_hypotheses=self.previous_hypotheses,
                    drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
                if not final:
                    self.emitted += self.shift
                if txts and txts[0]:
                    hyp = txts[0]
                    text = hyp.text if hasattr(hyp, "text") else str(hyp)
                    if self.strip:
                        text = _strip_lang_tags(text)
                    self.current_text = text
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Nemotron ASR] chunk error (skipped): {exc}", file=sys.stderr, flush=True)


class NemotronStreamingASR:
    """Nemotron 3.5 ASR streaming モデルをロードし、StreamingSession を保持する in-process ラッパ。

    ASRService から `feed(audio_f32) -> str` / `finalize() -> str` / `reset()` を呼ぶ。
    返り値は「現在までに認識した全文」（増分ではない）。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_size_ms: int = 320,
        target_lang: str = "ja-JP",
        strip_lang_tags: bool = True,
        device: str = "cuda",
        window_sec: float = DEFAULT_WINDOW_SEC,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
        log: object | None = None,
    ) -> None:
        import os

        import torch

        import nemo.collections.asr as nemo_asr
        from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

        self._torch = torch
        self._log_fn = log

        if chunk_size_ms not in CHUNK_TO_RIGHT_CONTEXT:
            raise ValueError(
                f"未対応の chunk_size_ms={chunk_size_ms}。利用可能: {sorted(CHUNK_TO_RIGHT_CONTEXT)}"
            )

        # --- デバイス決定（CUDA 優先、無ければ CPU フォールバック）---
        want_cuda = str(device).strip().lower().startswith("cuda")
        if want_cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            if want_cuda:
                self._log("CUDA が無いため CPU で実行します（低速）。")
            self.device = "cpu"

        # --- モデルロード ---
        is_local = model_name.endswith(".nemo") or os.path.exists(model_name)
        if is_local:
            self._log(f"Loading local model: {model_name}")
            model = nemo_asr.models.ASRModel.restore_from(model_name, map_location="cpu")
        else:
            self._log(f"Loading model: {model_name}")
            model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")

        # --- チャンクサイズ -> att_context_size ---
        right_context = CHUNK_TO_RIGHT_CONTEXT[chunk_size_ms]
        if not hasattr(model.encoder, "set_default_att_context_size"):
            raise RuntimeError(
                "このモデルは cache-aware streaming（set_default_att_context_size）に対応していません。"
            )
        all_opts = [list(o) for o in model.encoder.att_context_size_all]
        match = next((o for o in all_opts if int(o[1]) == right_context), None)
        if match is None:
            raise RuntimeError(
                f"chunk-size {chunk_size_ms}ms (right_context={right_context}) はこのモデルでは利用できません。"
                f" モデルが受理可能な att_context_size 候補: {all_opts}"
            )
        left = int(match[0])
        model.encoder.set_default_att_context_size(match)
        model.encoder.setup_streaming_params()

        # --- greedy + CUDA graph 無効（逐次ストリーミング向け）、max_symbols ---
        if hasattr(model, "change_decoding_strategy") and hasattr(model, "joint"):
            rnnt_decoding = RNNTDecodingConfig(fused_batch_size=-1)
            rnnt_decoding.greedy.use_cuda_graph_decoder = False
            rnnt_decoding.greedy.max_symbols = max_symbols
            model.change_decoding_strategy(rnnt_decoding)

        # --- 言語指定（プロンプト）。失敗時は auto にフォールバック ---
        if hasattr(model, "set_inference_prompt"):
            try:
                model.set_inference_prompt(target_lang)
            except Exception:  # noqa: BLE001
                model.set_inference_prompt("auto")
        if strip_lang_tags and hasattr(model, "decoding") and hasattr(model.decoding, "set_strip_lang_tags"):
            try:
                model.decoding.set_strip_lang_tags(True)
            except Exception:  # noqa: BLE001
                pass

        # 決定的前処理（dither 無効化）
        if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
            model.preprocessor.featurizer.dither = 0.0

        model = model.to(device=self.device, dtype=torch.float32)
        model.eval()
        self.model = model

        self.session = StreamingSession(
            model,
            strip_lang_tags=strip_lang_tags,
            window_sec=window_sec,
        )

        actual_ms = self.session.chunk_mel_frames * 10  # mel frame = 10ms
        self._log(
            f"Device={self.device} lang={target_lang} att_context_size=[{left}, {right_context}] "
            f"streaming chunk ~ {actual_ms} ms window={window_sec:.0f}s max_symbols={max_symbols}"
        )
        self._warmup()

    def _log(self, message: str) -> None:
        if callable(self._log_fn):
            try:
                self._log_fn(f"[Nemotron] {message}")
                return
            except Exception:  # noqa: BLE001
                pass
        print(f"[Nemotron] {message}", flush=True)

    def _warmup(self) -> None:
        """無音を数チャンク流して CUDA カーネルをコンパイル/メモリ確保。"""
        start = time.perf_counter()
        silence = np.zeros(self.session.shift * self.session.hop, dtype=np.float32)
        for _ in range(5):
            self.session.feed(silence)
        self.session.reset()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._log(f"Warmup done in {elapsed_ms:.0f}ms")

    def feed(self, audio_f32: np.ndarray) -> str:
        return self.session.feed(np.asarray(audio_f32, dtype=np.float32).reshape(-1))

    def finalize(self) -> str:
        return self.session.finalize()

    def reset(self) -> None:
        self.session.reset()


# ---------------------------------------------------------------------------
# サブプロセス橋渡し用プロトコル（length-prefixed JSON フレーム）
#   4byte big-endian 長さ N + N byte の UTF-8 JSON。
#   feed の音声は JSON 内に float32-LE を base64 で格納する。
# サーバ（専用 venv で実行）と クライアント（アプリ env）で共有する。
# ---------------------------------------------------------------------------


def send_frame(sock: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while reading frame")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> dict:
    header = _recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    payload = _recv_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def encode_audio_f32(audio: np.ndarray) -> str:
    arr = np.asarray(audio, dtype="<f4").reshape(-1)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_audio_f32(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64.encode("ascii"))
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


class NemotronStreamingClient:
    """専用 venv で動く nemotron_asr_server.py へ接続する薄いクライアント。

    `NemotronStreamingASR` と同じ `feed/finalize/reset` インターフェースを提供するので、
    ASRService 側の推論ループはそのまま使える。RPC は同期（1リクエスト=1レスポンス）。
    """

    def __init__(self, host: str, port: int, timeout: float = 30.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._sock = socket.create_connection((host, int(port)), timeout=self.timeout)
        self._sock.settimeout(self.timeout)

    def _rpc(self, payload: dict) -> dict:
        send_frame(self._sock, payload)
        resp = recv_frame(self._sock)
        if isinstance(resp, dict) and resp.get("error"):
            raise RuntimeError(str(resp.get("error")))
        return resp

    def feed(self, audio_f32: np.ndarray) -> str:
        resp = self._rpc({"op": "feed", "audio_b64": encode_audio_f32(audio_f32)})
        return str(resp.get("text", ""))

    def finalize(self) -> str:
        resp = self._rpc({"op": "finalize"})
        return str(resp.get("text", ""))

    def reset(self) -> None:
        self._rpc({"op": "reset"})

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass
