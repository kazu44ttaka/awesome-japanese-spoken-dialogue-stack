"""日本語 ASR の各モデルを同じ形（16 kHz mono float32 の配列 → 文字列）で呼べるようにする。

    from asrkit.backends import BACKENDS, create
    recognize = create("moonshine-tiny-ja", device="cpu")
    text = recognize(audio)   # audio: np.ndarray（16 kHz・mono・float32）

transformers / funasr / sherpa-onnx のうち、使うモデルに必要なものだけ入っていればよい。
sherpa-onnx のモデルは model_dir を省略すると models/ に自動でダウンロードする。
"""

from __future__ import annotations

import re
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .models import ensure_sherpa_model

REPO = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    """apps/asr_engines.toml（無ければ example）を読む。NeMo の環境など。"""
    for candidate in ([path] if path else [REPO / "apps" / "asr_engines.toml", REPO / "apps" / "asr_engines.example.toml"]):
        if candidate and candidate.is_file():
            return tomllib.loads(candidate.read_text(encoding="utf-8"))
    return {}


@dataclass(frozen=True)
class BackendInfo:
    name: str
    label: str
    runtime: str  # "transformers" / "funasr" / "sherpa-onnx"
    gpu: bool  # GPU で動かせるか（sherpa-onnx の配布物は CPU 用）
    streaming: bool = False  # 真のストリーミング（create_streaming で作る）


BACKENDS: dict[str, BackendInfo] = {b.name: b for b in [
    BackendInfo("nemotron-560ms", "Nemotron 3.5 streaming 0.6B・560 ms（真のストリーミング・CPU）", "sherpa-onnx", False, True),
    BackendInfo("nemotron-80ms", "Nemotron 3.5 streaming 0.6B・80 ms（真のストリーミング・CPU）", "sherpa-onnx", False, True),
    BackendInfo("nemotron-nemo", "Nemotron 3.5 streaming 0.6B・NeMo（真のストリーミング・GPU・専用の環境）", "nemo", False, True),
    BackendInfo("sherpa-sensevoice", "SenseVoiceSmall（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-parakeet-ctc", "Parakeet TDT-CTC 0.6B 日本語（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-reazonspeech-k2", "ReazonSpeech k2-v2（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-cohere", "Cohere ASR 日本語（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-funasr-nano", "Fun-ASR-Nano（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-qwen3-asr", "Qwen3-ASR 0.6B（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("faster-whisper", "Whisper large-v3-turbo（faster-whisper）", "faster-whisper", True),
    BackendInfo("qwen3-asr-0.6b", "Qwen3-ASR 0.6B（qwen-asr・PyTorch）", "qwen-asr", True),
    BackendInfo("qwen3-asr-1.7b", "Qwen3-ASR 1.7B（qwen-asr・PyTorch）", "qwen-asr", True),
    BackendInfo("moonshine-tiny-ja", "Moonshine tiny-ja（transformers）", "transformers", True),
    BackendInfo("granite", "Granite Speech 5.0 470M TurboCTC（transformers）", "transformers", True),
    BackendInfo("whisper-large-v3-turbo", "Whisper large-v3-turbo（transformers）", "transformers", True),
    BackendInfo("kotoba-whisper-v2.0", "Kotoba-Whisper v2.0（transformers）", "transformers", True),
    BackendInfo("fun-asr-nano", "Fun-ASR-Nano-2512（funasr）", "funasr", True),
    BackendInfo("fun-asr-mlt-nano", "Fun-ASR-MLT-Nano-2512（funasr）", "funasr", True),
]}


class GraniteCTC:
    """IBM Granite Speech 5.0 TurboCTC（transformers>=5.16、AutoModelForCTC）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", device=self.device)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


class Moonshine:
    """Moonshine（transformers の MoonshineForConditionalGeneration）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        self.torch = torch
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_id).to(device, self.dtype).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device, self.dtype)
        # 公式の例と同じく、音声 1 秒あたり 13 トークンを上限にする（繰り返しの暴走を防ぐ）
        max_length = int((inputs.attention_mask.sum(dim=-1) * (13 / sr)).max().item())
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_length=max(8, max_length))
        return self.processor.decode(ids[0], skip_special_tokens=True)


class FunASRNano:
    """Fun-ASR-Nano / Fun-ASR-MLT-Nano（funasr の AutoModel）。"""

    def __init__(self, model_id: str, device: str, language: str = "日文") -> None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(model_id))
        kwargs = {"trust_remote_code": True, "device": device, "disable_update": True}
        if (local / "model.py").is_file():
            kwargs["remote_code"] = str(local / "model.py")
        self.model = AutoModel(model=str(local), **kwargs)
        self.language = language

    def __call__(self, audio: np.ndarray) -> str:
        import torch

        # funasr の Fun-ASR-Nano は音声を torch.Tensor かファイルパスでしか受け取らない
        res = self.model.generate(
            input=[torch.from_numpy(np.ascontiguousarray(audio))], cache={}, batch_size=1,
            language=self.language, itn=True, llm_kwargs={"do_sample": False},
        )
        text = str(res[0].get("text", "")) if res else ""
        return re.sub(r"<\|[^|]+\|>", "", text)


class WhisperHF:
    """Whisper 系（transformers の pipeline。日本語・書き起こしを指定）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype, device=device)

    def __call__(self, audio: np.ndarray) -> str:
        out = self.pipe({"raw": audio, "sampling_rate": 16000},
                        generate_kwargs={"language": "ja", "task": "transcribe"})
        return str(out.get("text", ""))


class Sherpa:
    """sherpa-onnx のオフライン認識（k2-fsa の配布物。CPU）。"""

    def __init__(self, kind: str, model_dir: Path | None, threads: int) -> None:
        import sherpa_onnx

        d = Path(model_dir) if model_dir else ensure_sherpa_model(kind)
        R = sherpa_onnx.OfflineRecognizer
        if kind == "sensevoice":
            self.rec = R.from_sense_voice(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                          num_threads=threads, language="ja", use_itn=True)
        elif kind == "parakeet-ctc":
            self.rec = R.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                       num_threads=threads)
        elif kind == "reazonspeech-k2":
            self.rec = R.from_transducer(
                encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"), decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
                joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads)
        elif kind == "cohere":
            self.rec = R.from_cohere_transcribe(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                                                tokens=str(d / "tokens.txt"), num_threads=threads, language="ja",
                                                use_punct=True, use_itn=True)
        elif kind == "qwen3-asr":
            # int8 は CUDA に載せても MatMulInteger が CPU に落ちるので、CPU のまま使う
            self.rec = R.from_qwen3_asr(conv_frontend=str(d / "conv_frontend.onnx"), encoder=str(d / "encoder.int8.onnx"),
                                        decoder=str(d / "decoder.int8.onnx"), tokenizer=str(d / "tokenizer"),
                                        num_threads=threads, sample_rate=16000, feature_dim=128)
        elif kind == "funasr-nano":
            self.rec = R.from_funasr_nano(encoder_adaptor=str(d / "encoder_adaptor.int8.onnx"), llm=str(d / "llm.int8.onnx"),
                                          embedding=str(d / "embedding.int8.onnx"), tokenizer=str(d / "Qwen3-0.6B"),
                                          num_threads=threads, language="ja", itn=True, max_new_tokens=128)
        else:
            raise ValueError(kind)

    def __call__(self, audio: np.ndarray) -> str:
        stream = self.rec.create_stream()
        stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        self.rec.decode_stream(stream)
        return re.sub(r"<\|[^|]+\|>", "", stream.result.text)


class FasterWhisper:
    """faster-whisper（CTranslate2）。GPU は float16、CPU は int8。"""

    def __init__(self, model_id: str, device: str) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_id, device="cuda" if device.startswith("cuda") else "cpu",
                                  compute_type="float16" if device.startswith("cuda") else "int8")

    def __call__(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(np.ascontiguousarray(audio, dtype=np.float32), language="ja",
                                                beam_size=1, vad_filter=False, condition_on_previous_text=False)
        return "".join(s.text for s in segments)


class Qwen3ASRTorch:
    """Qwen3-ASR（qwen-asr パッケージの transformers 実装）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        cuda = device.startswith("cuda")
        self.model = Qwen3ASRModel.from_pretrained(model_id, dtype=torch.bfloat16 if cuda else torch.float32,
                                                   device_map="cuda:0" if cuda else "cpu",
                                                   max_inference_batch_size=1, max_new_tokens=256)

    def __call__(self, audio: np.ndarray) -> str:
        results = self.model.transcribe(audio=(np.ascontiguousarray(audio, dtype=np.float32), 16000), language="Japanese")
        return str(results[0].text) if results else ""


class NemoNemotron:
    """Nemotron 3.5 streaming を NeMo（PyTorch）で動かす。MaAI-SDS と同じく専用の環境のサーバーに TCP でつなぐ。

    NeMo は依存が重く nemo-toolkit==3.0.0 に固定が要るので、別の環境（apps/asr_engines.toml の
    [nemotron_nemo] python）で asrkit/nemo/nemotron_asr_server.py を起動する。
    使い方は NemotronStreaming と同じ（start / feed / finish）。
    """

    def __init__(self, cfg: dict) -> None:
        from .nemo.nemotron_asr import encode_audio_f32, recv_frame, send_frame

        self._enc, self._recv, self._send = encode_audio_f32, recv_frame, send_frame
        python = str(cfg.get("python", "")).strip()
        python_path = Path(python) if Path(python).is_absolute() else REPO / python
        if not python or not python_path.is_file():
            raise RuntimeError("apps/asr_engines.toml の [nemotron_nemo] python に、nemo-toolkit を入れた環境の python を書いてください")
        self.chunk_ms = int(cfg.get("chunk_ms", 560))
        with socket.socket() as s:  # 空いているポート
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = Path(__file__).resolve().parent / "nemo" / "nemotron_asr_server.py"
        self.proc = subprocess.Popen(
            [str(python_path), str(server), "--port", str(port), "--chunk-size", str(self.chunk_ms), "--lang", "ja-JP",
             "--device", str(cfg.get("device", "cuda"))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + float(cfg.get("load_timeout_sec", 300))
        while True:  # モデルを読み込み終わると READY を出して待ち受ける
            if time.monotonic() > deadline:
                self.proc.terminate()
                raise RuntimeError("NeMo のサーバーの起動がタイムアウトしました")
            line = self.proc.stdout.readline()
            if not line and self.proc.poll() is not None:
                raise RuntimeError(f"NeMo のサーバーが終了しました（exit {self.proc.returncode}）")
            if line.strip() == "READY":
                break
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=60)
        self.stream = None

    def _call(self, req: dict) -> dict:
        self._send(self.sock, req)
        res = self._recv(self.sock)
        if "error" in res:
            raise RuntimeError(res["error"])
        return res

    def start(self) -> None:
        self._call({"op": "reset"})
        self.stream = True

    def feed(self, audio: np.ndarray) -> str:
        return str(self._call({"op": "feed", "audio_b64": self._enc(np.asarray(audio, dtype=np.float32))}).get("text", ""))

    def finish(self) -> str:
        text = str(self._call({"op": "finalize"}).get("text", ""))
        self.stream = None
        return text

    def __call__(self, audio: np.ndarray) -> str:
        self.start()
        self.feed(audio)
        return self.finish()

    def close(self) -> None:
        try:
            self.sock.close()
        finally:
            if self.proc.poll() is None:
                self.proc.terminate()


class NemotronStreaming:
    """Nemotron 3.5 streaming（sherpa-onnx の OnlineRecognizer）。音声を少しずつ入れて内部状態を引き継ぐ。

        s = NemotronStreaming(560)
        s.start()
        for chunk in mic: text = s.feed(chunk)   # 途中経過
        text = s.finish()                        # 確定
    PyPI の sherpa-onnx 2.0.1 で、MaAI-SDS のカスタムビルドと同じくらいの精度（JSUT 20 発話で確認）。
    """

    def __init__(self, chunk_ms: int = 560, threads: int = 4, model_dir: Path | None = None) -> None:
        import sherpa_onnx

        d = Path(model_dir) if model_dir else ensure_sherpa_model(f"nemotron-{chunk_ms}ms")
        self.chunk_ms = chunk_ms
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), num_threads=threads, sample_rate=16000, feature_dim=128,
            decoding_method="greedy_search", dither=0.0)
        self.stream = None

    def start(self) -> None:
        self.stream = self.rec.create_stream()
        self.stream.set_option("language", "ja")

    def feed(self, audio: np.ndarray) -> str:
        self.stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return str(self.rec.get_result(self.stream)).strip()

    def finish(self) -> str:
        # 最後のチャンクを押し出すため、チャンク 1 つ分と少しの無音を足してから終える
        self.stream.accept_waveform(16000, np.zeros(int(16000 * (self.chunk_ms / 1000 + 0.2)), dtype=np.float32))
        self.stream.input_finished()
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        text = str(self.rec.get_result(self.stream)).strip()
        self.stream = None
        return text

    def __call__(self, audio: np.ndarray) -> str:
        """一括で認識する（ほかのモデルと同じ呼び方）。"""
        self.start()
        self.feed(audio)
        return self.finish()


def create_streaming(name: str, threads: int = 4, model_dir: Path | None = None, config: dict | None = None):
    if name == "nemotron-nemo":
        return NemoNemotron((config if config is not None else load_config()).get("nemotron_nemo", {}))
    chunk = {"nemotron-560ms": 560, "nemotron-80ms": 80}[name]
    return NemotronStreaming(chunk, threads, model_dir)


def create(name: str, device: str = "cpu", threads: int = 8, model_dir: Path | None = None) -> Callable[[np.ndarray], str]:
    """名前からモデルを読み込み、audio → text の関数を返す。"""
    table = {
        "granite": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc", device),
        "granite-nc": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc-nc", device),
        "moonshine-tiny-ja": lambda: Moonshine("UsefulSensors/moonshine-tiny-ja", device),
        "fun-asr-mlt-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-MLT-Nano-2512", device),
        "fun-asr-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-Nano-2512", device),
        "whisper-large-v3-turbo": lambda: WhisperHF("openai/whisper-large-v3-turbo", device),
        "kotoba-whisper-v2.0": lambda: WhisperHF("kotoba-tech/kotoba-whisper-v2.0", device),
        "faster-whisper": lambda: FasterWhisper("deepdml/faster-whisper-large-v3-turbo-ct2", device),
        "qwen3-asr-0.6b": lambda: Qwen3ASRTorch("Qwen/Qwen3-ASR-0.6B", device),
        "qwen3-asr-1.7b": lambda: Qwen3ASRTorch("Qwen/Qwen3-ASR-1.7B", device),
    }
    if name in table:
        return table[name]()
    if name.startswith("sherpa-"):
        return Sherpa(name[len("sherpa-"):], model_dir, threads)
    if name in BACKENDS and BACKENDS[name].streaming:
        return create_streaming(name, threads, model_dir)
    raise ValueError(f"unknown backend: {name}")
