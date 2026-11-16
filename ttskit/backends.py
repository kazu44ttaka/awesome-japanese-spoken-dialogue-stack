"""日本語 TTS の各エンジンを同じ形（文字列 → (波形 float32, サンプリング周波数)）で呼べるようにする。

    from ttskit.backends import BACKENDS, create, load_config
    tts = create("aivisspeech", load_config())
    voices = tts.voices()                  # [(id, 表示名), ...]
    audio, rate = tts.synthesize("こんにちは。", voice=voices[0][0])

エンジンごとの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピーする）。必要なパッケージはエンジンごとに違う（README 参照）。
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    label: str
    note: str


BACKENDS: dict[str, BackendInfo] = {b.name: b for b in [
    BackendInfo("aivisspeech", "AivisSpeech（エンジンに HTTP で接続）", "AivisSpeech を入れて起動しておく（または「エンジンを起動」）"),
    BackendInfo("voicevox", "VOICEVOX（エンジンに HTTP で接続）", "VOICEVOX を入れて起動しておく（または「エンジンを起動」）"),
    BackendInfo("voicevox_core", "VOICEVOX Core（アプリの中で動かす）", "voicevox_core の wheel と、ダウンローダーで取った一式（runtime_dir）"),
    BackendInfo("irodori", "Irodori-TTS v4.1 Small（GPU）", "irodori-tts（README の手順）。声は文章で指定する"),
    BackendInfo("supertonic", "Supertonic 3（ONNX・CPU）", "pip install supertonic"),
    BackendInfo("mio", "mio-tts-cpp（サーバーに HTTP で接続・CPU/GPU）", "mio-tts-server をビルドしておく（または起動済み）"),
    BackendInfo("qwen3_tts_cpp", "Qwen3-TTS 1.7B（audio.cpp のサーバー・GPU）", "audiocpp_server をビルドし、GGUF を置く"),
    BackendInfo("qwen3-tts", "Qwen3-TTS 0.6B（PyTorch・GPU）", "pip install qwen-tts（別の環境を推奨）"),
    BackendInfo("tsukasa", "Tsukasa Speech（GPU・非商用）", "Tsukasa のフォルダーと参照音声を tts_engines.toml に書く"),
    BackendInfo("kokoro", "Kokoro 82M（比較用・日本語は崩れやすい）", "pip install kokoro \"misaki[ja]\""),
    BackendInfo("zonos3", "ZONOS3-Mini（C++ サーバー・GPU）", "zonos3-server をビルドし、GGUF を置く（apps/README.md）"),
    BackendInfo("kitten", "Kitten TTS 2（別の環境）", "pip install kittenml（別の環境 .venv-kitten）。商用は登録が要る"),
    BackendInfo("lfm_audio", "LFM2.5-Audio-1.5B-JP（別の環境・GPU）", "pip install liquid-audio（別の環境 .venv-lfm）"),
    BackendInfo("moss_nano", "MOSS-TTS-Nano 0.1B（別の環境・CPU）", "MOSS-TTS-Nano を clone して入れる（別の環境 .venv-nano）"),
]}


def load_config(path: Path | None = None) -> dict:
    """apps/tts_engines.toml（無ければ example）を読む。"""
    for candidate in ([path] if path else [REPO / "apps" / "tts_engines.toml", REPO / "apps" / "tts_engines.example.toml"]):
        if candidate and candidate.is_file():
            return tomllib.loads(candidate.read_text(encoding="utf-8"))
    return {}


def _path(value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else REPO / p


def split_sentences(text: str) -> list[str]:
    """句点・感嘆符・疑問符・改行で文に分ける（文ごとに合成して、最初の文から再生するため）。"""
    parts = re.split(r"(?<=[。！？!?])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _http(url: str, data: bytes | None = None, timeout: float = 120.0, method: str | None = None) -> bytes:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class _Server:
    """設定に書いた実行ファイルを、つながらないときだけ起動して待つ（アプリ終了時に止める）。"""

    def __init__(self, health_url: str) -> None:
        self.health_url = health_url
        self.process: subprocess.Popen | None = None

    def alive(self) -> bool:
        try:
            _http(self.health_url, timeout=2.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def ensure(self, args: list[str], cwd: Path, dll_dirs: list[str] | None = None, wait_sec: float = 180.0) -> None:
        if self.alive():
            return
        exe = Path(args[0])
        if not exe.is_file():
            raise RuntimeError(f"サーバーにつながらず、実行ファイルもありません: {exe}")
        # ビルドによっては DLL（llama.dll・ggml*.dll・CUDA ランタイム）が bin/ などにあるので PATH に足す
        dirs = [exe.parent, exe.parent / "bin", *(_path(d) for d in (dll_dirs or []))]
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(d) for d in dirs if d.is_dir()] + [env.get("PATH", "")])
        self.process = subprocess.Popen(args, cwd=str(cwd), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"サーバーが終了しました（exit {self.process.returncode}）")
            if self.alive():
                return
            time.sleep(0.5)
        raise RuntimeError("サーバーの起動がタイムアウトしました")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.process = None


# ---------------------------------------------------------------------------- HTTP engines
class VoicevoxCompatible:
    """VOICEVOX 互換の HTTP API（AivisSpeech も同じ形）。"""

    def __init__(self, base_url: str, exe_candidates: list[Path]) -> None:
        self.base_url = base_url.rstrip("/")
        self.exe_candidates = exe_candidates

    def available(self) -> bool:
        try:
            _http(self.base_url + "/version", timeout=1.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def start_engine(self, use_gpu: bool = True, wait_sec: float = 90.0) -> bool:
        """既定のインストール先のエンジンを起動し、応答するまで待つ（use_gpu で --use_gpu を付ける）。"""
        if self.available():
            return True
        exe = next((p for p in self.exe_candidates if p.is_file()), None)
        if exe is None:
            return False
        subprocess.Popen([str(exe), *(["--use_gpu"] if use_gpu else [])], cwd=str(exe.parent),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if self.available():
                return True
            time.sleep(1.0)
        return False

    def voices(self) -> list[tuple[str, str]]:
        out = []
        for sp in json.loads(_http(self.base_url + "/speakers", timeout=10)):
            for st in sp.get("styles", []):
                out.append((str(st["id"]), f"{sp.get('name', '')}（{st.get('name', '')}）"))
        return out

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        q = urllib.parse.urlencode({"text": text, "speaker": voice})
        query = json.loads(_http(f"{self.base_url}/audio_query?{q}", data=b"", method="POST"))
        query["speedScale"] = float(speed)
        wav = _http(f"{self.base_url}/synthesis?speaker={urllib.parse.quote(voice)}", json.dumps(query).encode("utf-8"))
        return _read_wav(wav)


class Mio:
    """mio-tts-cpp のサーバー（POST /mio/tts。結果の WAV はサーバーがファイルに書く）。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.url = str(cfg.get("url", "http://127.0.0.1:18089")).rstrip("/")
        self.server = _Server(self.url + "/health")

    def available(self) -> bool:
        if not self.server.alive() and self.cfg.get("server_exe"):
            exe = _path(self.cfg["server_exe"])
            refs = [{"key": k, "path": str(_path(v))} for k, v in self.cfg.get("references", {}).items()]
            port = urllib.parse.urlparse(self.url).port or 18089
            self.server.ensure([str(exe), "-m", str(_path(self.cfg["model"])), "-mv", str(_path(self.cfg["vocoder"])),
                                "--tts-wavlm-model", str(_path(self.cfg["wavlm"])), "--port", str(port),
                                # 既定の 768 では長い文（参照 + 生成 800 トークン）が入らず 400 になる
                                "--ctx-size", str(int(self.cfg.get("ctx_size", 2048))),
                                "--reference-file-json", json.dumps(refs)], cwd=exe.parent,
                               dll_dirs=self.cfg.get("dll_dirs"))
        return self.server.alive()

    def voices(self) -> list[tuple[str, str]]:
        try:
            keys = json.loads(_http(self.url + "/mio/references", timeout=5))
            keys = keys.get("references", keys) if isinstance(keys, dict) else keys
            keys = [k.get("key", k) if isinstance(k, dict) else k for k in keys]
        except Exception:  # noqa: BLE001
            keys = list(self.cfg.get("references", {}))
        return [(str(k), str(k)) for k in keys] or [("jp_female", "jp_female")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        out = Path(tempfile.gettempdir()) / f"mio-tts-{os.getpid()}-{threading.get_ident()}.wav"
        body = {"text": text, "reference_key": voice, "n_predict": 800, "top_k": 50, "top_p": 1.0, "temp": 0.8,
                "repeat_penalty": 1.0, "output_file": str(out)}
        _http(self.url + "/mio/tts", json.dumps(body, ensure_ascii=False).encode("utf-8"))
        data = out.read_bytes()
        out.unlink(missing_ok=True)
        return _read_wav(data)

    def close(self) -> None:
        self.server.close()


class Qwen3TTSCpp:
    """Qwen3-TTS CustomVoice を audio.cpp のサーバーで動かす（POST /v1/audio/speech）。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.url = str(cfg.get("url", "http://127.0.0.1:18081")).rstrip("/")
        self.server = _Server(self.url + "/health")
        self.model_id = "qwen3-tts-customvoice"

    def available(self) -> bool:
        if not self.server.alive() and self.cfg.get("server_exe"):
            exe = _path(self.cfg["server_exe"])
            port = urllib.parse.urlparse(self.url).port or 18081
            config = {"host": "127.0.0.1", "port": port, "backend": "cuda", "device": 0, "threads": 8, "lazy_load": False,
                      "models": [{"id": self.model_id, "family": "qwen3_tts", "path": str(_path(self.cfg["model"])),
                                  "task": "tts", "mode": "offline"}]}
            cfg_file = Path(tempfile.gettempdir()) / f"qwen3-tts-cpp-{port}.json"
            cfg_file.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            self.server.ensure([str(exe), "--config", str(cfg_file)], cwd=exe.parent, dll_dirs=self.cfg.get("dll_dirs"))
        return self.server.alive()

    def voices(self) -> list[tuple[str, str]]:
        return [(v, v + ("（日本語）" if v == "Ono_Anna" else "")) for v in self.cfg.get("voices", ["Ono_Anna"])]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        options = {"speaker": voice, "instruct": "", "language": "japanese", "do_sample": True, "temperature": 0.9,
                   "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05}
        body = {"model": self.model_id, "input": text, "voice": voice, "response_format": "wav", "options": options}
        return _read_wav(_http(self.url + "/v1/audio/speech", json.dumps(body, ensure_ascii=False).encode("utf-8")))

    def close(self) -> None:
        self.server.close()


# ---------------------------------------------------------------------------- in-process engines
class VoicevoxCore:
    """VOICEVOX Core（voicevox_core の blocking API）。声は使うときにその VVM だけ読み込む。"""

    def __init__(self, cfg: dict) -> None:
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

        root = _path(cfg.get("runtime_dir", "voicevox_runtime"))
        ort_name = "voicevox_onnxruntime.dll" if sys.platform == "win32" else "libvoicevox_onnxruntime.so"
        ort_file = next(iter(sorted(root.glob(f"onnxruntime/**/{ort_name}*"))), None)
        if ort_file is None:
            raise RuntimeError(f"{root} に onnxruntime が見つかりません（ダウンローダーで取った一式を置く）")
        dict_dir = next(iter(sorted(root.glob("dict/open_jtalk_dic_utf_8-*"))), None)
        mode = str(cfg.get("acceleration", "CPU")).upper()
        if mode in ("GPU", "AUTO"):
            # CUDA 版の ONNX Runtime は cuDNN 9・cuBLAS などの DLL を要る。PyTorch（CUDA 版）が入っていればその DLL を使う
            dll_dirs = [_path(d) for d in cfg.get("dll_dirs", [])]
            try:
                import torch

                dll_dirs.append(Path(torch.__file__).parent / "lib")
            except ImportError:
                pass
            for d in dll_dirs:
                if d.is_dir():
                    os.add_dll_directory(str(d))
                    os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        ort = Onnxruntime.load_once(filename=str(ort_file))
        # cpu_num_threads を省くと遅い（24 コアの PC で 8 スレッドの約半分の速さだった）
        self.synth = Synthesizer(ort, OpenJtalk(str(dict_dir)), acceleration_mode=mode,
                                 cpu_num_threads=int(cfg.get("cpu_threads", 8)))
        self.VoiceModelFile = VoiceModelFile
        self.style_to_vvm: dict[int, Path] = {}
        self.names: list[tuple[str, str]] = []
        for vvm in sorted((root / "models" / "vvms").glob("*.vvm"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            with VoiceModelFile.open(str(vvm)) as model:
                for character in model.metas:
                    for style in character.styles:
                        self.style_to_vvm[style.id] = vvm
                        self.names.append((str(style.id), f"{character.name}（{style.name}）"))
        self.loaded: set[Path] = set()

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return self.names

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        style = int(voice)
        vvm = self.style_to_vvm[style]
        if vvm not in self.loaded:
            with self.VoiceModelFile.open(str(vvm)) as model:
                self.synth.load_voice_model(model)
            self.loaded.add(vvm)
        query = self.synth.create_audio_query(text, style)
        query.speed_scale = float(speed)
        query.pre_phoneme_length = 0.0  # 先頭の無音（既定 0.1 秒）を入れない
        return _read_wav(bytes(self.synth.synthesis(query, style)))


class Irodori:
    """Irodori-TTS v4.1（irodori_tts の InferenceRuntime。参照音声なしで、声はキャプションで指定）。"""

    def __init__(self, cfg: dict) -> None:
        import torch
        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint

        device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        precision = "bf16" if device.startswith("cuda") else "fp32"
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")
        self.SamplingRequest = SamplingRequest
        self.steps = int(cfg.get("steps", 40))
        self.captions = list(cfg.get("captions", ["明るく自然な日本語の声"]))
        self.runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=download_hf_checkpoint(str(cfg.get("checkpoint", "Aratako/Irodori-TTS-v4.1-Small"))),
            model_device=device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim", model_precision=precision,
            codec_device=device, codec_precision=precision, codec_deterministic_encode=True,
            codec_deterministic_decode=True, compile_model=False, compile_dynamic=False,
        ))

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [(c, c) for c in self.captions]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        req = self.SamplingRequest(text=text, caption=voice, num_steps=self.steps, cfg_scale_text=3.0,
                                   cfg_scale_caption=3.0, cfg_scale_speaker=5.0, seed=42, no_ref=True)
        result = self.runtime.synthesize(req, log_fn=None)
        return result.audio.detach().float().cpu().numpy().reshape(-1), int(result.sample_rate)


class Supertonic:
    """Supertonic 3（ONNX・CPU。リポジトリは 2026-07 にアーカイブ済み）。"""

    def __init__(self, cfg: dict | None = None) -> None:
        from supertonic import TTS

        self.tts = TTS(model="supertonic-3", auto_download=True)
        self._styles: dict[str, object] = {}

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [(v, f"{'女性' if v.startswith('F') else '男性'} {v}") for v in
                ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        style = self._styles.get(voice) or self.tts.get_voice_style(voice)
        self._styles[voice] = style
        wav, _dur = self.tts.synthesize(text=text, voice_style=style, lang="ja", total_steps=8, speed=1.05 * speed)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(self.tts.sample_rate)


class Qwen3TTS:
    """Qwen3-TTS 0.6B CustomVoice（PyTorch。日本語の声は Ono_Anna）。"""

    MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def __init__(self, cfg: dict | None = None) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model = Qwen3TTSModel.from_pretrained(
            self.MODEL, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [("Ono_Anna", "Ono_Anna（日本語）")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        wavs, rate = self.model.generate_custom_voice(
            text=text, language="Japanese", speaker=voice, instruct="",
            max_new_tokens=min(1024, max(256, len(text) * 6 + 120)), do_sample=True, temperature=0.9,
        )
        return np.asarray(wavs[0], dtype=np.float32), int(rate)


class Tsukasa:
    """Tsukasa Speech（Respair/Tsukasa_Speech のフォルダーを読み込む。参照音声の声で話す）。"""

    def __init__(self, cfg: dict) -> None:
        self.root = _path(cfg.get("root", "Tsukasa-HF"))
        self.reference = str(cfg.get("reference_audio", "")).strip()
        if not self.root.is_dir():
            raise RuntimeError(f"Tsukasa のフォルダーがありません: {self.root}")
        if not self.reference or not _path(self.reference).is_file():
            raise RuntimeError("tts_engines.toml の [tsukasa] reference_audio に、使ってよい参照音声の WAV を書いてください")
        import transformers

        if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
            transformers.PreTrainedModel.all_tied_weights_keys = {}  # 新しい transformers との互換
        old = Path.cwd()
        sys.path.insert(0, str(self.root))
        try:
            os.chdir(self.root)  # importable.py は設定ファイルを今いるフォルダーから読む
            import importable
            from Utils.phonemize import mixed_phon
        finally:
            os.chdir(old)
        self.module, self.phonemizer = importable, mixed_phon
        self.style = self.module.compute_style_through_clip(str(_path(self.reference)))

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [("ref", f"参照音声: {Path(self.reference).name}")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        phonemes = self.phonemizer.smart_phonemize(text)  # 生の日本語を渡すと波形が崩れる
        wav = self.module.inference(phonemes, self.style, alpha=0.3, beta=0.7, diffusion_steps=5,
                                    embedding_scale=1, rate_of_speech=float(speed))
        return np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1)), 24000


class Kokoro:
    """Kokoro 82M（日本語は misaki の音素化。読みが崩れやすいので比較用）。"""

    def __init__(self, cfg: dict) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code="j")
        self.voice_names = list(cfg.get("voices", ["jf_alpha"]))

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [(v, v) for v in self.voice_names]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        chunks = [np.asarray(audio, dtype=np.float32) for _g, _p, audio in self.pipeline(text, voice=voice, speed=speed)]
        return (np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)), 24000


class Zonos3:
    """Zyphra ZONOS3-Mini を C++/ggml のサーバー（zonos3-server）で動かす（POST /v2/generate）。

    コマンドラインの zonos3-cli は Windows で日本語の引数が文字化けするので、UTF-8 の JSON を送れるサーバーを使う。
    日本語の既定の声は無いので、ほかの言語の既定の声で日本語を読ませる（参照音声で声を変えることもできる）。
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.url = str(cfg.get("url", "http://127.0.0.1:1919")).rstrip("/")
        self.server = _Server(self.url + "/v2/capabilities")
        self._voices: list[tuple[str, str]] | None = None

    def available(self) -> bool:
        if not self.server.alive() and self.cfg.get("server_exe"):
            exe = _path(self.cfg["server_exe"])
            model = _path(self.cfg.get("model", "models/src/ZONOS3-Mini/servers/ggml/gguf/backbone-ptq.gguf"))
            port = urllib.parse.urlparse(self.url).port or 1919
            args = [str(exe), str(model), "--gpu" if self.cfg.get("gpu", True) else "--cpu", "--port", str(port)]
            if self.cfg.get("voices_dir"):
                args += ["--tts-default-voices-dir", str(_path(self.cfg["voices_dir"]))]
            self.server.ensure(args, cwd=model.parent.parent, dll_dirs=self.cfg.get("dll_dirs"))
        return self.server.alive()

    def voices(self) -> list[tuple[str, str]]:
        if self._voices is None:
            speakers = json.loads(_http(self.url + "/v2/speakers"))["speakers"]
            self._voices = [("", "既定の声")] + [(s["id"], s["label"]) for s in speakers]
        return self._voices

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        body = {"text": text, "language": "ja", "stream": False, "format": "wav", "seed": int(self.cfg.get("seed", 1))}
        if voice:
            body["speaker_embedding_id"] = voice
        return _read_wav(_http(self.url + "/v2/generate", json.dumps(body, ensure_ascii=False).encode("utf-8")))

    def close(self) -> None:
        self.server.close()


# ---------------------------------------------------------------------------- 別の環境で動かすエンジン
class Kitten:
    """KittenML Kitten TTS 2（kittenml）。日本語の声が 4 つある。商用は Stellon Labs への登録が要る。"""

    def __init__(self, cfg: dict) -> None:
        from kittenml import KittenTTS

        self.model = KittenTTS(str(cfg.get("model", "KittenML/kitten-tts-2")), device=str(cfg.get("device", "cuda")))

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [(v, v) for v in self.model.available_voices if v.startswith("Japanese")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        # 英語以外は normalize=False（英語向けの数字の読み替えをしない）
        audio = self.model.generate(text, voice=voice or "JapaneseGrandma", normalize=False)
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(self.model.sample_rate)


class LfmAudio:
    """Liquid AI LFM2.5-Audio-1.5B-JP の TTS（system に「Perform TTS in japanese.」）。声は選べない。"""

    def __init__(self, cfg: dict) -> None:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        repo = str(cfg.get("model", "LiquidAI/LFM2.5-Audio-1.5B-JP"))
        self.processor = LFM2AudioProcessor.from_pretrained(repo).eval()
        self.model = LFM2AudioModel.from_pretrained(repo).eval()

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [("", "既定の声（女性）")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import torch
        from liquid_audio import ChatState

        chat = ChatState(self.processor)
        chat.new_turn("system")
        chat.add_text("Perform TTS in japanese.")
        chat.end_turn()
        chat.new_turn("user")
        chat.add_text(text)
        chat.end_turn()
        chat.new_turn("assistant")
        frames = [t for t in self.model.generate_sequential(**chat, max_new_tokens=1024, audio_temperature=0.8,
                                                            audio_top_k=64) if t.numel() > 1]
        if len(frames) < 2:
            return np.zeros(0, dtype=np.float32), 24000
        wav = self.processor.decode(torch.stack(frames[:-1], 1).unsqueeze(0))  # 最後のフレームは終わりの印
        return wav.cpu()[0].float().numpy().reshape(-1), 24000


class MossNano:
    """OpenMOSS MOSS-TTS-Nano（0.1B・CPU で動く）。日本語の声は配布物の jp_2.wav（Yui）。

    Windows では torchaudio 2.9 以降の load / save が FFmpeg の DLL を要る torchcodec を使うので、soundfile に置き換える。
    文章の正規化（WeTextProcessing、Windows で入れにくい）は使わない。
    """

    def __init__(self, cfg: dict) -> None:
        import soundfile as sf
        import torch
        import torchaudio

        def _load(path, *args, **kwargs):
            x, rate = sf.read(str(path), dtype="float32", always_2d=True)
            return torch.from_numpy(x.T.copy()), rate

        def _save(path, wav, rate, *args, **kwargs):
            sf.write(str(path), wav.detach().cpu().float().numpy().T, rate)

        torchaudio.load, torchaudio.save = _load, _save
        sys.path.insert(0, str(_path(cfg.get("root", "models/src/MOSS-TTS-Nano"))))
        from moss_tts_nano_runtime import NanoTTSService

        self.service = NanoTTSService(device=str(cfg.get("device", "cpu")),
                                      output_dir=Path(tempfile.gettempdir()) / "moss-tts-nano")

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        presets = self.service.voice_presets
        found = [(k, f"{k}（{v.description}）") for k, v in presets.items()
                 if v.prompt_audio_path.is_file() and "Japanese" in v.description]
        return found or [("Yui", "Yui")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        result = self.service.synthesize(text=text, voice=voice or "Yui")
        return np.asarray(result["waveform_numpy"], dtype=np.float32).reshape(-1), int(result["sample_rate"])


class VenvEngine:
    """エンジンを別の Python 環境の子プロセス（ttskit.worker）で動かす。設定の python にその環境の python を書く。"""

    def __init__(self, name: str, cfg: dict) -> None:
        self.name, self.cfg = name, cfg
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        env = dict(os.environ, TTSKIT_CONFIG=json.dumps(self.cfg, ensure_ascii=False), PYTHONIOENCODING="utf-8",
                   HF_HUB_DISABLE_PROGRESS_BARS="1")
        # 親の環境の Python を指す変数を引き継ぐと、版の違う子の Python が親の標準ライブラリを読んで落ちる
        for key in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)
        if self.cfg.get("hf_cache"):
            env["HF_HUB_CACHE"] = str(_path(self.cfg["hf_cache"]))
        self.process = subprocess.Popen([str(_path(self.cfg["python"])), "-m", "ttskit.worker", self.name],
                                        cwd=str(REPO), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        first = json.loads(self.process.stdout.readline() or '{"error": "子プロセスが終了しました"}')
        if "error" in first:
            self.close()
            raise RuntimeError(first["error"])

    def _call(self, req: dict) -> dict:
        with self.lock:
            self._start()
            self.process.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        res = json.loads(line or '{"error": "子プロセスが終了しました"}')
        if "error" in res:
            raise RuntimeError(res["error"])
        return res

    def available(self) -> bool:
        self._start()
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [tuple(v) for v in self._call({"cmd": "voices"})["voices"]]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import base64

        res = self._call({"cmd": "synth", "text": text, "voice": voice, "speed": speed})
        return np.frombuffer(base64.b64decode(res["audio"]), dtype=np.float32).copy(), int(res["rate"])

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.process.kill()
        self.process = None


def _program_files(*parts: str) -> list[Path]:
    local = os.environ.get("LOCALAPPDATA", "")
    return [Path(local, "Programs", *parts)] if local else []


_LOCAL = {"voicevox_core": "VoicevoxCore", "irodori": "Irodori", "supertonic": "Supertonic", "mio": "Mio",
          "qwen3_tts_cpp": "Qwen3TTSCpp", "qwen3-tts": "Qwen3TTS", "tsukasa": "Tsukasa", "kokoro": "Kokoro",
          "zonos3": "Zonos3", "kitten": "Kitten", "lfm_audio": "LfmAudio", "moss_nano": "MossNano"}


def create_local(name: str, cfg: dict):
    """この Python 環境の中でエンジンを作る。"""
    if name == "aivisspeech":
        return VoicevoxCompatible(cfg.get("url", "http://127.0.0.1:10101"),
                                  _program_files("AivisSpeech", "AivisSpeech-Engine", "run.exe"))
    if name == "voicevox":
        return VoicevoxCompatible(cfg.get("url", "http://127.0.0.1:50021"),
                                  _program_files("VOICEVOX", "vv-engine", "run.exe"))
    if name not in _LOCAL:
        raise ValueError(f"unknown backend: {name}")
    return globals()[_LOCAL[name]](cfg)


def create(name: str, config: dict | None = None):
    """設定（apps/tts_engines.toml の節）に python があれば、その環境の子プロセスで動かす。"""
    cfg = (config if config is not None else load_config()).get(name, {})
    if cfg.get("python"):
        return VenvEngine(name, cfg)
    return create_local(name, cfg)
