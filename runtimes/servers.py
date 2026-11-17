"""ggml 系の実行系（CrispASR・speech.cpp・audio.cpp）のサーバーを立ち上げて、OpenAI 互換の API で呼ぶ。

3 つとも `/v1/audio/transcriptions`（ASR）と `/v1/audio/speech`（TTS）を持つので、クライアントは共通。
サーバーはこのリポジトリの中に置いた実行ファイルを使い、モデルも `models/` の下に置く（docs/runtimes.md）。

    from runtimes.servers import create_asr
    asr = create_asr("speechcpp", "haas-ja-v1.0", device="cuda")   # speech.cpp は GPU = Vulkan
    text = asr(audio_16k_float32)
    asr.close()
"""

from __future__ import annotations

import atexit
import io
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CRISPASR_EXE = REPO / "models/src/crispasr-bin/crispasr-windows-x86_64-cuda/crispasr.exe"
SPEECHCPP_EXE = REPO / "models/src/speechcpp-bin/speech.exe"
AUDIOCPP_EXE = REPO / "models/src/audio.cpp/build/windows-cuda-release/bin/audiocpp_server.exe"
CRISPASR_CACHE = REPO / "models/crispasr"
SPEECHCPP_MODELS = REPO / "models/speechcpp"
AUDIOCPP_MODELS = REPO / "models/audiocpp"
SPEECHCPP_GPU = os.environ.get("SPEECHCPP_GPU", "Vulkan1")  # speech devices で見た NVIDIA の名前


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 600.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as ex:
        raise RuntimeError(f"HTTP {ex.code}: {ex.read().decode('utf-8', 'replace')[:500]}") from ex


class Server:
    """実行ファイルを立ち上げ、応答するまで待つ。close で止める。"""

    def __init__(self, args: list[str], health_url: str, cwd: Path | None = None, env: dict | None = None,
                 wait_sec: float = 1800.0, log: Path | None = None) -> None:
        self.health_url = health_url
        self.log_file = open(log, "ab") if log else None  # noqa: SIM115
        out = self.log_file or subprocess.DEVNULL
        self.process = subprocess.Popen(args, cwd=str(cwd or REPO), env={**os.environ, **(env or {})},
                                        stdout=out, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        atexit.register(self.close)  # 呼んだ側が close を忘れても、終わるときにサーバーを止める
        deadline = time.monotonic() + wait_sec  # 初回はモデルのダウンロードを待つので長め
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"サーバーが終了しました（exit {self.process.returncode}）: {' '.join(args[:4])}")
            try:
                _http(health_url, timeout=2.0)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        self.close()
        raise RuntimeError("サーバーの起動がタイムアウトしました")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.log_file:
            self.log_file.close()
            self.log_file = None


def _wav_bytes(audio: np.ndarray, rate: int = 16000) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _multipart(fields: dict[str, str], file_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n".encode() + file_bytes + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAIASR:
    """OpenAI 互換の /v1/audio/transcriptions を呼ぶ（audio → text）。"""

    def __init__(self, base_url: str, model: str, server: Server | None, device: str, language: str = "ja",
                 fields: dict | None = None) -> None:
        self.base_url, self.model, self.server, self.device = base_url, model, server, device
        # speech.cpp はモデルを 1 つだけ載せたサーバーでは model を省ける（名前の大文字小文字が配布名と違うため省く）
        self.fields = {**({"model": model} if model else {}), "response_format": "json",
                       **({"language": language} if language else {}),
                       **(fields or {})}
        self.precision = None

    def __call__(self, audio: np.ndarray) -> str:
        body, ctype = _multipart(self.fields, _wav_bytes(audio))
        res = json.loads(_http(self.base_url + "/v1/audio/transcriptions", body, {"Content-Type": ctype}))
        return str(res.get("text", "")).strip()

    def close(self) -> None:
        if self.server:
            self.server.close()
            self.server = None


class OpenAITTS:
    """OpenAI 互換の /v1/audio/speech を呼ぶ（ttskit のエンジンと同じ形: available / voices / synthesize / close）。"""

    def __init__(self, base_url: str, model: str, server: Server | None, voices: list[tuple[str, str]],
                 options: dict | None = None) -> None:
        self.base_url, self.model, self.server = base_url, model, server
        self._voices = voices
        self.options = options or {}

    def available(self) -> bool:
        return self.server is None or self.server.process.poll() is None

    def voices(self) -> list[tuple[str, str]]:
        return self._voices

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import soundfile as sf

        body = {"model": self.model, "input": text, "response_format": "wav", **self.options}
        if voice:
            body["voice"] = voice
        if speed != 1.0:
            body["speed"] = float(speed)
        wav = _http(self.base_url + "/v1/audio/speech", json.dumps(body, ensure_ascii=False).encode("utf-8"),
                    {"Content-Type": "application/json"})
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(rate)

    def close(self) -> None:
        if self.server:
            self.server.close()
            self.server = None


# ---------------------------------------------------------------------------- 立ち上げ方
def start_crispasr(backend: str, model: str = "auto", device: str = "cuda", threads: int = 8,
                   extra: list[str] | None = None) -> tuple[Server, str]:
    port = free_port()
    args = [str(CRISPASR_EXE), "--server", "--host", "127.0.0.1", "--port", str(port), "--backend", backend,
            "-m", model, "--cache-dir", str(CRISPASR_CACHE), "-t", str(threads), *(extra or [])]
    if not device.startswith("cuda"):
        args.append("-ng")
    url = f"http://127.0.0.1:{port}"
    return Server(args, url + "/health", log=REPO / "data" / "runtime_crispasr.log"), url


def start_speechcpp(models: list[str], device: str = "cuda", threads: int = 8,
                    extra: list[str] | None = None) -> tuple[Server, str]:
    port = free_port()
    dev = SPEECHCPP_GPU if device.startswith("cuda") else "cpu"
    args = [str(SPEECHCPP_EXE), "serve", *models, "--host", "127.0.0.1", "--port", str(port), "--device", dev,
            "--threads", str(threads), *(extra or [])]
    url = f"http://127.0.0.1:{port}"
    return (Server(args, url + "/v1/models", env={"SPEECH_MODEL_DIR": str(SPEECHCPP_MODELS)},
                   log=REPO / "data" / "runtime_speechcpp.log"), url)


def start_audiocpp(models: list[dict], device: str = "cuda", threads: int = 8) -> tuple[Server, str]:
    port = free_port()
    config = {"host": "127.0.0.1", "port": port, "backend": "cuda" if device.startswith("cuda") else "cpu",
              "device": 0, "threads": threads, "lazy_load": False, "models": models}
    cfg = Path(tempfile.gettempdir()) / f"audiocpp-{port}.json"
    cfg.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    url = f"http://127.0.0.1:{port}"
    return (Server([str(AUDIOCPP_EXE), "--config", str(cfg)], url + "/health", cwd=AUDIOCPP_EXE.parents[3],
                   log=REPO / "data" / "runtime_audiocpp.log"), url)


def create_asr(runtime: str, key: str, device: str = "cuda", threads: int = 8, **kw) -> OpenAIASR:
    """runtime: crispasr / speechcpp / audiocpp。key はその実行系でのモデルの名前（runtimes/catalog.py）。"""
    from runtimes.catalog import ASR

    spec = ASR[runtime][key]
    if runtime == "crispasr":
        server, url = start_crispasr(spec["backend"], spec.get("model", "auto"), device, threads, spec.get("args"))
        model = spec["backend"]
    elif runtime == "speechcpp":
        server, url = start_speechcpp([spec["model"]], device, threads, spec.get("args"))
        model = ""
    elif runtime == "audiocpp":
        model = key
        server, url = start_audiocpp([{"id": key, "family": spec["family"], "path": str(AUDIOCPP_MODELS / spec["path"]),
                                       "task": "asr", "mode": "offline", **spec.get("load", {})}], device, threads)
    else:
        raise ValueError(runtime)
    dev = ("vulkan" if runtime == "speechcpp" else "cuda") if device.startswith("cuda") else "cpu"
    return OpenAIASR(url, model, server, dev, language=spec.get("language", "ja"), fields=spec.get("fields"))
