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
import urllib.parse
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
                 wait_sec: float = 5400.0, log: Path | None = None) -> None:
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
        raw = _http(self.base_url + "/v1/audio/transcriptions", body, {"Content-Type": ctype})
        # audio.cpp の Fun-ASR-Nano は、まれに UTF-8 の途中で切れたトークンを返す。止めずに「�」に置き換える
        res = json.loads(raw.decode("utf-8", "replace"))
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
        self.kana = False  # 真なら、文をカタカナの読みに直してから送る（CosyVoice3 は日本語をカタカナで受ける）

    def available(self) -> bool:
        return self.server is None or self.server.process.poll() is None

    def voices(self) -> list[tuple[str, str]]:
        return self._voices

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import soundfile as sf

        if self.kana:
            text = to_kana(text)

        body = {**({"model": self.model} if self.model else {}), "input": text, "response_format": "wav", **self.options}
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
def fetch(url: str, folder: Path) -> Path:
    """1 つのファイルのモデルを取ってくる（あればそのまま使う）。"""
    dest = folder / url.rsplit("/", 1)[-1]
    if not dest.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 22):
                out.write(chunk)
        tmp.replace(dest)
    return dest


def start_crispasr(backend: str, model: str = "auto", device: str = "cuda", threads: int = 8,
                   extra: list[str] | None = None, url: str | None = None, quant: str | None = None,
                   language: str | None = "ja") -> tuple[Server, str]:
    port = free_port()
    if url:  # 登録表の既定（多くは Q4_K）ではなく、決めたファイルを使う
        model = str(fetch(url, CRISPASR_CACHE))
    args = [str(CRISPASR_EXE), "--server", "--host", "127.0.0.1", "--port", str(port), "--backend", backend,
            "-m", model, "--cache-dir", str(CRISPASR_CACHE), "-t", str(threads), *(extra or [])]
    if quant and model == "auto":
        args += ["--model-quant", quant]
    if language:
        args += ["-l", language]
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


def audiocpp_install(package: str) -> Path:
    """audio.cpp のモデル管理でパッケージを models/audiocpp に入れ、モデルのファイル（複数ならフォルダー）を返す。"""
    import sys

    tools = AUDIOCPP_EXE.parents[3] / "tools" / "model_manager_v2.py"
    info = json.loads(subprocess.run([sys.executable, str(tools), "info", package, "--json"], capture_output=True,
                                     text=True, encoding="utf-8", check=True).stdout)
    files = [f if isinstance(f, str) else (f.get("local") or f.get("path")) for f in info.get("files", [])]
    target = AUDIOCPP_MODELS / info["target_directory"]
    # 置き場所は target_directory の下。ファイル名にフォルダーが付いているパッケージと付いていないものがある
    prefix = info["target_directory"] + "/"
    files = [f if f.startswith(prefix) else prefix + f for f in files if f]
    dl = info.get("download") or {}
    if files and dl.get("kind") == "huggingface_snapshot" and not dl.get("gated"):
        # モデル管理の取得（huggingface_hub の Xet）は 70 KB/s 程度まで落ちることがあるので、curl で直接取る
        base = f"https://huggingface.co/{dl['repo']}/resolve/{dl.get('revision', 'main')}/"
        for f in files:
            dest = AUDIOCPP_MODELS / f
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            r = subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(part),
                                base + urllib.parse.quote(f)])
            if r.returncode == 0:
                part.replace(dest)
    if not files or not all((AUDIOCPP_MODELS / f).exists() for f in files):
        subprocess.run([sys.executable, str(tools), "install", package, "--models-root", str(AUDIOCPP_MODELS)],
                       check=True, capture_output=True)
    ggufs = [f for f in files if f and f.endswith(".gguf")]
    return AUDIOCPP_MODELS / ggufs[0] if len(ggufs) == 1 and len(files) == 1 else target


def create_asr(runtime: str, key: str, device: str = "cuda", threads: int = 8, **kw) -> OpenAIASR:
    """runtime: crispasr / speechcpp / audiocpp。key はその実行系でのモデルの名前（runtimes/catalog.py）。"""
    from runtimes.catalog import ASR

    spec = ASR[runtime][key]
    if runtime == "crispasr":
        for u in spec.get("prefetch", []):  # サーバーが自分では取ってこない付属のファイル（トークナイザーなど）
            fetch(u, CRISPASR_CACHE)
        server, url = start_crispasr(spec["backend"], spec.get("model", "auto"), device, threads, spec.get("args"),
                                     url=spec.get("url"), quant=spec.get("quant", "q8_0"),
                                     language=spec.get("language", "ja"))
        model = spec["backend"]
    elif runtime == "speechcpp":
        server, url = start_speechcpp([spec["model"]], device, threads, spec.get("args"))
        model = ""
    elif runtime == "audiocpp":
        model = key
        path = audiocpp_install(spec["package"])
        if spec.get("file"):
            path = path / spec["file"]
        server, url = start_audiocpp([{"id": key, "family": spec["family"], "path": str(path),
                                       "task": "asr", "mode": "offline", **spec.get("load", {})}], device, threads)
    else:
        raise ValueError(runtime)
    dev = ("vulkan" if runtime == "speechcpp" else "cuda") if device.startswith("cuda") else "cpu"
    return OpenAIASR(url, model, server, dev, language=spec.get("language", "ja"), fields=spec.get("fields"))


REF_WAV = REPO / "data" / "voices" / "ref_ja.wav"  # Common Voice 8.0（CC0）の 1 発話。クローンが要るモデルの声
REF_WAV_24K = REPO / "data" / "voices" / "ref_ja_24k.wav"  # 24 kHz を求めるモデル（Qwen3-TTS など）用
REF_TEXT_FILE = REPO / "data" / "voices" / "ref_ja.txt"


def _ref_text(kana: bool = False) -> str:
    text = REF_TEXT_FILE.read_text(encoding="utf-8").strip() if REF_TEXT_FILE.is_file() else ""
    return to_kana(text) if kana and text else text


def to_kana(text: str) -> str:
    """漢字かな交じりの文を、発音どおりのカタカナにする（pyopenjtalk。「は」は「ワ」、長音は「ー」）。"""
    import pyopenjtalk

    return pyopenjtalk.g2p(text, kana=True)


def create_tts(runtime: str, key: str, device: str = "cuda", threads: int = 8, mode: str = "offline") -> OpenAITTS:
    """runtimes/catalog.py の TTS の構成を立ち上げる。spec["clone"] が真なら参照音声（REF_WAV）で声をクローンする。
    mode="streaming" は audio.cpp でストリーミングできるように登録する（runtimes/streaming.py）。"""
    from runtimes.catalog import TTS

    spec = TTS[runtime][key]
    clone = bool(spec.get("clone"))
    voices = [(v, v) for v in spec.get("voices", [])] or [("", "既定の声" if not clone else "参照音声（クローン）")]
    options = dict(spec.get("options", {}))
    if runtime == "crispasr":
        extra = list(spec.get("args", []))
        for u in spec.get("prefetch", []):  # サーバーが自分では取ってこない付属のファイル
            fetch(u, CRISPASR_CACHE)
        if clone:
            ref = REF_WAV_24K if spec.get("ref_24k") else REF_WAV
            extra += ["--voice", str(ref)] + (["--ref-text", _ref_text(spec.get("kana", False))]
                                              if spec.get("ref_text", True) else [])
        server, url = start_crispasr(spec["backend"], spec.get("model", "auto"), device, threads, extra,
                                     url=spec.get("url"), quant=spec.get("quant", "q8_0"), language=None)
        model = spec["backend"]
        options.setdefault("language", spec.get("language", "ja"))
    elif runtime == "speechcpp":
        extra = ["--add-voice", f"ref={REF_WAV_24K}"] if spec.get("add_ref_voice") else []  # 参照音声を声 ref として足す
        server, url = start_speechcpp([spec["model"]], device, threads, ["--language", "ja", *extra, *spec.get("args", [])])
        model = ""
    elif runtime == "audiocpp":
        path = audiocpp_install(spec["package"])
        if spec.get("file"):
            path = path / spec["file"]
        model = key
        entry = {"id": key, "family": spec["family"], "path": str(path), "task": "tts", "mode": mode,
                 **spec.get("load", {})}
        if mode == "streaming":  # ストリーミングのときだけ要る設定（VoxCPM2 の retry_badcase=false など）
            entry.update(spec.get("stream_load", {}))
        if spec.get("session_packages"):  # 別パッケージのモデル（コーデックなど）の場所
            entry["session_options"] = {k: str(audiocpp_install(v)) for k, v in spec["session_packages"].items()}
        server, url = start_audiocpp([entry], device, threads)
        options.setdefault("language", spec.get("language", "ja"))
        if clone:
            options["voice_ref"] = str(REF_WAV)
            if spec.get("ref_text", True):
                options["reference_text"] = _ref_text()
    else:
        raise ValueError(runtime)
    tts = OpenAITTS(url, model, server, voices, options)
    tts.kana = bool(spec.get("kana"))
    tts.device = ("vulkan" if runtime == "speechcpp" else "cuda") if device.startswith("cuda") else "cpu"
    return tts
