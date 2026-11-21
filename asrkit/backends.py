"""日本語 ASR の各モデルを同じ形（16 kHz mono float32 の配列 → 文字列）で呼べるようにする。

    from asrkit.backends import BACKENDS, create
    recognize = create("moonshine-tiny-ja", device="cpu")
    text = recognize(audio)   # audio: np.ndarray（16 kHz・mono・float32）

transformers / funasr / sherpa-onnx のうち、使うモデルに必要なものだけ入っていればよい。
sherpa-onnx のモデルは model_dir を省略すると models/ に自動でダウンロードする。
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
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
    cpu: bool = True  # CPU で動かせるか（False なら GPU だけ。haas-asr は Triton のカーネルを使うので GPU でしか動かない）


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
    BackendInfo("faster-whisper-sds", "Kotoba-Whisper v2.2（faster-whisper・MaAI-SDS の設定）", "faster-whisper", True),
    BackendInfo("sherpa-reazonspeech-k2-sds", "ReazonSpeech k2-v2（sherpa-onnx・CPU・MaAI-SDS の設定）", "sherpa-onnx", False),
    BackendInfo("qwen3-asr-0.6b", "Qwen3-ASR 0.6B（qwen-asr・PyTorch）", "qwen-asr", True),
    BackendInfo("qwen3-asr-1.7b", "Qwen3-ASR 1.7B（qwen-asr・PyTorch）", "qwen-asr", True),
    BackendInfo("moonshine-tiny-ja", "Moonshine tiny-ja（transformers）", "transformers", True),
    BackendInfo("granite", "Granite Speech 5.0 470M TurboCTC（transformers）", "transformers", True),
    BackendInfo("granite-4.0-1b", "Granite 4.0 1B Speech（transformers）", "transformers", True),
    BackendInfo("voxtral-realtime", "Voxtral Mini 4B Realtime・480 ms（transformers・GPU 約 9 GB）", "transformers", True),
    BackendInfo("moonshine-base-ja", "Moonshine base-ja（transformers）", "transformers", True),
    BackendInfo("haas-ja", "Haas-ja-v1.0（haas-asr・GPU だけ・専用の環境）", "haas-asr", True, cpu=False),
    BackendInfo("parakeet-ctc-1.1b-ja", "Parakeet CTC 1.1B 日本語（NeMo・専用の環境）", "nemo", True),
    BackendInfo("reazonspeech-nemo-v2", "ReazonSpeech NeMo v2（NeMo・専用の環境）", "nemo", True),
    BackendInfo("parakeet-tdt-ctc-0.6b-ja", "Parakeet TDT-CTC 0.6B 日本語（NeMo・専用の環境）", "nemo", True),
    BackendInfo("whisper-large-v3-turbo", "Whisper large-v3-turbo（transformers）", "transformers", True),
    BackendInfo("kotoba-whisper-v2.0", "Kotoba-Whisper v2.0（transformers）", "transformers", True),
    BackendInfo("anime-whisper", "anime-whisper（transformers）", "transformers", True),
    BackendInfo("anime-whisper-faster", "anime-whisper（faster-whisper・CTranslate2 fp16）", "faster-whisper", True),
    BackendInfo("fun-asr-nano", "Fun-ASR-Nano-2512（funasr）", "funasr", True),
    BackendInfo("fun-asr-mlt-nano", "Fun-ASR-MLT-Nano-2512（funasr）", "funasr", True),
]}


def _add_runtime_backends() -> None:
    """ggml 系の実行系（CrispASR・audio.cpp・speech.cpp）の構成を、測定と同じ一覧（runtimes/catalog.py）から足す。"""
    try:
        from runtimes.catalog import APP_STREAMING_ASR, ASR, label
    except ImportError:  # リポジトリの外から asrkit だけを使うとき
        return
    for runtime, models in ASR.items():
        for key, spec in models.items():
            name = f"{runtime}-{key}"
            BACKENDS[name] = BackendInfo(name, label(runtime, key, spec), runtime, True, name in APP_STREAMING_ASR)


_add_runtime_backends()


class GraniteCTC:
    """IBM Granite Speech 5.0 TurboCTC（transformers>=5.16、AutoModelForCTC）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        # transformers 5 は既定で設定どおりの型（このモデルは bfloat16）で読む。CPU の bfloat16 は遅いので float32 にする
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCTC.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", device=self.device)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


class Granite4Speech:
    """IBM Granite 4.0 1B Speech（音声エンコーダ + LLM。chat の形で「書き起こして」と頼む）。"""

    PROMPT = "<|audio|>can you transcribe the speech into a written format?"

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": self.PROMPT}], tokenize=False, add_generation_prompt=True)
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        inputs = self.processor(self.prompt, wav, device=self.device, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=1)
        new = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True).strip()


class VoxtralRealtime:
    """Mistral Voxtral Mini 4B Realtime を一括で認識する（transformers>=5.2）。

    遅延は既定の 480 ms（6 トークン）。少しずつ入れる使い方は transformers の
    VoxtralRealtime の説明（is_streaming=True）を参照。
    """

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        return self.batch([audio])[0]

    def batch(self, audios: list[np.ndarray]) -> list[str]:
        """複数の発話をまとめて認識する（1 トークンごとの CPU の手間がまとめて減る。評価を速くする用）。"""
        inputs = self.processor([np.ascontiguousarray(a, dtype=np.float32) for a in audios], return_tensors="pt")
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs)
        return [t.strip() for t in self.processor.batch_decode(out, skip_special_tokens=True)]


class VenvASR:
    """モデルを別の Python 環境の子プロセス（asrkit.worker）で動かす。

    Haas-ja と Parakeet CTC 1.1B ja は NeMo 2.7 を使う専用の環境（.venv-haas）にしか入らないので、
    いまの環境に haas_asr / nemo が無いときはこれで動かす（デモアプリから選べるように）。
    """

    def __init__(self, name: str, python: Path, device: str = "cuda") -> None:
        import json

        env = dict(os.environ, PYTHONIOENCODING="utf-8", HF_HUB_DISABLE_PROGRESS_BARS="1")
        # 親の環境の Python を指す変数を引き継ぐと、版の違う子の Python が親の標準ライブラリを読んで落ちる
        for key in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)
        env.setdefault("HF_HUB_CACHE", str(REPO / "models" / "hf"))
        self.device = device
        self.process = subprocess.Popen([str(python), "-m", "asrkit.worker", name, device], cwd=str(REPO), env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding="utf-8",
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        first = json.loads(self.process.stdout.readline() or '{"error": "子プロセスが終了しました"}')
        if "error" in first:
            self.close()
            raise RuntimeError(first["error"])

    def __call__(self, audio: np.ndarray) -> str:
        import base64
        import json

        pcm = base64.b64encode(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).decode("ascii")
        self.process.stdin.write(json.dumps({"audio": pcm}) + "\n")
        self.process.stdin.flush()
        res = json.loads(self.process.stdout.readline() or '{"error": "子プロセスが終了しました"}')
        if "error" in res:
            raise RuntimeError(res["error"])
        return str(res["text"])

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.process.kill()
        self.process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


# 専用の環境で動かすモデル → (いまの環境に要るモジュール, 専用の環境の python)
_VENV_MODELS = {
    "haas-ja": ("haas_asr", REPO / ".venv-haas" / "Scripts" / "python.exe"),
    "parakeet-ctc-1.1b-ja": ("nemo", REPO / ".venv-haas" / "Scripts" / "python.exe"),
    "reazonspeech-nemo-v2": ("nemo", REPO / ".venv-haas" / "Scripts" / "python.exe"),
    "parakeet-tdt-ctc-0.6b-ja": ("nemo", REPO / ".venv-haas" / "Scripts" / "python.exe"),
}


class HaasASR:
    """Third Intelligence の Haas-ja-v1.0（haas-asr。NeMo 2.7 と flash-linear-attention を使う）。

    公式は Linux 専用だが、Windows でも triton-windows を入れれば動く（別の環境 .venv-haas。apps/README.md）。
    1 回目の認識は Triton のカーネルを作るので 10 秒ほどかかる。
    """

    def __init__(self, model_id: str) -> None:
        import haas_asr

        self.model = haas_asr.load_model(model_id)

    def __call__(self, audio: np.ndarray) -> str:
        out = self.model.transcribe([np.ascontiguousarray(audio, dtype=np.float32)])
        return out[0].text.strip() if out else ""


class NemoASR:
    """.nemo の ASR モデルを NeMo で読み込む（parakeet-ctc-1.1b-ja など）。"""

    def __init__(self, repo: str, filename: str, device: str) -> None:
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename)
        self.model = nemo_asr.models.ASRModel.restore_from(path, map_location=device).eval()

    def __call__(self, audio: np.ndarray) -> str:
        out = self.model.transcribe([np.ascontiguousarray(audio, dtype=np.float32)], batch_size=1, verbose=False)
        r = out[0] if out else ""
        return str(getattr(r, "text", r)).replace("⁇", "").lstrip("…").strip()


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
        # 繰り返しの暴走を止める: 圧縮率（トークン空間、transformers の推奨 1.35）や平均の対数確率が悪い出力は、
        # 温度を上げて出し直す（OpenAI の Whisper や faster-whisper の既定と同じ考え方）
        out = self.pipe({"raw": audio, "sampling_rate": 16000},
                        generate_kwargs={"language": "ja", "task": "transcribe",
                                         "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                         "compression_ratio_threshold": 1.35, "logprob_threshold": -1.0})
        return str(out.get("text", ""))


def add_cuda_dll_dirs() -> None:
    """CUDA 版の ONNX Runtime（sherpa-onnx など）が要る cuDNN 9・cuBLAS の DLL を、PyTorch か nvidia-* の wheel から探す。"""
    if sys.platform != "win32":
        return
    dirs = []
    try:
        import torch

        dirs.append(Path(torch.__file__).parent / "lib")
    except ImportError:
        pass
    try:
        import nvidia

        for base in nvidia.__path__:
            dirs += [d for d in Path(base).glob("*/bin") if d.is_dir()]
    except ImportError:
        pass
    for d in dirs:
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


def sherpa_has_cuda() -> bool:
    """入っている sherpa-onnx が CUDA 版（版に +cuda が付く）か。CPU 版に cuda を頼むと黙って CPU で動くため確かめる。"""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return "cuda" in version("sherpa-onnx")
    except PackageNotFoundError:
        return False


class Sherpa:
    """sherpa-onnx のオフライン認識（k2-fsa の配布物）。CUDA 版の sherpa-onnx なら GPU でも動く。

    CPU では int8 を使う。int8 は一部の演算（MatMulInteger など）が CUDA に載らず CPU に戻って遅くなるので、
    GPU では fp32 の配布があるもの（sensevoice・reazonspeech-k2・funasr-nano）は fp32 を使う。
    """

    def __init__(self, kind: str, model_dir: Path | None, threads: int, provider: str = "cpu") -> None:
        import sherpa_onnx

        if provider == "cuda":
            add_cuda_dll_dirs()
        self.device = provider
        self.kind = kind
        fp32 = provider == "cuda" and kind in ("sensevoice", "reazonspeech-k2", "funasr-nano")
        self.precision = "fp32" if fp32 else "int8"
        q = "" if fp32 else ".int8"
        if model_dir:
            d = Path(model_dir)
        elif fp32 and kind != "reazonspeech-k2":
            d = ensure_sherpa_model(f"{kind}-fp32")
        else:
            d = ensure_sherpa_model(kind)
        R = sherpa_onnx.OfflineRecognizer
        if kind == "sensevoice":
            self.rec = R.from_sense_voice(model=str(d / f"model{q}.onnx"), tokens=str(d / "tokens.txt"),
                                          num_threads=threads, provider=provider, language="ja", use_itn=True)
        elif kind == "parakeet-ctc":
            self.rec = R.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                       num_threads=threads, provider=provider)
        elif kind == "reazonspeech-k2":
            self.rec = R.from_transducer(
                encoder=str(d / f"encoder-epoch-99-avg-1{q}.onnx"), decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
                joiner=str(d / f"joiner-epoch-99-avg-1{q}.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads, provider=provider)
        elif kind == "cohere":
            self.rec = R.from_cohere_transcribe(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                                                tokens=str(d / "tokens.txt"), num_threads=threads, provider=provider, language="ja",
                                                use_punct=True, use_itn=True)
        elif kind == "qwen3-asr":
            # fp32 の配布が無い。言語は認識のたびに stream.set_option("language", "Japanese") で指定する（__call__）
            self.rec = R.from_qwen3_asr(conv_frontend=str(d / "conv_frontend.onnx"), encoder=str(d / "encoder.int8.onnx"),
                                        decoder=str(d / "decoder.int8.onnx"), tokenizer=str(d / "tokenizer"),
                                        num_threads=threads, provider=provider, sample_rate=16000, feature_dim=128)
        elif kind == "funasr-nano":
            self.rec = R.from_funasr_nano(encoder_adaptor=str(d / f"encoder_adaptor{q}.onnx"), llm=str(d / ("llm.fp32.onnx" if fp32 else "llm.int8.onnx")),
                                          embedding=str(d / f"embedding{q}.onnx"), tokenizer=str(d / "Qwen3-0.6B"),
                                          num_threads=threads, provider=provider, language="ja", itn=True, max_new_tokens=128)
        else:
            raise ValueError(kind)

    def __call__(self, audio: np.ndarray) -> str:
        stream = self.rec.create_stream()
        if self.kind == "qwen3-asr":
            # 言語を日本語に固定する（MaAI-SDS の engines_local.py と同じ）。指定しないと自動判定で
            # 韓国語・英語・中国語に流れることがある
            stream.set_option("language", "Japanese")
        stream.accept_waveform(16000, np.ascontiguousarray(audio, dtype=np.float32))
        self.rec.decode_stream(stream)
        text = re.sub(r"<\|[^|]+\|>", "", stream.result.text)
        return clean_qwen_text(text) if self.kind == "qwen3-asr" else text


def clean_qwen_text(text: str) -> str:
    """Qwen3-ASR の "language Japanese<asr_text>本文" から本文だけを取り出す（MaAI-SDS の _clean_qwen_text と同じ）。"""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    m = re.search(r"<\s*asr_text\s*>", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = cleaned[m.end():]
    cleaned = re.sub(r"</?\s*asr_text\s*>", "", cleaned, flags=re.IGNORECASE)
    if m is None:
        cleaned = re.sub(r"^\s*language\s+[A-Za-z\-]+\s*[:：]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


class FasterWhisper:
    """faster-whisper（CTranslate2）。既定は GPU で float16・CPU で int8・beam 1。

    MaAI-SDS の設定（faster-whisper-sds）は、Kotoba-Whisper v2.2 を int8_float16・beam 5 で動かし、
    先頭に 0.5 秒の無音を足す（MaAI-SDS の engines_local.py と同じ）。
    """

    def __init__(self, model_id: str, device: str, threads: int = 0, beam_size: int = 1,
                 compute_type: str | None = None, lead_silence_sec: float = 0.0) -> None:
        from faster_whisper import WhisperModel

        cuda = device.startswith("cuda")
        self.model = WhisperModel(model_id, device="cuda" if cuda else "cpu",
                                  compute_type=compute_type or ("float16" if cuda else "int8"), cpu_threads=threads)
        self.beam_size = beam_size
        self.lead = np.zeros(int(16000 * lead_silence_sec), dtype=np.float32)

    def __call__(self, audio: np.ndarray) -> str:
        audio = np.concatenate([self.lead, np.asarray(audio, dtype=np.float32)]) if len(self.lead) else audio
        segments, _info = self.model.transcribe(np.ascontiguousarray(audio, dtype=np.float32), language="ja",
                                                beam_size=self.beam_size, vad_filter=False, condition_on_previous_text=False)
        return "".join(s.text for s in segments)


class LeadSilence:
    """認識の前に先頭へ無音を足す（MaAI-SDS は zipformer に 0.5 秒足して先頭を安定させている）。"""

    def __init__(self, inner, sec: float = 0.5) -> None:
        self.inner = inner
        self.lead = np.zeros(int(16000 * sec), dtype=np.float32)
        self.device = getattr(inner, "device", "cpu")
        self.precision = getattr(inner, "precision", None)

    def __call__(self, audio: np.ndarray) -> str:
        return self.inner(np.concatenate([self.lead, np.asarray(audio, dtype=np.float32)]))

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


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

    def __init__(self, chunk_ms: int = 560, threads: int = 4, model_dir: Path | None = None, provider: str = "cpu") -> None:
        import sherpa_onnx

        if provider == "cuda":
            add_cuda_dll_dirs()
        self.device = provider

        d = Path(model_dir) if model_dir else ensure_sherpa_model(f"nemotron-{chunk_ms}ms")
        self.chunk_ms = chunk_ms
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"), num_threads=threads, sample_rate=16000, feature_dim=128,
            decoding_method="greedy_search", dither=0.0, provider=provider)
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


class RuntimeStreaming:
    """ggml 系の実行系のストリーミング（runtimes/streaming.py）を、アプリの真のストリーミングと同じ形
    （start / feed / finish、.stream、audio → text）で使う。実行系のサーバーは自動で立ち上げ、close で止める。"""

    FINISH_TIMEOUT = 10.0  # 確定の文字が返らないときは、最後の途中経過を確定にする

    def __init__(self, runtime: str, key: str, device: str = "cuda", threads: int = 4) -> None:
        from runtimes.streaming import create_stream_asr

        self.inner = create_stream_asr(runtime, key, device=device, threads=threads)
        self.device = getattr(self.inner, "device", device)
        self.stream = None

    def _latest(self) -> str:
        return self.inner.events[-1][1] if self.inner.events else ""

    def start(self) -> None:
        self.inner.start()
        self.stream = True

    def feed(self, audio: np.ndarray) -> str:
        self.inner.feed(np.ascontiguousarray(audio, dtype=np.float32))
        return self._latest()

    def finish(self) -> str:
        try:
            return self.inner.finish(self.FINISH_TIMEOUT)
        except TimeoutError:
            return self._latest()
        finally:
            self.stream = None

    def __call__(self, audio: np.ndarray) -> str:
        """区間全体を一度に認識する（ウォームアップ用）。0.1 秒ずつ送る。"""
        self.start()
        for i in range(0, len(audio), RATE_16K // 10):
            self.feed(audio[i:i + RATE_16K // 10])
        return self.finish()

    def close(self) -> None:
        self.inner.close()


RATE_16K = 16000


def create_streaming(name: str, threads: int = 4, model_dir: Path | None = None, config: dict | None = None,
                     device: str = "cpu"):
    """真のストリーミングのモデルを作る。device="cuda" は CUDA 版の sherpa-onnx のときだけ GPU になる。"""
    runtime, _, key = name.partition("-")
    if runtime in ("crispasr", "speechcpp", "audiocpp"):
        return RuntimeStreaming(runtime, key, device, threads)
    if name == "nemotron-nemo":
        cfg = dict((config if config is not None else load_config()).get("nemotron_nemo", {}))
        return NemoNemotron(cfg)
    chunk = {"nemotron-560ms": 560, "nemotron-80ms": 80}[name]
    provider = "cuda" if device.startswith("cuda") and sherpa_has_cuda() else "cpu"
    return NemotronStreaming(chunk, threads, model_dir, provider)


def create(name: str, device: str = "cpu", threads: int = 8, model_dir: Path | None = None) -> Callable[[np.ndarray], str]:
    """名前からモデルを読み込み、audio → text の関数を返す。"""
    table = {
        "granite": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc", device),
        "granite-nc": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc-nc", device),
        "moonshine-tiny-ja": lambda: Moonshine("UsefulSensors/moonshine-tiny-ja", device),
        "moonshine-base-ja": lambda: Moonshine("UsefulSensors/moonshine-base-ja", device),
        "granite-4.0-1b": lambda: Granite4Speech("ibm-granite/granite-4.0-1b-speech", device),
        "voxtral-realtime": lambda: VoxtralRealtime("mistralai/Voxtral-Mini-4B-Realtime-2602", device),
        "haas-ja": lambda: HaasASR("third-intelligence/Haas-ja-v1.0"),
        "parakeet-ctc-1.1b-ja": lambda: NemoASR("grider-transwithai/parakeet-ctc-1.1b-ja", "parakeet-ja.nemo", device),
        # ggml の実行系（speech.cpp・CrispASR）と比べるための元の実装（NeMo）
        "reazonspeech-nemo-v2": lambda: NemoASR("reazon-research/reazonspeech-nemo-v2", "reazonspeech-nemo-v2.nemo", device),
        "parakeet-tdt-ctc-0.6b-ja": lambda: NemoASR("nvidia/parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt_ctc-0.6b-ja.nemo", device),
        "fun-asr-mlt-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-MLT-Nano-2512", device),
        "fun-asr-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-Nano-2512", device),
        "whisper-large-v3-turbo": lambda: WhisperHF("openai/whisper-large-v3-turbo", device),
        "kotoba-whisper-v2.0": lambda: WhisperHF("kotoba-tech/kotoba-whisper-v2.0", device),
        # アニメ・ゲームの台詞で追加学習した Kotoba-Whisper v2.0（MIT）。faster-whisper 版は有志の変換
        "anime-whisper": lambda: WhisperHF("litagin/anime-whisper", device),
        "anime-whisper-faster": lambda: FasterWhisper("quantumcookie/anime-whisper-ct2-fp16", device, threads),
        "faster-whisper": lambda: FasterWhisper("deepdml/faster-whisper-large-v3-turbo-ct2", device, threads),
        "faster-whisper-sds": lambda: FasterWhisper("RoachLin/kotoba-whisper-v2.2-faster", device, threads, beam_size=5,
                                                    compute_type="int8_float16" if device.startswith("cuda") else "int8",
                                                    lead_silence_sec=0.5),
        # MaAI-SDS と同じく CPU の int8 エンコーダ・fp32 デコーダで、先頭に 0.5 秒の無音を足す
        "sherpa-reazonspeech-k2-sds": lambda: LeadSilence(Sherpa("reazonspeech-k2", model_dir, threads, "cpu"), 0.5),
        "qwen3-asr-0.6b": lambda: Qwen3ASRTorch("Qwen/Qwen3-ASR-0.6B", device),
        "qwen3-asr-1.7b": lambda: Qwen3ASRTorch("Qwen/Qwen3-ASR-1.7B", device),
    }
    runtime, _, key = name.partition("-")
    if runtime in ("crispasr", "speechcpp", "audiocpp"):
        # ggml 系の実行系のサーバー（runtimes/servers.py・runtimes/catalog.py）。例: speechcpp-haas-ja
        from runtimes.servers import create_asr as create_runtime_asr

        return create_runtime_asr(runtime, key, device=device, threads=threads)
    if name in _VENV_MODELS:
        import importlib.util

        module, python = _VENV_MODELS[name]
        if importlib.util.find_spec(module) is None and python.is_file():
            return VenvASR(name, python, device)
    if name in table:
        return table[name]()
    if name.startswith("sherpa-"):
        provider = "cuda" if device.startswith("cuda") and sherpa_has_cuda() else "cpu"
        return Sherpa(name[len("sherpa-"):], model_dir, threads, provider)
    if name in BACKENDS and BACKENDS[name].streaming:
        return create_streaming(name, threads, model_dir, device=device)
    raise ValueError(f"unknown backend: {name}")
