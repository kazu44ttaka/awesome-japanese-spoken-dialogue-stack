"""日本語 ASR の各モデルを同じ形（16 kHz mono float32 の配列 → 文字列）で呼べるようにする。

    from asrkit.backends import BACKENDS, create
    recognize = create("moonshine-tiny-ja", device="cpu")
    text = recognize(audio)   # audio: np.ndarray（16 kHz・mono・float32）

transformers / funasr / sherpa-onnx のうち、使うモデルに必要なものだけ入っていればよい。
sherpa-onnx のモデルは model_dir を省略すると models/ に自動でダウンロードする。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .models import ensure_sherpa_model


@dataclass(frozen=True)
class BackendInfo:
    name: str
    label: str
    runtime: str  # "transformers" / "funasr" / "sherpa-onnx"
    gpu: bool  # GPU で動かせるか（sherpa-onnx の配布物は CPU 用）


BACKENDS: dict[str, BackendInfo] = {b.name: b for b in [
    BackendInfo("sherpa-sensevoice", "SenseVoiceSmall（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-parakeet-ctc", "Parakeet TDT-CTC 0.6B 日本語（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-reazonspeech-k2", "ReazonSpeech k2-v2（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-cohere", "Cohere ASR 日本語（sherpa-onnx・CPU）", "sherpa-onnx", False),
    BackendInfo("sherpa-funasr-nano", "Fun-ASR-Nano（sherpa-onnx・CPU）", "sherpa-onnx", False),
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
    }
    if name in table:
        return table[name]()
    if name.startswith("sherpa-"):
        return Sherpa(name[len("sherpa-"):], model_dir, threads)
    raise ValueError(f"unknown backend: {name}")
