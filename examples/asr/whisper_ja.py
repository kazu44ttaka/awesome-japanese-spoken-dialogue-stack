"""Whisper 系（Whisper large-v3-turbo / Kotoba-Whisper v2.0）で WAV を 1 つ書き起こす。

    pip install torch transformers soundfile
    python whisper_ja.py input.wav                                  # openai/whisper-large-v3-turbo
    python whisper_ja.py input.wav kotoba-tech/kotoba-whisper-v2.0  # 日本語特化の蒸留版
"""

import sys

import soundfile as sf
import torch
from transformers import pipeline

model_id = sys.argv[2] if len(sys.argv) > 2 else "openai/whisper-large-v3-turbo"  # MIT / Apache-2.0
device = "cuda" if torch.cuda.is_available() else "cpu"
asr = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device=device,
)

audio, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
out = asr(
    {"raw": audio.mean(axis=1), "sampling_rate": sr},  # 16 kHz 以外は pipeline が変換する
    generate_kwargs={"language": "ja", "task": "transcribe"},  # 言語を決め打ちすると誤判定が減る
)
print(out["text"])
