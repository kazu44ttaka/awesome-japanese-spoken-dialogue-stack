"""Moonshine tiny-ja で WAV を 1 つ書き起こす。

    pip install torch "transformers>=4.48" soundfile
    python moonshine_tiny_ja.py input.wav
"""

import sys

import soundfile as sf
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

MODEL = "UsefulSensors/moonshine-tiny-ja"  # 英語以外は非商用の Moonshine Community License

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
processor = AutoProcessor.from_pretrained(MODEL)
model = MoonshineForConditionalGeneration.from_pretrained(MODEL).to(device, dtype)

audio, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
rate = processor.feature_extractor.sampling_rate  # 16000
if sr != rate:
    import torchaudio.functional as F

    audio = F.resample(torch.from_numpy(audio), sr, rate).numpy()

inputs = processor(audio, sampling_rate=rate, return_tensors="pt").to(device, dtype)
# 音声 1 秒あたり 13 トークンを上限にする（公式の例と同じ。繰り返しの暴走を防ぐ）
max_length = int(inputs.attention_mask.sum() * 13 / rate)
ids = model.generate(**inputs, max_length=max(8, max_length))
print(processor.decode(ids[0], skip_special_tokens=True))
