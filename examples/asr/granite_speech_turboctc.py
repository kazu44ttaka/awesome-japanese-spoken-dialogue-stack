"""IBM Granite Speech 5.0 470M TurboCTC で WAV を 1 つ書き起こす（日本語は 2026-10 から対応）。

    pip install torch "transformers>=5.16" soundfile
    python granite_speech_turboctc.py input.wav

読点（、）をほとんど出さないので、句読点が要るなら後処理で補う。-nc 版は日本語に対応していない。
"""

import sys

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

MODEL = "ibm-granite/granite-speech-5.0-470m-turboctc"  # Apache-2.0

device = "cuda" if torch.cuda.is_available() else "cpu"  # CTC なので GPU では 1 発話数十 ms
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCTC.from_pretrained(MODEL).to(device).eval()

audio, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
rate = processor.feature_extractor.sampling_rate
if sr != rate:
    import torchaudio.functional as F

    audio = F.resample(torch.from_numpy(audio), sr, rate).numpy()

inputs = processor(audio, sampling_rate=rate, return_tensors="pt", device=device)
inputs = inputs.to(device, dtype=model.dtype)
with torch.inference_mode():
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])
