"""Fun-ASR-Nano-2512（日本語・中国語・英語）/ Fun-ASR-MLT-Nano-2512（31 言語）で WAV を 1 つ書き起こす。

    pip install torch funasr soundfile
    python fun_asr_nano.py input.wav          # Fun-ASR-Nano-2512
    python fun_asr_nano.py input.wav --mlt    # Fun-ASR-MLT-Nano-2512
"""

import sys

import soundfile as sf
import torch
from funasr import AutoModel
from huggingface_hub import snapshot_download

repo = "FunAudioLLM/Fun-ASR-MLT-Nano-2512" if "--mlt" in sys.argv else "FunAudioLLM/Fun-ASR-Nano-2512"
model = AutoModel(
    model=snapshot_download(repo),
    trust_remote_code=True,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    disable_update=True,
)

audio, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True)
audio = torch.from_numpy(audio.mean(axis=1))
if sr != 16000:
    import torchaudio.functional as F

    audio = F.resample(audio, sr, 16000)

# 音声は torch.Tensor（16 kHz）かファイルパスで渡す。numpy の配列は受け付けない
res = model.generate(input=[audio], cache={}, batch_size=1, language="日文", itn=True,
                     llm_kwargs={"do_sample": False})
print(res[0]["text"])
