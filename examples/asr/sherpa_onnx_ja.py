"""sherpa-onnx（CPU）の日本語対応モデルで WAV を 1 つ書き起こす。モデルは初回に自動でダウンロードする。

    pip install sherpa-onnx soundfile
    python sherpa_onnx_ja.py input.wav sensevoice
    python sherpa_onnx_ja.py input.wav parakeet-ctc
    python sherpa_onnx_ja.py input.wav reazonspeech-k2
    python sherpa_onnx_ja.py input.wav cohere          # 約 2.7 GB
    python sherpa_onnx_ja.py input.wav funasr-nano     # 約 1 GB

PyTorch は要らない。配布物は int8 量子化で、CPU のほうが GPU より速い。
"""

import sys
import tarfile
import urllib.request
from pathlib import Path

import sherpa_onnx
import soundfile as sf

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
MODELS = {
    "sensevoice": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",  # 2025-09-09 版は広東語向けなので使わない
    "parakeet-ctc": "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
    "reazonspeech-k2": "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
    "cohere": "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    "funasr-nano": "sherpa-onnx-funasr-nano-int8-2025-12-30",
}


def download(name: str, root: Path = Path("models")) -> Path:
    folder = root / MODELS[name]
    if not folder.is_dir():
        root.mkdir(exist_ok=True)
        archive = root / f"{MODELS[name]}.tar.bz2"
        print(f"downloading {archive.name} ...", file=sys.stderr)
        urllib.request.urlretrieve(f"{RELEASE}/{archive.name}", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        archive.unlink()
    return folder


def recognizer(name: str, d: Path) -> sherpa_onnx.OfflineRecognizer:
    R = sherpa_onnx.OfflineRecognizer
    if name == "sensevoice":
        return R.from_sense_voice(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                  language="ja", use_itn=True)  # ITN を有効にすると句読点が付く
    if name == "parakeet-ctc":
        return R.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"))
    if name == "reazonspeech-k2":
        return R.from_transducer(encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"),
                                 decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
                                 joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"), tokens=str(d / "tokens.txt"))
    if name == "cohere":
        return R.from_cohere_transcribe(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                                        tokens=str(d / "tokens.txt"), language="ja", use_punct=True, use_itn=True)
    if name == "funasr-nano":
        # 自己回帰の LLM デコーダなので、生成の上限が最悪の遅延を決める（既定 512 は長すぎる）
        return R.from_funasr_nano(encoder_adaptor=str(d / "encoder_adaptor.int8.onnx"), llm=str(d / "llm.int8.onnx"),
                                  embedding=str(d / "embedding.int8.onnx"), tokenizer=str(d / "Qwen3-0.6B"),
                                  language="ja", itn=True, max_new_tokens=128)
    raise SystemExit(f"model must be one of {', '.join(MODELS)}")


wav, name = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sensevoice"
rec = recognizer(name, download(name))
audio, sr = sf.read(wav, dtype="float32", always_2d=True)
stream = rec.create_stream()
stream.accept_waveform(sr, audio.mean(axis=1))  # 16 kHz 以外は sherpa-onnx が変換する
rec.decode_stream(stream)
print(stream.result.text)
