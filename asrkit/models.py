"""sherpa-onnx のモデルと Silero VAD を、k2-fsa の配布元から models/ に取ってくる。"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

# 名前 → k2-fsa の配布物（tar.bz2 を展開したフォルダー名）
SHERPA_MODELS = {
    "sensevoice": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
    "parakeet-ctc": "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
    "reazonspeech-k2": "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
    "cohere": "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
    "funasr-nano": "sherpa-onnx-funasr-nano-int8-2025-12-30",
    "qwen3-asr": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
    # 真のストリーミング（OnlineRecognizer）。チャンクが短いほど途中経過が速く出るが、計算が増える
    "nemotron-80ms": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11",
    "nemotron-320ms": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-320ms-int8-2026-06-11",
    "nemotron-560ms": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
    "nemotron-1120ms": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def ensure_sherpa_model(name: str, root: Path = MODELS_DIR) -> Path:
    """モデルのフォルダーを返す。無ければダウンロードして展開する。"""
    folder = SHERPA_MODELS[name]
    target = root / folder
    if target.is_dir():
        return target
    archive = root / f"{folder}.tar.bz2"
    if not archive.is_file():
        _download(f"{RELEASE}/{folder}.tar.bz2", archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(root, filter="data")
    archive.unlink()
    return target


def ensure_silero_vad(root: Path = MODELS_DIR) -> Path:
    path = root / "silero_vad.onnx"
    if not path.is_file():
        _download(f"{RELEASE}/silero_vad.onnx", path)
    return path
