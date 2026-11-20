"""会話の評価データを作る（data/eval/ に 16 kHz の WAV と manifest.json を書き出す）。

    python benchmarks/asr/scripts/make_extra_sets.py

- magichub-conv-1000-seed0: MagicHub の Japanese Spontaneous Conversation（2 人の自由会話を話者ごとに録音、
  iPhone）。書き起こしの区切りで切り出し、聞き取れない・重なり・別の人の声などの印がある区間と、録音の始めの
  決まり文句は除く。長さ 0.5〜20 秒から 1000 区間を乱数で選ぶ。著作権は Magic Data にあり再配布できないので、
  評価にだけ使う（音声も書き起こしも公開しない）
"""

from __future__ import annotations

import io
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data" / "eval"
MAGIC = "MagicHub/Japanese_Spontaneous_Conversation_Training_Dataset"
_DROP = re.compile(r"\[(\*|\+|NPS|PII|MUSIC|SYSTEM)\]")
_TAGS = re.compile(r"\[(LAUGHTER|SONANT|ENS)\]")


def _resample(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return audio
    import torch
    import torchaudio.functional as F

    return F.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, 16000).numpy()


def make_magichub(n: int = 1000, seed: int = 0) -> None:
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    out = EVAL / f"magichub-conv-{n}-seed{seed}"
    if (out / "manifest.json").is_file():
        return
    files = HfApi().list_repo_files(MAGIC, repo_type="dataset")
    segs = []
    for txt in sorted(f for f in files if f.endswith(".txt") and "/TXT/" in f):
        wav = txt.replace("/TXT/", "/WAV/").replace(".txt", ".wav")
        if wav not in files:
            continue
        for line in Path(hf_hub_download(MAGIC, txt, repo_type="dataset")).read_text(encoding="utf-8").splitlines():
            m = re.match(r"\[([\d.]+),([\d.]+)\]\t(\S+)\t\S+\t(.*)$", line.strip())
            if not m or m.group(3) == "0":
                continue
            s, e, text = float(m.group(1)), float(m.group(2)), m.group(4).strip()
            if _DROP.search(text) or "マジックデータ" in text or "トピック" in text or not (0.5 <= e - s <= 20):
                continue
            text = _TAGS.sub("", text).strip()
            if text:
                segs.append((wav, s, e, text))
    pick = sorted(random.Random(seed).sample(range(len(segs)), min(n, len(segs))))
    out.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for i in pick:
        wav, s, e, text = segs[i]
        if wav not in cache:
            cache.clear()
            a, rate = sf.read(hf_hub_download(MAGIC, wav, repo_type="dataset"), dtype="float32", always_2d=True)
            cache[wav] = (_resample(a.mean(axis=1), rate), 16000)
        a, rate = cache[wav]
        clip = a[int(s * rate):int(e * rate)]
        name = f"{i:05d}.wav"
        sf.write(str(out / name), clip, rate, subtype="PCM_16")
        rows.append({"wav": name, "ref": text})
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("magichub", len(segs), "segments ->", len(rows))


if __name__ == "__main__":
    make_magichub()
