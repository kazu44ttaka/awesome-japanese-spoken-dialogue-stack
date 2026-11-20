"""JSUT basic5000 から、音素ごとの時刻が付いた評価データを作る（ストリーミングで、音素が入ってから文字が出るまでを測るため）。

    python benchmarks/asr/scripts/make_jsut_align.py   # models/jsut/jsut_ver1.1.zip と models/src/jsut-lab が要る

- 音声: JSUT v1.1（東京大学 猿渡研究室。研究・個人の非商用の利用。再配布しない）を 16 kHz にして data/eval/jsut-align-60/ へ
- 音素の時刻: jsut-lab（r9y9、MIT）の HTS 形式のラベル。Julius の強制アラインメントで付けたもの（100 ns 単位）
- 5000 発話から 83 発話おきに 60 発話。manifest の各行に phones: [[音素, 始まり秒, 終わり秒], ...]
  （無音 sil・pau は除く）
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ZIP = REPO / "models" / "jsut" / "jsut_ver1.1.zip"
LAB = REPO / "models" / "src" / "jsut-lab" / "basic5000" / "lab"
OUT = REPO / "data" / "eval" / "jsut-align-60"


def phones(lab: Path) -> list[list]:
    out = []
    for line in lab.read_text(encoding="utf-8").splitlines():
        s, e, ctx = line.split(maxsplit=2)
        ph = re.search(r"-(.+?)\+", ctx).group(1)
        if ph in ("sil", "pau"):
            continue
        out.append([ph, round(int(s) / 1e7, 4), round(int(e) / 1e7, 4)])
    return out


def main() -> None:
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    z = zipfile.ZipFile(ZIP)
    trans = {}
    for line in z.read("jsut_ver1.1/basic5000/transcript_utf8.txt").decode("utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            trans[k] = v.strip()
    ids = sorted(trans)[::83][:60]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in ids:
        a, rate = sf.read(io.BytesIO(z.read(f"jsut_ver1.1/basic5000/wav/{k}.wav")), dtype="float32", always_2d=True)
        a = F.resample(torch.from_numpy(np.ascontiguousarray(a.mean(axis=1))), rate, 16000).numpy()
        sf.write(str(OUT / f"{k}.wav"), a, 16000, subtype="PCM_16")
        rows.append({"wav": f"{k}.wav", "ref": trans[k], "phones": phones(LAB / f"{k}.lab")})
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "utterances ->", OUT)


if __name__ == "__main__":
    main()
