"""推論時間を同じ条件で測るための小さな評価セット（100 発話）を作る。

    python benchmarks/asr/scripts/make_timing_set.py

eval_asr.py で取得済みの 3 つの評価データから、乱数（seed 1）で ReazonSpeech 40・Common Voice 30・FLEURS 30 発話を選び、
data/eval/timing100/ に WAV と manifest.json を書く。長さの違う発話を混ぜて、1 発話あたりの時間と RTF を見る。
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data" / "eval"


def from_manifest(folder: str, n: int, prefix: str, rng: random.Random) -> list[dict]:
    rows = [r for r in json.loads((EVAL / folder / "manifest.json").read_text(encoding="utf-8")) if r["ref"].strip()]
    return [{"src": EVAL / folder / r["wav"], "wav": f"{prefix}_{Path(r['wav']).stem}.wav", "ref": r["ref"]}
            for r in sorted(rng.sample(rows, n), key=lambda r: r["wav"])]


def from_fleurs(n: int, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    tsv = Path(hf_hub_download("google/fleurs", "data/ja_jp/test.tsv", repo_type="dataset"))
    rows = []
    with tsv.open(encoding="utf-8") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            wav = EVAL / "fleurs_ja" / "test" / rec[1]
            if len(rec) >= 3 and wav.is_file():
                rows.append({"src": wav, "wav": f"fleurs_{wav.stem}.wav", "ref": rec[2]})
    uniq = {r["wav"]: r for r in rows}  # FLEURS は同じ文を複数の話者が読むので、ファイル単位で選ぶ
    return sorted(rng.sample(sorted(uniq.values(), key=lambda r: r["wav"]), n), key=lambda r: r["wav"])


def main() -> None:
    rng = random.Random(1)
    rows = (from_manifest("reazonspeech-test-1000-seed0", 40, "reazon", rng)
            + from_manifest("cv8-ja-1000-seed0", 30, "cv8", rng) + from_fleurs(30, rng))
    out = EVAL / "timing100"
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        shutil.copyfile(r["src"], out / r["wav"])
    (out / "manifest.json").write_text(json.dumps([{"wav": r["wav"], "ref": r["ref"]} for r in rows],
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} 発話を {out} に書きました")


if __name__ == "__main__":
    main()
