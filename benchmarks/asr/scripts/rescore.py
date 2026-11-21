"""採点の決まり（scoring.py）を変えたあとで、結果の JSON を採点し直す（音声は使わない）。

    python benchmarks/asr/scripts/rescore.py fleurs-ja reazonspeech-test cv8-ja magichub-conv timing_gpu timing_cpu \
        asr_stream asr_stream_jsut

data/results/<フォルダー>/*.json の rows に残した正解文（ref）と認識結果（hyp）から、発話ごとの CER と
全体の集計を計算し直して上書きする。上書きする前のファイルは --backup の下に同じ形で残す。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score_pair, summarize  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="data/results/ の下のフォルダー")
    ap.add_argument("--backup", required=True, help="data/results/ の下の、元のファイルを残すフォルダー")
    args = ap.parse_args()
    for name in args.dirs:
        src = REPO / "data" / "results" / name
        for path in sorted(src.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("rows") or []
            if not rows or "ref" not in rows[0] or "hyp" not in rows[0]:
                continue
            before = data.get("corpus_cer_kana")
            for row in rows:
                row.update(score_pair(row["ref"], row["hyp"]))
            data.update(summarize(rows))
            bak = REPO / "data" / "results" / args.backup / name / path.name
            if not bak.exists():
                bak.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, bak)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{name}/{path.stem}: {before} -> {data['corpus_cer_kana']}", flush=True)


if __name__ == "__main__":
    main()
