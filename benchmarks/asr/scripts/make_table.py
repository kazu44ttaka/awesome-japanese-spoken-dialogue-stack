"""data/results/<評価セット>/*.json（eval_asr.py の出力）を採点し直して、Markdown の表にする。

    python benchmarks/asr/scripts/make_table.py fleurs-ja reazonspeech-test cv8-ja

結果の JSON に残した正解文と認識結果から scoring.py で採点し直すので、採点の決まりを変えても
測り直さなくてよい。表は読みで比べた全体の CER の小さい順。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score_pair, summarize  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def table(dataset: str) -> str:
    rows_out = []
    for path in sorted((REPO / "data" / "results" / dataset).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scored = [score_pair(r["ref"], r["hyp"]) for r in data["rows"]]
        s = summarize(scored)
        rows_out.append((s["corpus_cer_kana"], path.stem, data, s))
    rows_out.sort(key=lambda x: x[0])
    lines = [
        f"### {dataset}（{rows_out[0][3]['count'] if rows_out else 0} 発話）",
        "",
        "| モデル | 実行 | CER 句読点あり | CER 句読点なし | CER 読み（95% 区間） | 平均 ms | p95 ms | RTF |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _k, name, d, s in rows_out:
        lo, hi = s["ci95_corpus_cer_kana"]
        lines.append(
            f"| {name} | {d.get('device', '')} | {s['corpus_cer']:.3f} | {s['corpus_cer_nopunct']:.3f} | "
            f"**{s['corpus_cer_kana']:.3f}**（{lo:.3f}〜{hi:.3f}） | {d['mean_ms']:.0f} | {d['p95_ms']:.0f} | {d['rtf']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    for dataset in sys.argv[1:] or ["fleurs-ja"]:
        print(table(dataset))
        print()


if __name__ == "__main__":
    main()
