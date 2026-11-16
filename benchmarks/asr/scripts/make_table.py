"""data/results/<評価セット>/*.json（eval_asr.py の出力）を採点し直して、Markdown の表にする。

    python benchmarks/asr/scripts/make_table.py fleurs-ja reazonspeech-test cv8-ja
    python benchmarks/asr/scripts/make_table.py --summary fleurs-ja reazonspeech-test cv8-ja

--summary は評価セットを横に並べた 1 つの表（読みの CER と平均 ms）を出す。行は CER の平均順位の順。

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


def summary(datasets: list[str]) -> str:
    """評価セットごとの読みの CER と平均 ms を横に並べる。すべてのセットで測ったモデルだけを載せる。"""
    per: dict[str, dict[str, tuple[float, float, str]]] = {}
    for ds in datasets:
        for path in sorted((REPO / "data" / "results" / ds).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            s = summarize([score_pair(r["ref"], r["hyp"]) for r in data["rows"]])
            per.setdefault(path.stem, {})[ds] = (s["corpus_cer_kana"], data["mean_ms"], data.get("device", ""))
    names = [n for n, v in per.items() if all(ds in v for ds in datasets)]
    ranks = {n: 0.0 for n in names}
    for ds in datasets:
        for i, n in enumerate(sorted(names, key=lambda n: per[n][ds][0])):
            ranks[n] += (i + 1) / len(datasets)
    names.sort(key=lambda n: ranks[n])
    head = "| モデル | 実行 | " + " | ".join(f"{ds} CER" for ds in datasets) + " | 平均順位 | " + " | ".join(f"{ds} ms" for ds in datasets) + " |"
    lines = [head, "|---|---|" + "---:|" * (2 * len(datasets) + 1)]
    for n in names:
        cers = " | ".join(f"{per[n][ds][0]:.3f}" for ds in datasets)
        mss = " | ".join(f"{per[n][ds][1]:.0f}" for ds in datasets)
        lines.append(f"| {n} | {per[n][datasets[0]][2]} | {cers} | {ranks[n]:.1f} | {mss} |")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        print(summary(args[1:] or ["fleurs-ja", "reazonspeech-test", "cv8-ja"]))
        return
    for dataset in args or ["fleurs-ja"]:
        print(table(dataset))
        print()


if __name__ == "__main__":
    main()
