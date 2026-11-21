"""data/models.toml から README.md（日本語）と README.en.md（英語）の表を作る。

    python scripts/build_readme.py          # README を書き直す
    python scripts/build_readme.py --check  # 書き直しが必要なら 1 で終わる（CI 用）

テンプレート（templates/README.*.md）の `<!-- table:asr -->` などの行を表に置き換える。
標準ライブラリだけで動く（tomllib は Python 3.11 以降）。
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "ja": {
        "header": ["モデル", "開発元", "日本語", "ストリーミング", "動かし方", "大きさ", "ライセンス", "商用", "メモ"],
        "todo_header": ["モデル", "メモ"],
        "ja": {"native": "専用", "multi": "多言語"},
        "streaming": {"true": "○", "pseudo": "△（区切って再推論）", "no": "—"},
        "commercial": {"yes": "可", "no": "不可", "cond": "条件つき", "model": "声・モデルごと", "unknown": "不明", "tbd": "要確認"},
        "bench": "測定",
        "note": "note_ja",
        "measured": "値：{}",
        "score_cols": {
            "asr": ["FLEURS", "ReazonSpeech", "CV", "会話"],
            "tts": ["1 文 ms", "読みの CER", "UTMOS", "最初の声 ms"],
        },
    },
    "en": {
        "header": ["Model", "Developer", "Japanese", "Streaming", "Runtime", "Size", "License", "Commercial", "Notes"],
        "todo_header": ["Model", "Notes"],
        "ja": {"native": "dedicated", "multi": "multilingual"},
        "streaming": {"true": "yes", "pseudo": "pseudo", "no": "—"},
        "commercial": {"yes": "yes", "no": "no", "cond": "conditional", "model": "per voice/model", "unknown": "unknown", "tbd": "TBD"},
        "bench": "results",
        "note": "note_en",
        "measured": "values: {}",
        "score_cols": {
            "asr": ["FLEURS", "ReazonSpeech", "CV", "Conversation"],
            "tts": ["ms/sentence", "Reading CER", "UTMOS", "First audio ms"],
        },
    },
}

# 英語版で日本語のまま出すと読めない値の置き換え（大きさ・ライセンス欄の定型句）
EN_REPLACE = {
    # 語句（先に置き換える）
    "FunASR モデルライセンス": "FunASR model license",
    "モデル固有": "model-specific",
    "声はキャラクターごとの規約": "per-character terms for voices",
    "声はモデルごと": "voices per model",
    "コード・モデル": "code and model",
    "独立エンジン + HTTP API": "standalone engine + HTTP API",
    "Python 不要": "no Python",
    "要カスタムビルド": "custom build",
    "日本語版の配布は CrispASR": "Japanese build distributed by CrispASR",
    "音声エンコーダ + Qwen3-0.6B デコーダ": "audio encoder + Qwen3-0.6B decoder",
    "AIVMX モデルごと": "per AIVMX model",
    "実行時 VRAM": "runtime VRAM",
    "配布一式": "full download",
    "コーデック": "codec",
    "京都大学": "Kyoto University",
    "国立情報学研究所 LLM-jp": "NII LLM-jp",
    "エンジン": "engine",
    "重み": "weights",
    "活性": "active",
    "以上": "+",
    "要確認": "TBD",
    # 記号（最後に置き換える）
    "約 ": "~",
    "（": " (",
    "）": ")",
    "、": ", ",
    "・": ", ",
    "〜": "-",
}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _en(text: str) -> str:
    out = str(text)
    for src, dst in EN_REPLACE.items():
        out = out.replace(src, dst)
    return out


# 測った値の列。(フィールド, 小数の桁, よいほう)。値は data/models.toml に書き写してある（nan は測っていない）
SCORES = {
    "asr": [("cer", i, 3, min) for i in range(4)],
    "tts": [("tts_ms", None, 0, min), ("tts_cer", None, 3, min), ("utmos", None, 2, max), ("first_voice_ms", None, 0, min)],
}


def _score(row: dict, field: str, idx: int | None) -> float | None:
    v = row.get(field)
    if v is not None and idx is not None:
        v = v[idx] if idx < len(v) else None
    return None if v is None or v != v else float(v)  # nan は v != v


def _score_cells(rows: list[dict], kind: str | None) -> list[list[str]]:
    """行ごとの測った値のセル。各列でいちばんよい値を太字にする。"""
    if not kind:
        return [[] for _ in rows]
    cols = []
    for field, idx, nd, better in SCORES[kind]:
        vals = [_score(r, field, idx) for r in rows]
        known = [v for v in vals if v is not None]
        best = round(better(known), nd) if known else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
            else:
                text = f"{v:.{nd}f}"
                cells.append(f"**{text}**" if round(v, nd) == best else text)
        cols.append(cells)
    return [list(c) for c in zip(*cols)]


def render_table(rows: list[dict], lang: str, prefix: str) -> str:
    lab = LABELS[lang]
    kind = "asr" if any("cer" in r for r in rows) else "tts" if any("tts_ms" in r for r in rows) else None
    extra = lab["score_cols"][kind] if kind else []
    header = lab["header"][:4] + extra + lab["header"][4:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * 4 + "---:|" * len(extra) + "---|" * (len(header) - 4 - len(extra))]
    scores = _score_cells(rows, kind)
    for row, score in zip(rows, scores):
        def conv(field: str) -> str:
            # 英語版は <field>_en があればそれを、無ければ定型句を置き換えたものを使う
            if lang == "en" and row.get(f"{field}_en"):
                return str(row[f"{field}_en"])
            value = str(row.get(field, ""))
            return value if lang == "ja" else _en(value)

        name = f"[{_cell(row['name'])}]({row['url']})"
        note = _cell(row.get(lab["note"], ""))
        measured = row.get("measured_en" if lang == "en" else "measured")
        if measured:
            note += (f"（{lab['measured'].format(measured)}）" if lang == "ja" else f" ({lab['measured'].format(measured)})")
        if row.get("bench"):
            note += f" （[{lab['bench']}]({prefix}{row['bench']})）" if lang == "ja" else f" ([{lab['bench']}]({prefix}{row['bench']}))"
        cells = [
            name,
            _cell(conv("org")),
            lab["ja"].get(row.get("ja", ""), row.get("ja", "")),
            lab["streaming"].get(row.get("streaming", ""), row.get("streaming", "")),
            *score,
            _cell(conv("runtime")),
            _cell(conv("size")),
            _cell(conv("license")),
            lab["commercial"].get(row.get("commercial", ""), row.get("commercial", "")),
            note,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_todo(rows: list[dict], lang: str) -> str:
    lab = LABELS[lang]
    lines = ["| " + " | ".join(lab["todo_header"]) + " |", "|---|---|"]
    for row in rows:
        lines.append(f"| [{_cell(row['name'])}]({row['url']}) | {_cell(row.get(lab['note'], ''))} |")
    return "\n".join(lines)


def build(lang: str, data: dict) -> str:
    template = (ROOT / "templates" / f"README.{lang}.md").read_text(encoding="utf-8")
    out: list[str] = []
    for line in template.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!-- table:") and stripped.endswith("-->"):
            key = stripped[len("<!-- table:"):-len("-->")].strip()
            if key.endswith("_todo"):
                out.append(render_todo(data.get(key, []), lang))
            else:
                out.append(render_table(data.get(key, []), lang, ""))
        else:
            out.append(line)
    header = "<!-- このファイルは scripts/build_readme.py が templates/ と data/models.toml から作る。直接編集しない。 -->\n"
    if lang == "en":
        header = "<!-- Generated by scripts/build_readme.py from templates/ and data/models.toml. Do not edit. -->\n"
    return header + "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="README が最新でなければ 1 で終わる")
    args = parser.parse_args()
    data = tomllib.loads((ROOT / "data" / "models.toml").read_text(encoding="utf-8"))
    targets = {"ja": ROOT / "README.md", "en": ROOT / "README.en.md"}
    stale = []
    for lang, path in targets.items():
        text = build(lang, data)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            if args.check:
                stale.append(path.name)
            else:
                path.write_text(text, encoding="utf-8", newline="\n")
                print(f"wrote {path.name}")
    if stale:
        print("README が最新ではありません: " + ", ".join(stale) + "（python scripts/build_readme.py を実行）")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
