"""bench_tts.py で作った音声を ASR で書き起こし、元の文との CER（読み）を付けて表にする。

    python benchmarks/tts/scripts/score_tts.py                     # data/results/tts/*.json をすべて
    python benchmarks/tts/scripts/score_tts.py --asr faster-whisper

ASR は asrkit の名前で選ぶ（既定は faster-whisper の Whisper large-v3-turbo）。先頭と末尾に 0.5 秒の無音を足してから
書き起こす（音声がいきなり始まると先頭を落とす ASR があるため）。声の自然さではなく、読み落とし・繰り返し・
読み間違いを見つけるための検査。

ASR 自身の誤りや表記の揺れを TTS の誤りと数えないように、1 文ごとに次のようにする:
- 各 ASR の書き起こしで、読み（かな）で比べた CER と、数字を漢数字にそろえて表記で比べた CER の小さいほう
  （「10分」→「十分」と書き起こされると、読みでは「じゅうぶん」になって誤りになるため）
- 3 つの ASR（既定は faster-whisper・sherpa-cohere・sherpa-reazonspeech-k2）の値の中央値
  （最小値にすると、繰り返しを聞き落とす ASR が 1 つあるだけで本物の誤りが消えるため）
それでも 0.02 程度の差は気にしない。1 つ目の ASR の読みの CER も cer_kana_single として残す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmarks" / "asr" / "scripts"))

from scoring import lev, norm_kana, norm_nopunct, score_pair, summarize  # noqa: E402

_DIGITS = "〇一二三四五六七八九"


def _kanji_number(n: int) -> str:
    if n == 0:
        return "〇"
    out = ""
    for unit_value, unit in ((10 ** 8, "億"), (10 ** 4, "万"), (1, "")):
        part, n = divmod(n, unit_value)
        if not part:
            continue
        s = ""
        for v, u in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
            d, part = divmod(part, v)
            if d:
                s += ("" if d == 1 and u else _DIGITS[d]) + u
        out += s + unit
    return out


def norm_surface(text: str) -> str:
    """句読点を除き、アラビア数字を漢数字にそろえた表記。"""
    import re
    import unicodedata

    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\d+", lambda m: _kanji_number(int(m.group())) if len(m.group()) <= 12 else m.group(), t)
    return norm_nopunct(t)


def robust_cer(ref: str, hyps: list[str]) -> tuple[float, int, int, str]:
    """(CER, 誤り数, 正解の文字数, 採った書き起こし)。ASR ごとに読みと表記の小さいほう、その ASR 間の中央値。"""
    per_asr = []
    for hyp in hyps:
        best = None
        for fn in (norm_kana, norm_surface):
            r, h = fn(ref), fn(hyp)
            e = lev(r, h)
            cand = (e / max(1, len(r)), e, len(r), hyp)
            if best is None or cand[0] < best[0]:
                best = cand
        per_asr.append(best)
    per_asr.sort(key=lambda c: c[0])
    return per_asr[(len(per_asr) - 1) // 2]  # 偶数個のときは小さいほうの中央


def load16k(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        import torch
        import torchaudio.functional as F

        audio = F.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, 16000).numpy()
    pad = np.zeros(8000, dtype=np.float32)
    return np.concatenate([pad, audio, pad])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="結果の名前（省くとすべて）")
    ap.add_argument("--asr", default="faster-whisper,sherpa-cohere,sherpa-reazonspeech-k2", help="ASR の名前（カンマ区切りで複数）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from asrkit.backends import create

    res_dir = REPO / "data" / "results" / "tts"
    paths = [res_dir / f"{n}.json" for n in args.names] if args.names else sorted(res_dir.glob("*.json"))
    names = [a.strip() for a in args.asr.split(",") if a.strip()]
    recognizers = [create(n, device=args.device) for n in names]
    table = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        scored = []
        edits = lens = 0
        for row in data["rows"]:
            audio = load16k(REPO / row["wav"])
            hyps = [rec(audio) for rec in recognizers]
            row["asr_all"] = dict(zip(names, hyps))
            scored.append(score_pair(row["text"], hyps[0]))
            row["cer_kana_single"] = scored[-1]["cer_kana"]
            cer, e, n, picked = robust_cer(row["text"], hyps)
            row["cer_kana"], row["asr"] = round(cer, 4), picked
            edits, lens = edits + e, lens + n
        s = summarize(scored)
        data["asr_backend"] = args.asr
        data["corpus_cer_kana_single"] = round(s["corpus_cer_kana"], 4)
        data["corpus_cer_kana"] = round(edits / max(1, lens), 4)
        data["worst"] = sorted(({"i": r["i"], "cer_kana": r["cer_kana"], "asr": r["asr"]} for r in data["rows"]),
                               key=lambda r: -r["cer_kana"])[:3]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        table.append(data)
        print(f"{data['name']}: CER {data['corpus_cer_kana']:.3f}", flush=True)

    table.sort(key=lambda d: d["mean_synth_ms"])
    print()
    print("| エンジン | 声 | 1 文の合成（平均 ms） | 短い文（ms） | p95 ms | RTF | 読みの CER |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for d in table:
        print(f"| {d['name']} | {d['voice']} | {d['mean_synth_ms']:.0f} | {d['short_synth_ms']:.0f} | {d['p95_synth_ms']:.0f} | "
              f"{d['rtf']:.3f} | {d['corpus_cer_kana']:.3f} |")


if __name__ == "__main__":
    main()
