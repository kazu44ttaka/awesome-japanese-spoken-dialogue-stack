"""bench_tts.py で作った音声を ASR で書き起こし、元の文との CER（読み）を付けて表にする。

    python benchmarks/tts/scripts/score_tts.py                     # data/results/tts/*.json をすべて
    python benchmarks/tts/scripts/score_tts.py --asr faster-whisper

ASR は asrkit の名前で選ぶ（既定は faster-whisper の Whisper large-v3-turbo）。先頭と末尾に 0.5 秒の無音を足してから
書き起こす（音声がいきなり始まると先頭を落とす ASR があるため）。声の自然さではなく、読み落とし・繰り返し・
読み間違いを見つけるための検査。ASR 自身の誤りも混ざるので、0.02〜0.03 程度の差は気にしない。
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

from scoring import score_pair, summarize  # noqa: E402


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
    ap.add_argument("--asr", default="faster-whisper")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from asrkit.backends import create

    res_dir = REPO / "data" / "results" / "tts"
    paths = [res_dir / f"{n}.json" for n in args.names] if args.names else sorted(res_dir.glob("*.json"))
    recognize = create(args.asr, device=args.device)
    table = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        scored = []
        for row in data["rows"]:
            hyp = recognize(load16k(REPO / row["wav"]))
            row["asr"] = hyp
            scored.append(score_pair(row["text"], hyp))
            row["cer_kana"] = scored[-1]["cer_kana"]
        s = summarize(scored)
        data["asr_backend"] = args.asr
        data["corpus_cer_kana"] = round(s["corpus_cer_kana"], 4)
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
