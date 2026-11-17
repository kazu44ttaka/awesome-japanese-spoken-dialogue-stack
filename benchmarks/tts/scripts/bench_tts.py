"""TTS エンジンで同じ文を 1 文ずつ合成し、時間と音声を保存する（ttskit を使う。デモアプリと同じ読み込み方）。

    python benchmarks/tts/scripts/bench_tts.py --engine aivisspeech --voice 888753760
    python benchmarks/tts/scripts/bench_tts.py --engine kitten          # 声を省くと最初の声

1 文目の前に 1 回ウォームアップする。結果は data/results/tts/<名前>.json、音声は data/tts_out/<名前>/ に書く。
読み間違いの検査（ASR で書き起こして CER を見る）は score_tts.py で、合成とは別に行う
（合成中に ASR が GPU を使って時間がずれないように）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from ttskit.backends import create, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--voice", default=None, help="声の ID（省くと最初の声）")
    ap.add_argument("--name", default=None, help="結果の名前（省くとエンジン名）")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="エンジンの設定を上書きする（例: --set acceleration=CPU）")
    ap.add_argument("--sentences", type=Path, default=REPO / "benchmarks" / "tts" / "sentences_ja.txt")
    args = ap.parse_args()

    import soundfile as sf

    name = args.name or args.engine
    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    out_dir = REPO / "data" / "tts_out" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    from runtimes.memwatch import MemWatch

    watch = MemWatch().start()  # GPU のメモリは読み込む前を基準にした増え方の最大、RAM は子プロセスも含めた最大
    t0 = time.perf_counter()
    config = load_config()
    for item in args.set:
        key, value = item.split("=", 1)
        config.setdefault(args.engine, {})[key] = value
    engine = create(args.engine, config)
    if hasattr(engine, "start_engine") and not engine.available():
        engine.start_engine(use_gpu=True)  # AivisSpeech / VOICEVOX は入っていれば GPU で起動する
    if hasattr(engine, "available") and not engine.available():
        raise SystemExit(f"{args.engine} を使えません")
    voice = args.voice if args.voice is not None else engine.voices()[0][0]
    engine.synthesize(sentences[0], voice)  # ウォームアップ
    load_s = time.perf_counter() - t0

    rows = []
    for i, text in enumerate(sentences):
        t = time.perf_counter()
        audio, rate = engine.synthesize(text, voice)
        sec = time.perf_counter() - t
        dur = len(audio) / rate if rate else 0.0
        wav = out_dir / f"{i:02d}.wav"
        sf.write(str(wav), audio, rate)
        rows.append({"i": i, "text": text, "synth_s": round(sec, 3), "audio_s": round(dur, 3),
                     "rtf": round(sec / dur, 3) if dur else None, "wav": str(wav.relative_to(REPO)).replace("\\", "/")})
        print(f"{i:02d} {sec * 1000:.0f} ms / {dur:.2f} s | {text}", flush=True)
    mem = watch.stop()
    if hasattr(engine, "close"):
        engine.close()

    synth = sorted(r["synth_s"] for r in rows)
    total_audio = sum(r["audio_s"] for r in rows)
    summary = {
        "engine": args.engine, "name": name, "voice": voice, "settings": args.set, "load_s": round(load_s, 1), **mem, "count": len(rows),
        "mean_synth_ms": round(1000 * sum(synth) / len(synth), 1),
        "p95_synth_ms": round(1000 * synth[min(len(synth) - 1, int(0.95 * len(synth)))], 1),
        "short_synth_ms": round(1000 * sum(r["synth_s"] for r in rows if len(r["text"]) <= 12) /
                                max(1, sum(1 for r in rows if len(r["text"]) <= 12)), 1),
        "rtf": round(sum(synth) / total_audio, 3) if total_audio else None,
        "rows": rows,
    }
    res_dir = REPO / "data" / "results" / "tts"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
