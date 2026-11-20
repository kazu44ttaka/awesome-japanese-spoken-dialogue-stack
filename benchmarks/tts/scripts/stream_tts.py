"""TTS をストリーミングで呼び、最初の音が届くまでの時間を測る（TTS のページと同じ 20 文）。

    python benchmarks/tts/scripts/stream_tts.py --engine audiocpp-qwen3-tts-0.6b-base
    # 元の実装（CosyVoice3 の stream=True）は、その環境の python で直接動かす
    .venv-cosyvoice/Scripts/python.exe benchmarks/tts/scripts/stream_tts.py --engine cosyvoice3

1 文ずつ「要求を送ってから」次を測る:
- first_chunk_ms: 音の最初のかたまりが届くまで（無音のこともある）
- first_voice_ms: 声が始まるまで（振幅が 1% を超えた最初のサンプルが届いたとき。先頭の無音を除く）
- total_ms: 最後まで届くまで
結果は data/results/tts_stream/<名前>.json、音声は data/tts_out/stream-<名前>/（score_tts.py で読みを採点できる）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def summarize(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {}
    return {f"mean_{key}": round(statistics.mean(v), 1), f"p50_{key}": round(statistics.median(v), 1),
            f"p95_{key}": round(sorted(v)[max(0, int(len(v) * 0.95 + 0.5) - 1)], 1)}


def run_runtime(engine: str, sentences: list[str], out_dir: Path, warmup: int) -> tuple[list[dict], dict]:
    from runtimes.streaming import create_stream_tts, stream_speech

    runtime, key = engine.split("-", 1)
    t0 = time.perf_counter()
    tts = create_stream_tts(runtime, key)
    voice = tts.voices()[0][0]
    for s in sentences[:warmup]:
        stream_speech(tts, s, voice)
    load_s = time.perf_counter() - t0
    rows = []
    try:
        for i, text in enumerate(sentences):
            r = stream_speech(tts, text, voice)
            audio = r.pop("audio")
            _save(out_dir / f"{i:03d}.wav", audio, r["rate"])
            rows.append({"i": i, "text": text, **r})
            print(i, r["first_chunk_ms"], r["first_voice_ms"], r["total_ms"], flush=True)
    finally:
        tts.close()
    return rows, {"load_s": round(load_s, 1), "device": getattr(tts, "device", None)}


def run_python(engine: str, sentences: list[str], out_dir: Path, warmup: int, threshold: float = 0.01) -> tuple[list[dict], dict]:
    """synthesize_stream を持つ ttskit のエンジン（いまは CosyVoice3）を、この python で直接動かす。"""
    import tomllib

    from ttskit import backends as tb

    cfg = {}
    for f in ("apps/tts_engines.toml", "apps/tts_engines.example.toml"):
        if (REPO / f).is_file():
            cfg = tomllib.loads((REPO / f).read_text(encoding="utf-8")).get(engine, {})
            break
    kana = bool(cfg.get("kana"))
    if kana:  # この環境には pyopenjtalk が無いので、カタカナは親の環境で作ったものを使う（--kana-file）
        cfg = {**cfg, "ref_text": KANA["ref"]}
        sentences = [KANA["sentences"][s] for s in sentences]
    t0 = time.perf_counter()
    eng = getattr(tb, tb._LOCAL[engine])(cfg)
    for s in sentences[:warmup]:
        list(eng.synthesize_stream(s))
    load_s = time.perf_counter() - t0
    rate = eng.sample_rate
    rows = []
    for i, text in enumerate(sentences):
        t = time.perf_counter()
        first_chunk = first_voice = None
        parts = []
        for a in eng.synthesize_stream(text):
            now = time.perf_counter()
            if first_chunk is None and a.size:
                first_chunk = now
            if first_voice is None and np.any(np.abs(a) > threshold):
                first_voice = now
            parts.append(a)
        end = time.perf_counter()
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        _save(out_dir / f"{i:03d}.wav", audio, rate)
        ms = lambda x: None if x is None else round((x - t) * 1000, 1)  # noqa: E731
        r = {"i": i, "text": text, "first_chunk_ms": ms(first_chunk), "first_voice_ms": ms(first_voice),
             "total_ms": ms(end), "audio_s": round(len(audio) / rate, 3), "rate": rate, "chunks": len(parts)}
        rows.append(r)
        print(i, r["first_chunk_ms"], r["first_voice_ms"], r["total_ms"], r["chunks"], flush=True)
    return rows, {"load_s": round(load_s, 1), "device": "cuda"}


def _save(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile as sf

    sf.write(path, audio, rate, subtype="PCM_16")


KANA: dict = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sentences", type=Path, default=REPO / "benchmarks" / "tts" / "sentences_ja.txt")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--kana-file", type=Path, default=None,
                    help="カタカナで渡すエンジン用の {ref, sentences: {文: カタカナ}}（make_kana で作る）")
    args = ap.parse_args()
    name = args.name or args.engine
    sentences = [s.strip() for s in args.sentences.read_text(encoding="utf-8").splitlines() if s.strip()]
    out_dir = REPO / "data" / "tts_out" / f"stream-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.kana_file:
        KANA.update(json.loads(args.kana_file.read_text(encoding="utf-8")))
    if args.engine.split("-", 1)[0] in ("crispasr", "audiocpp", "speechcpp"):
        rows, meta = run_runtime(args.engine, sentences, out_dir, args.warmup)
    else:
        rows, meta = run_python(args.engine, sentences, out_dir, args.warmup)
    for r in rows:  # 採点（score_tts.py）は元の文で行う
        r["text"] = sentences[r["i"]]
    summary = {"engine": args.engine, "name": name, **meta, "count": len(rows),
               **summarize(rows, "first_chunk_ms"), **summarize(rows, "first_voice_ms"), **summarize(rows, "total_ms"),
               "mean_audio_s": round(statistics.mean(r["audio_s"] for r in rows if r["audio_s"]), 2)}
    out = REPO / "data" / "results" / "tts_stream" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
