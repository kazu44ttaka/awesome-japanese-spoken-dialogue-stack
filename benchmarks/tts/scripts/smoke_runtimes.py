"""ggml 系の実行系の TTS の構成を 1 つずつ立ち上げ、2 文を合成して動くかを確かめる（モデルの取得もここで済ませる）。

    python benchmarks/tts/scripts/smoke_runtimes.py [構成名 ...]   # 省くと runtimes/catalog.py の全部

結果は 1 行 1 構成の JSON で data/smoke_tts.jsonl に追記し、音声は data/tts_out/smoke/ に書く。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from runtimes.catalog import TTS  # noqa: E402
from ttskit.backends import create  # noqa: E402

texts = ["こんにちは。", "明日の会議は午後3時から始まります。"]
names = sys.argv[1:] or [f"{rt}-{k}" for rt, d in TTS.items() for k in d]
out = REPO / "data" / "smoke_tts.jsonl"
wav_dir = REPO / "data" / "tts_out" / "smoke"
wav_dir.mkdir(parents=True, exist_ok=True)
for name in names:
    rec = {"name": name, "t": time.strftime("%H:%M:%S")}
    t0 = time.time()
    tts = None
    try:
        tts = create(name, {})
        rec["start_s"] = round(time.time() - t0, 1)
        voice = tts.voices()[0][0]
        res = []
        for i, text in enumerate(texts):
            t = time.perf_counter()
            audio, rate = tts.synthesize(text, voice)
            res.append({"ms": round((time.perf_counter() - t) * 1000), "sec": round(len(audio) / rate, 2), "rate": rate})
            sf.write(str(wav_dir / f"{name}-{i}.wav"), audio, rate)
        rec["ok"], rec["res"] = True, res
    except Exception as ex:  # noqa: BLE001
        rec["ok"], rec["error"] = False, f"{type(ex).__name__}: {ex}"[:400]
    finally:  # 失敗してもサーバーを閉じる（残すとメモリを使い続ける）
        if tts is not None and hasattr(tts, "close"):
            tts.close()
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
