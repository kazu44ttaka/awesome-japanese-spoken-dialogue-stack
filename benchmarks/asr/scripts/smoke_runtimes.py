"""ggml 系の実行系の ASR の構成を 1 つずつ立ち上げ、2 発話で動くかを確かめる（モデルの取得もここで済ませる）。

    python benchmarks/asr/scripts/smoke_runtimes.py [構成名 ...]   # 省くと runtimes/catalog.py の全部

結果は 1 行 1 構成の JSON で data/smoke_asr.jsonl に追記する。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from runtimes.catalog import ASR  # noqa: E402
from runtimes.servers import create_asr  # noqa: E402

rows = json.loads((REPO / "data/eval/timing100/manifest.json").read_text(encoding="utf-8"))
pick = [rows[0], next(r for r in rows if r["wav"].startswith("fleurs"))]
audios = [(r["ref"], sf.read(str(REPO / "data/eval/timing100" / r["wav"]), dtype="float32")[0]) for r in pick]
names = sys.argv[1:] or [f"{rt}-{k}" for rt, d in ASR.items() for k in d]
out = REPO / "data" / "smoke_asr.jsonl"
for name in names:
    rt, _, key = name.partition("-")
    rec = {"name": name, "t": time.strftime("%H:%M:%S")}
    t0 = time.time()
    asr = None
    try:
        asr = create_asr(rt, key, device="cuda")
        rec["start_s"] = round(time.time() - t0, 1)
        res = []
        for ref, a in audios:
            t = time.perf_counter()
            hyp = asr(a)
            res.append({"ms": round((time.perf_counter() - t) * 1000), "ref": ref[:30], "hyp": hyp[:60]})
        rec["ok"], rec["res"] = True, res
    except Exception as ex:  # noqa: BLE001
        rec["ok"], rec["error"] = False, f"{type(ex).__name__}: {ex}"[:400]
    finally:  # 失敗してもサーバーを閉じる（残すとメモリを使い続ける）
        if asr is not None:
            asr.close()
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
