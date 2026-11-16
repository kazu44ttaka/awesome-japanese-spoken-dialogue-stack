"""別の Python 環境で TTS エンジンを動かす子プロセス（ttskit.backends.VenvEngine から起動する）。

    <その環境の python> -m ttskit.worker <エンジン名>

標準入力から 1 行 1 つの JSON を読み、標準出力に 1 行 1 つの JSON を返す。
    {"cmd": "voices"}                              -> {"voices": [[id, 表示名], ...]}
    {"cmd": "synth", "text": ..., "voice": ...}    -> {"rate": 24000, "audio": "<float32 の base64>"}
エラーは {"error": "..."}。エンジンの設定（apps/tts_engines.toml の節）は環境変数 TTSKIT_CONFIG（JSON）で受け取る。
エンジンのライブラリが出すログで標準出力が汚れないよう、応答は元の標準出力にだけ書き、ほかは標準エラーへ回す。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    name = sys.argv[1]
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr  # print するライブラリの出力を応答に混ぜない

    import numpy as np

    from ttskit.backends import create_local

    cfg = json.loads(os.environ.get("TTSKIT_CONFIG", "{}"))
    try:
        engine = create_local(name, cfg)
        if hasattr(engine, "available") and not engine.available():
            raise RuntimeError(f"{name} を使えません")
        out.write(json.dumps({"ready": True}) + "\n")
    except Exception as e:  # noqa: BLE001
        out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            if req["cmd"] == "voices":
                res = {"voices": [list(v) for v in engine.voices()]}
            elif req["cmd"] == "synth":
                audio, rate = engine.synthesize(req["text"], req.get("voice", ""), float(req.get("speed", 1.0)))
                pcm = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
                res = {"rate": int(rate), "audio": base64.b64encode(pcm).decode("ascii")}
            else:
                res = {"error": f"unknown cmd: {req['cmd']}"}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
