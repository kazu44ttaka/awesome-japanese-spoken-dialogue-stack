"""別の Python 環境で ASR モデルを動かす子プロセス（asrkit.backends.VenvASR から起動する）。

    <その環境の python> -m asrkit.worker <モデル名> <device>

標準入力から 1 行 1 つの JSON を読み、標準出力に 1 行 1 つの JSON を返す。
    {"audio": "<16 kHz・float32 の base64>"}  -> {"text": "..."}
最初に {"ready": true} か {"error": "..."} を 1 行返す。
ライブラリが出すログで標準出力が汚れないよう、応答は元の標準出力にだけ書き、ほかは標準エラーへ回す。
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    name, device = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "cuda")
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr  # print するライブラリの出力を応答に混ぜない

    import numpy as np

    from asrkit.backends import create

    try:
        recognize = create(name, device=device)
        out.write(json.dumps({"ready": True}) + "\n")
    except Exception as e:  # noqa: BLE001
        out.write(json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
        return
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            audio = np.frombuffer(base64.b64decode(json.loads(line)["audio"]), dtype=np.float32)
            res = {"text": recognize(audio)}
        except Exception as e:  # noqa: BLE001
            res = {"error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
