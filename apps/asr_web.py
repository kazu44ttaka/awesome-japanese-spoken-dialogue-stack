"""ASR デモの Web 版。ブラウザのマイクの声をこの PC に送って認識する。

    uv pip install "pyside6-essentials==6.12.0" fastapi "uvicorn[standard]"   # PySide6 は GUI 版だけで使う
    python apps/asr_web.py            # http://127.0.0.1:8765 を開く
    python apps/asr_web.py --host 0.0.0.0 --port 8765 --no-browser

- 中身（VAD で区切って認識する部分）は PySide6 版と同じ apps/asr_core.py
- ブラウザは 48 kHz などで録音した音を 16 kHz に変換し、WebSocket で送る
- ブラウザのマイクは「安全な接続」でしか使えない。この PC で開く（127.0.0.1 / localhost）なら使えるが、
  ほかの PC から http://<IP>:8765 で開くとマイクを使えない（HTTPS にするか、ブラウザの設定で許可する）
- 同時に使えるのは 1 つのページだけ（あとから開いたページが引き継ぐ）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_core import DEFAULT_MODEL, AsrSession, ListenOptions, model_choices  # noqa: E402

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:
    sys.stderr.write('FastAPI / uvicorn が入っていません: uv pip install fastapi "uvicorn[standard]"\n')
    sys.exit(1)

WEB = Path(__file__).resolve().parent / "web"


class Hub:
    """AsrSession の通知を、いまつながっているページへ送る。"""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.owner = 0  # いまの接続の番号
        self.last_status = "モデルを選んで「読み込み」を押してください"
        self.session = AsrSession(self.emit)
        self.lock = threading.Lock()

    def emit(self, kind: str, *data) -> None:
        if kind in ("status", "loaded", "error"):
            self.last_status = data[0]
        if self.loop is None or self.queue is None:
            return
        msg = {"type": kind, "data": list(data)}
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def state(self) -> dict:
        s = self.session
        return {"type": "state", "model": s.model_name, "ready": s.ready, "loading": s.loading,
                "listening": s.listening, "status": self.last_status}


hub = Hub()
app = FastAPI(title="日本語 ASR デモ")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "asr.html")


@app.get("/api/models")
def models() -> JSONResponse:
    return JSONResponse({"models": model_choices(), "default": DEFAULT_MODEL})


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    with hub.lock:
        hub.owner += 1
        me = hub.owner
        hub.session.stop()  # 前のページの聞き取りは止める
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue()
        queue = hub.queue
    await socket.send_json(hub.state())

    async def sender() -> None:
        while True:
            msg = await queue.get()
            if msg["type"] == "level" and not hub.session.listening:
                continue
            await socket.send_json(msg)

    task = asyncio.create_task(sender())
    try:
        while True:
            msg = await socket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if hub.owner != me:  # ほかのページが引き継いだ
                await socket.send_json({"type": "error", "data": ["ほかのページが使い始めたので止めました。再読み込みしてください"]})
                break
            if msg.get("bytes") is not None:
                hub.session.feed(np.frombuffer(msg["bytes"], dtype="<f4"))
                continue
            req = json.loads(msg.get("text") or "{}")
            kind = req.get("type")
            if kind == "load":
                hub.session.load(str(req.get("name", DEFAULT_MODEL)), str(req.get("device", "cpu")), threads=8)
            elif kind == "start":
                try:
                    hub.session.start(ListenOptions(partial=bool(req.get("partial", True)),
                                                    interval_ms=int(req.get("interval_ms", 500)),
                                                    min_silence=float(req.get("min_silence", 0.4))),
                                      use_local_mic=False)
                    await socket.send_json({"type": "listening", "data": [True]})
                except Exception as ex:  # noqa: BLE001
                    await socket.send_json({"type": "error", "data": [f"開始できませんでした: {ex}"]})
            elif kind == "stop":
                hub.session.stop()
                await socket.send_json({"type": "listening", "data": [False]})
            elif kind == "state":
                await socket.send_json(hub.state())
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        if hub.owner == me:
            hub.session.stop()
            hub.queue = None


def main() -> None:
    ap = argparse.ArgumentParser(description="日本語 ASR デモ（Web 版）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="起動してもブラウザを開かない")
    args = ap.parse_args()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"日本語 ASR デモ: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        hub.session.close()


if __name__ == "__main__":
    main()
