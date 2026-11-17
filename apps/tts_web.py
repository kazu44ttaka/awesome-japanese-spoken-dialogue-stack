"""TTS デモの Web 版。この PC で合成した音声をブラウザで再生する。

    uv pip install --python .venv-tts fastapi "uvicorn[standard]"
    .venv-tts/Scripts/python apps/tts_web.py          # http://127.0.0.1:8766 を開く
    .venv-tts/Scripts/python apps/tts_web.py --host 0.0.0.0 --no-browser

- 中身（文ごとの合成）は PySide6 版と同じ apps/tts_core.py
- 「文ごとに合成」では、できた文から WebSocket で WAV を送り、ブラウザが順に再生する。
  ページに出す「最初の音まで」は、ボタンを押してからブラウザで再生が始まるまで（通信の時間も含む）
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_core import DEFAULT_ENGINE, SAMPLES, TtsSession, engine_choices, to_wav_bytes  # noqa: E402

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
except ImportError:
    sys.stderr.write('FastAPI / uvicorn が入っていません: uv pip install fastapi "uvicorn[standard]"\n')
    sys.exit(1)

WEB = Path(__file__).resolve().parent / "web"


class Hub:
    """TtsSession の通知を、いまつながっているページへ送る。"""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.owner = 0
        self.last_status = "エンジンを選んで「読み込み」を押してください"
        self.session = TtsSession(self.emit)
        self.lock = threading.Lock()

    def emit(self, kind: str, *data) -> None:
        if kind in ("status", "error"):
            self.last_status = data[0]
        if kind == "audio":
            i, audio, rate = data
            msg = {"type": "audio", "index": i, "wav": to_wav_bytes(audio, rate)}
        elif kind == "loaded":
            voices, secs = data
            self.last_status = f"読み込み完了（{secs:.1f} 秒）・声 {len(voices)} 種類"
            msg = {"type": "loaded", "data": [[list(v) for v in voices], secs]}
        else:
            msg = {"type": kind, "data": list(data)}
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    def state(self) -> dict:
        s = self.session
        return {"type": "state", "engine": s.engine_name, "ready": s.ready, "loading": s.loading,
                "voices": [list(v) for v in s.voices], "status": self.last_status}


hub = Hub()
app = FastAPI(title="日本語 TTS デモ")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "tts.html")


@app.get("/api/engines")
def engines() -> JSONResponse:
    return JSONResponse({"engines": engine_choices(), "default": DEFAULT_ENGINE, "samples": SAMPLES})


@app.get("/api/last.wav")
def last_wav() -> Response:
    joined = hub.session.joined_audio()
    if joined is None:
        return Response(status_code=404)
    audio, rate = joined
    return Response(to_wav_bytes(audio, rate), media_type="audio/wav",
                    headers={"Content-Disposition": 'attachment; filename="tts.wav"'})


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    with hub.lock:
        hub.owner += 1
        me = hub.owner
        hub.session.stop()
        hub.loop = asyncio.get_running_loop()
        hub.queue = asyncio.Queue()
        queue = hub.queue
    await socket.send_json(hub.state())

    async def sender() -> None:
        while True:
            msg = await queue.get()
            if msg["type"] == "audio":
                await socket.send_json({"type": "audio", "index": msg["index"]})
                await socket.send_bytes(msg["wav"])
            else:
                await socket.send_json(msg)

    task = asyncio.create_task(sender())
    try:
        while True:
            req = json.loads(await socket.receive_text())
            if hub.owner != me:
                await socket.send_json({"type": "error", "data": ["ほかのページが使い始めたので止めました。再読み込みしてください"]})
                break
            kind = req.get("type")
            if kind == "load":
                hub.session.load(str(req.get("name", DEFAULT_ENGINE)))
            elif kind == "start_engine":
                hub.session.start_engine(str(req.get("name", DEFAULT_ENGINE)), use_gpu=bool(req.get("use_gpu", True)))
            elif kind == "speak":
                hub.session.speak(str(req.get("text", "")), voice=req.get("voice") or None,
                                  speed=float(req.get("speed", 1.0)), split=bool(req.get("split", True)))
            elif kind == "stop":
                hub.session.stop()
            elif kind == "reset":
                await asyncio.to_thread(hub.session.close)
                hub.last_status = "エンジンを選んで「読み込み」を押してください"
                await socket.send_json(hub.state())
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        if hub.owner == me:
            hub.session.stop()
            hub.queue = None


def main() -> None:
    ap = argparse.ArgumentParser(description="日本語 TTS デモ（Web 版）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true", help="起動してもブラウザを開かない")
    args = ap.parse_args()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"日本語 TTS デモ: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        hub.session.close()


if __name__ == "__main__":
    main()
