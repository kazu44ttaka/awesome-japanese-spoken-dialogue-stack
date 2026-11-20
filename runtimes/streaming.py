"""ggml 系の実行系をストリーミングで呼ぶ（TTS は最初の音が出るまで、ASR は話し終えてから文字が確定するまでを測るため）。

TTS（どれも /v1/audio/speech。音は int16・モノラル・ヘッダーなしの PCM がチャンクで届く）
- audio.cpp: モデルを mode=streaming で登録し、"stream": true・"stream_format": "audio"・"response_format": "pcm"
- speech.cpp: "response_format": "pcm"（X-Sample-Rate ヘッダーでサンプリングレート）
- CrispASR: "stream": true。逐次に音を出すのは Qwen3-TTS だけで、ほかは 1 文ずつ作ってから送る

ASR（音声を実時間で少しずつ送り、途中の文字と、送り終えてから確定するまでを見る）
- audio.cpp: POST /v1/audio/transcriptions/live（チャンク転送で PCM を送り、SSE で文字が返る）
- speech.cpp: WebSocket /v1/realtime（OpenAI Realtime の形。24 kHz の PCM を base64 で送る）
- CrispASR: WebSocket /v1/realtime（vLLM Realtime の形。16 kHz。--ws-port を付けると開く）。0.5 秒ごとに全体を認識し直す

    s = create_stream_asr("audiocpp", "nemotron-streaming")
    s.start(); s.feed(chunk_16k_float32) ...; text = s.finish(); s.events  # [(時刻, 途中の文字)]
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
import urllib.parse

import numpy as np

from runtimes.servers import (REF_WAV, OpenAITTS, Server, _ref_text, audiocpp_install, create_tts, free_port,
                              start_audiocpp, start_crispasr, start_speechcpp)


def _pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


# ---------------------------------------------------------------------------- HTTP（チャンク転送）
class _HttpStream:
    """1 本の HTTP/1.1 接続。リクエストを送り、応答の本文を届いた順に読む（chunked にも対応）。"""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        u = urllib.parse.urlparse(base_url)
        self.host, self.port = u.hostname, u.port
        self.sock = socket.create_connection((self.host, self.port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = b""
        self.headers: dict[str, str] = {}
        self.status = 0
        self._chunked = False
        self._left = -1  # 今のチャンク（または Content-Length）の残り

    def send_request(self, method: str, path: str, headers: dict, body: bytes | None = None) -> None:
        lines = [f"{method} {path} HTTP/1.1", f"Host: {self.host}:{self.port}", *[f"{k}: {v}" for k, v in headers.items()]]
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
        self.sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + (body or b""))

    def send_chunk(self, data: bytes) -> None:
        self.sock.sendall(f"{len(data):x}\r\n".encode() + data + b"\r\n")

    def _fill(self) -> bool:
        d = self.sock.recv(65536)
        if not d:
            return False
        self.buf += d
        return True

    def read_headers(self) -> None:
        while b"\r\n\r\n" not in self.buf:
            if not self._fill():
                raise ConnectionError("応答が返る前に切れました")
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        self.status = int(lines[0].split()[1])
        self.headers = {k.strip().lower(): v.strip() for k, v in (ln.split(":", 1) for ln in lines[1:] if ":" in ln)}
        self._chunked = "chunked" in self.headers.get("transfer-encoding", "").lower()
        self._left = int(self.headers.get("content-length", "-1")) if not self._chunked else -1

    def read_some(self) -> bytes | None:
        """本文を届いた分だけ返す。終わりなら None。"""
        if not self._chunked:
            if self._left == 0:
                return None
            if not self.buf and not self._fill():
                return None
            take = self.buf if self._left < 0 else self.buf[: self._left]
            self.buf = self.buf[len(take):]
            if self._left > 0:
                self._left -= len(take)
            return take
        while True:
            if self._left <= 0:
                while b"\r\n" not in self.buf:
                    if not self._fill():
                        return None
                line, self.buf = self.buf.split(b"\r\n", 1)
                if not line.strip():
                    continue
                self._left = int(line.split(b";")[0], 16)
                if self._left == 0:
                    return None
            if not self.buf and not self._fill():
                return None
            take = self.buf[: self._left]
            self.buf = self.buf[len(take):]
            self._left -= len(take)  # チャンクの後ろの CRLF は、次のサイズの行を探すときに空行として読み飛ばす
            if take:
                return take

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------- TTS
def stream_speech(tts: OpenAITTS, text: str, voice: str, threshold: float = 0.01) -> dict:
    """1 文をストリーミングで合成し、最初の音が届くまで（first_chunk_ms）・声が始まるまで（first_voice_ms）・
    全部届くまで（total_ms）を返す。声の始まりは振幅が threshold を超えた最初のサンプル（先頭の無音を除く）。"""
    runtime = tts.runtime
    if tts.kana:
        from runtimes.servers import to_kana
        text = to_kana(text)
    body = {**({"model": tts.model} if tts.model else {}), "input": text, **tts.options}
    if voice:
        body["voice"] = voice
    if runtime == "audiocpp":
        body.update({"stream": True, "stream_format": "audio", "response_format": "pcm"})
    elif runtime == "speechcpp":
        body.update({"response_format": "pcm"})
    else:  # crispasr
        body.update({"stream": True, "response_format": "pcm"})
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    h = _HttpStream(tts.base_url)
    t0 = time.perf_counter()
    first_chunk = first_voice = None
    pcm = bytearray()
    try:
        h.send_request("POST", "/v1/audio/speech", {"Content-Type": "application/json"}, raw)
        h.read_headers()
        if h.status != 200:
            err = b""
            while (d := h.read_some()) is not None:
                err += d
            raise RuntimeError(f"HTTP {h.status}: {err[:300].decode('utf-8', 'replace')}")
        lim = int(threshold * 32767)
        while (d := h.read_some()) is not None:
            now = time.perf_counter()
            if first_chunk is None and d:
                first_chunk = now
            start = len(pcm) // 2
            pcm += d
            if first_voice is None and len(pcm) >= 2:
                a = np.frombuffer(bytes(pcm[: len(pcm) // 2 * 2]), dtype="<i2")
                hit = np.nonzero(np.abs(a[start:]) > lim)[0]
                if hit.size:
                    first_voice = now
        t_end = time.perf_counter()
    finally:
        h.close()
    rate = int(h.headers.get("x-sample-rate", 0)) or tts.stream_rate
    audio = np.frombuffer(bytes(pcm[: len(pcm) // 2 * 2]), dtype="<i2").astype(np.float32) / 32768.0
    ms = lambda t: None if t is None else round((t - t0) * 1000, 1)  # noqa: E731
    return {"first_chunk_ms": ms(first_chunk), "first_voice_ms": ms(first_voice), "total_ms": ms(t_end),
            "audio_s": round(len(audio) / rate, 3) if rate else None, "rate": rate, "audio": audio}


def create_stream_tts(runtime: str, key: str, device: str = "cuda", threads: int = 8) -> OpenAITTS:
    """create_tts と同じ構成をストリーミング向けに立ち上げる（audio.cpp は mode=streaming で登録する）。
    stream_rate（PCM のサンプリングレート）は、最初に 1 回ふつうに合成した WAV のヘッダーから取る。"""
    tts = create_tts(runtime, key, device, threads, mode="streaming")
    tts.runtime = runtime
    _, tts.stream_rate = tts.synthesize("こんにちは。", tts.voices()[0][0])
    return tts


# ---------------------------------------------------------------------------- ASR
class _StreamASRBase:
    """start → feed（何回でも）→ finish。events は (perf_counter の時刻, その時点の文字) の並び。"""

    def __init__(self) -> None:
        self.events: list[tuple[float, str]] = []
        self.final: str | None = None
        self.t_final: float | None = None
        self._done = threading.Event()
        self.server: Server | None = None

    def _partial(self, text: str) -> None:
        if text and (not self.events or self.events[-1][1] != text):
            self.events.append((time.perf_counter(), text))

    def _finalize(self, text: str) -> None:
        self.final, self.t_final = text, time.perf_counter()
        self._done.set()

    def finish(self, timeout: float = 120.0) -> str:
        self._end()
        if not self._done.wait(timeout):
            raise TimeoutError("確定した文字が返りませんでした")
        return (self.final or "").strip()

    def close(self) -> None:
        if self.server:
            self.server.close()
            self.server = None


class AudioCppLiveASR(_StreamASRBase):
    """audio.cpp の POST /v1/audio/transcriptions/live。"""

    def __init__(self, base_url: str, model: str, language: str = "ja") -> None:
        super().__init__()
        self.base_url, self.model, self.language = base_url, model, language

    def start(self) -> None:
        self.events, self.final, self.t_final = [], None, None
        self._done.clear()
        self.text = ""
        q = urllib.parse.urlencode({"model": self.model, "sample_rate": 16000, "channels": 1, "sample_format": "s16le",
                                    "language": self.language})
        self.h = _HttpStream(self.base_url)
        self.h.send_request("POST", f"/v1/audio/transcriptions/live?{q}",
                            {"Content-Type": "application/octet-stream", "Transfer-Encoding": "chunked"})
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            self.h.read_headers()
            if self.h.status != 200:
                err = b""
                while (d := self.h.read_some()) is not None:
                    err += d
                self._finalize(f"[error] HTTP {self.h.status}: {err[:300].decode('utf-8', 'replace')}")
                return
            pending = b""
            while (d := self.h.read_some()) is not None:
                pending += d
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        continue
                    ev = json.loads(payload.decode("utf-8", "replace"))
                    t = ev.get("type", "")
                    if t == "transcript.text.delta":
                        self.text += ev.get("delta", "")
                        self._partial(self.text)
                    elif t == "transcript.text.partial":
                        self._partial(self.text + ev.get("text", ""))
                    elif t == "transcript.text.done":
                        self._finalize(ev.get("text", self.text))
                    elif t == "error" or "error" in ev:
                        self._finalize(f"[error] {ev}")
        except Exception as ex:  # noqa: BLE001
            if not self._done.is_set():
                self._finalize(f"[error] {type(ex).__name__}: {ex}")
        finally:
            if not self._done.is_set():
                self._finalize(self.text)

    def feed(self, audio: np.ndarray) -> None:
        if self._done.is_set():  # サーバーがエラーを返して切った（finish でその文を返す）
            return
        try:
            self.h.send_chunk(_pcm16(audio))
        except OSError:
            self._done.wait(5)

    def _end(self) -> None:
        if not self._done.is_set():
            try:
                self.h.sock.sendall(b"0\r\n\r\n")
            except OSError:
                pass

    def finish(self, timeout: float = 120.0) -> str:
        try:
            return super().finish(timeout)
        finally:
            self.h.close()


class _RealtimeWS(_StreamASRBase):
    """OpenAI Realtime の形の WebSocket（speech.cpp の /v1/realtime・CrispASR の vLLM Realtime）。"""

    rate = 16000

    def __init__(self, ws_url: str, session: dict | None = None) -> None:
        super().__init__()
        self.ws_url, self.session = ws_url, session

    def start(self) -> None:
        from websockets.sync.client import connect

        self.events, self.final, self.t_final = [], None, None
        self._done.clear()
        self.text = ""
        self.ws = connect(self.ws_url, max_size=None, open_timeout=60)
        if self.session:
            self.ws.send(json.dumps({"type": "session.update", "session": self.session}))
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for msg in self.ws:
                ev = json.loads(msg)
                t = ev.get("type", "")
                if t.endswith("input_audio_transcription.delta"):
                    self._on_delta(ev.get("delta", ""))
                elif t.endswith("input_audio_transcription.completed"):
                    self._finalize(ev.get("transcript", self.text))
                    return
                elif t.endswith("failed") or t == "error":
                    self._finalize(f"[error] {json.dumps(ev, ensure_ascii=False)[:300]}")
                    return
        except Exception as ex:  # noqa: BLE001
            if not self._done.is_set():
                self._finalize(f"[error] {type(ex).__name__}: {ex}")

    def _on_delta(self, delta: str) -> None:
        self.text += delta
        self._partial(self.text)

    def feed(self, audio: np.ndarray) -> None:
        if self.rate != 16000:
            n = int(round(len(audio) * self.rate / 16000))
            audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
        self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(_pcm16(audio)).decode()}))

    def _end(self) -> None:
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def finish(self, timeout: float = 120.0) -> str:
        try:
            return super().finish(timeout)
        finally:
            self.ws.close()


class SpeechCppRealtimeASR(_RealtimeWS):
    rate = 24000


class CrispRealtimeASR(_RealtimeWS):
    """CrispASR の vLLM Realtime。新しい文字が前の続きでないときは全体を送り直すので、delta が前の文字で
    始まっていれば置き換え、そうでなければ足す。"""

    def _on_delta(self, delta: str) -> None:
        self.text = delta if (self.text and delta.startswith(self.text)) or len(delta) >= len(self.text) * 2 \
            else self.text + delta
        self._partial(self.text)


def create_stream_asr(runtime: str, key: str, device: str = "cuda", threads: int = 8):
    from runtimes.catalog import ASR

    spec = ASR[runtime][key]
    lang = spec.get("language", "ja")
    if runtime == "audiocpp":
        path = audiocpp_install(spec["package"])
        server, url = start_audiocpp([{"id": key, "family": spec["family"], "path": str(path), "task": "asr",
                                       "mode": "streaming", **spec.get("load", {}), **spec.get("stream_load", {})}],
                                     device, threads)
        s = AudioCppLiveASR(url, key, lang)
    elif runtime == "speechcpp":
        server, url = start_speechcpp([spec["model"]], device, threads, spec.get("args"))
        session = {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                                                "transcription": {"language": lang},
                                                                "turn_detection": None}}}
        s = SpeechCppRealtimeASR(url.replace("http", "ws") + "/v1/realtime", session)
    elif runtime == "crispasr":
        ws_port = free_port()
        server, url = start_crispasr(spec["backend"], spec.get("model", "auto"), device, threads,
                                     [*(spec.get("args") or []), "--ws-port", str(ws_port)],
                                     url=spec.get("url"), quant=spec.get("quant", "q8_0"), language=lang)
        # 0.8.54 では vLLM Realtime は HTTP と同じ番号の /v1/realtime で、0.0.0.0 に開く（--ws-port の方は Whisper
        # 専用の生 PCM）。HTTP は 127.0.0.1 に開くので、127.0.0.1 へつなぐと HTTP 側に届いて 404 になる。
        # 同じループバックの 127.0.0.2 へつなぐと 0.0.0.0 の方に届く
        s = CrispRealtimeASR(url.replace("http://127.0.0.1", "ws://127.0.0.2") + "/v1/realtime")
    else:
        raise ValueError(runtime)
    s.server = server
    s.device = ("vulkan" if runtime == "speechcpp" else "cuda") if device.startswith("cuda") else "cpu"
    return s


__all__ = ["create_stream_tts", "stream_speech", "create_stream_asr", "REF_WAV", "_ref_text"]
