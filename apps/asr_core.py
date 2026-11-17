"""ASR デモの中身（PySide6 版 asr_demo.py と Web 版 asr_web.py が共通で使う）。

マイクの音（このアプリが sounddevice で取る、またはブラウザから送られてくる 16 kHz の float32）を
発話区間検出（Silero VAD）で区切り、区間ごとに認識して確定する。途中経過も出せる。

- 「疑似ストリーミング」: 話している途中も、それまでの音声を一定間隔で認識し直して途中経過を出す
- 「VAD で区切る」: 区間の終わりで 1 回だけ認識して確定する
- 「真のストリーミング」のモデル（Nemotron 3.5 streaming）は、話し始めから音声を少しずつ入れ、
  内部状態を引き継ぎながら途中経過を出す（方式の選択は使わない）

結果は emit(kind, ...) で知らせる（別のスレッドから呼ばれる）:
    ("status", 文)  ("loaded", 文)  ("error", 文)  ("level", 0〜1)
    ("partial", text, ms)  ("final", text, 認識 ms, 音声の秒数, 話し終わりから表示までの ms)
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrkit.backends import BACKENDS, create, create_streaming  # noqa: E402
from asrkit.models import ensure_silero_vad  # noqa: E402

RATE = 16000
WINDOW = 512  # Silero VAD の 1 回分（32 ms）
PREROLL_S = 0.6  # 発話の検出前の音も入れる（検出は実際の話し始めより 0.2〜0.4 秒遅れる）
DEFAULT_MODEL = "sherpa-sensevoice"

Emit = Callable[..., None]


@dataclass
class ListenOptions:
    partial: bool = True  # 疑似ストリーミング（途中経過も出す）
    interval_ms: int = 500  # 途中経過の間隔
    min_silence: float = 0.4  # この秒数の無音で区切る


def clean_text(text: str) -> str:
    """単語の間に空白を入れるモデルがあるので、日本語の文字の隣の空白は詰める。"""
    return re.sub(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])", "", text or "").strip()


def model_choices() -> list[dict]:
    """[{name, label, gpu, streaming}]。GUI と Web のモデルの一覧。"""
    return [{"name": b.name, "label": b.label, "gpu": b.gpu, "streaming": b.streaming} for b in BACKENDS.values()]


def list_mics() -> list[tuple[int, str]]:
    """このアプリが直接使うマイク（既定のマイクを先頭に）。"""
    try:
        import sounddevice as sd

        default = sd.default.device[0]
        mics = [(i, f"{i}: {d['name']}") for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
        mics.sort(key=lambda m: m[0] != default)
        return mics
    except Exception:  # noqa: BLE001
        return []


class AsrSession:
    """モデルの読み込みと、音声 → VAD → 認識のループ（別スレッド）。"""

    def __init__(self, emit: Emit) -> None:
        self.emit = emit
        self.recognize = None
        self.streamer = None  # 真のストリーミングのモデル
        self.model_name: str | None = None
        self.loading = False
        self.audio_q: queue.Queue = queue.Queue()
        self.stream = None  # sounddevice の入力（マイクを直接使うとき）
        self.worker: threading.Thread | None = None
        self.running = threading.Event()
        self.options = ListenOptions()
        self._last_partial = ""
        self._lock = threading.Lock()  # 認識を同時に走らせない（読み込みと認識も）

    @property
    def ready(self) -> bool:
        return self.recognize is not None and not self.loading

    @property
    def listening(self) -> bool:
        return self.running.is_set()

    # ---------------------------------------------------------------- model
    def load(self, name: str, device: str = "cpu", threads: int = 8) -> None:
        if name not in BACKENDS:
            self.emit("error", f"知らないモデルです: {name}")
            return
        if not BACKENDS[name].gpu:
            device = "cpu"
        self.stop()
        self.loading = True

        def run() -> None:
            self.emit("status", f"{BACKENDS[name].label} を読み込み中…（初回はダウンロードに時間がかかります）")
            try:
                with self._lock:
                    self._close_model()
                    t0 = time.perf_counter()
                    if BACKENDS[name].streaming:
                        streamer = create_streaming(name, threads=4, device=device)
                        streamer(np.zeros(RATE // 2, dtype=np.float32))  # ウォームアップ
                        self.streamer, self.recognize = streamer, streamer
                    else:
                        recognize = create(name, device=device, threads=threads)
                        recognize(np.zeros(RATE // 2, dtype=np.float32))  # ウォームアップ
                        self.streamer, self.recognize = None, recognize
                    self.model_name = name
                self.loading = False
                self.emit("loaded", f"読み込み完了（{time.perf_counter() - t0:.1f} 秒）")
            except Exception as ex:  # noqa: BLE001
                self.loading = False
                self.emit("error", f"読み込めませんでした: {type(ex).__name__}: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def _close_model(self) -> None:
        close = getattr(self.recognize, "close", None)  # 前のモデルが起動したサーバーや子プロセスを止める
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        self.recognize = self.streamer = None
        self.model_name = None

    # ---------------------------------------------------------------- listen
    def start(self, options: ListenOptions, mic: int | None = None, use_local_mic: bool = True) -> None:
        """聞き始める。use_local_mic=False なら、音声は feed() で渡す（ブラウザのマイク）。"""
        if not self.ready:
            raise RuntimeError("先にモデルを読み込んでください")
        self.stop()
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(ensure_silero_vad())
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = float(options.min_silence)
        config.silero_vad.min_speech_duration = 0.2
        config.silero_vad.max_speech_duration = 20.0
        config.sample_rate = RATE
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

        self.options = options
        self.audio_q = queue.Queue()
        self.running.set()
        if use_local_mic:
            import sounddevice as sd

            def callback(indata, _frames, _time, _status) -> None:
                self.audio_q.put(indata[:, 0].copy())

            self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=WINDOW,
                                         device=mic, callback=callback)
            self.stream.start()
        self.worker = threading.Thread(target=self._loop, args=(vad,), daemon=True)
        self.worker.start()

    def feed(self, chunk: np.ndarray) -> None:
        """ブラウザなどから受け取った 16 kHz・float32 の音声を入れる。"""
        if self.running.is_set():
            self.audio_q.put(np.asarray(chunk, dtype=np.float32))

    def stop(self) -> None:
        self.running.clear()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=5)
        self.worker = None

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._close_model()

    # ---------------------------------------------------------------- loop
    def _loop(self, vad) -> None:
        opts = self.options
        interval_s = opts.interval_ms / 1000.0
        pending = np.zeros(0, dtype=np.float32)
        preroll = np.zeros(0, dtype=np.float32)
        speech: list[np.ndarray] = []
        speaking = False
        last_partial = 0.0
        self._last_partial = ""
        while self.running.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.emit("level", min(1.0, float(np.sqrt(np.mean(chunk ** 2)) + 1e-9) * 8))
            pending = np.concatenate([pending, chunk])
            while len(pending) >= WINDOW:
                window, pending = pending[:WINDOW], pending[WINDOW:]
                vad.accept_waveform(window)
                if vad.is_speech_detected():
                    if not speaking:
                        speaking = True
                        speech = [preroll.copy()]
                        last_partial = time.perf_counter()
                        if self.streamer is not None:
                            self.streamer.start()
                            self._stream_feed(preroll)
                    speech.append(window)
                    if self.streamer is not None:
                        self._stream_feed(window)
                else:
                    preroll = np.concatenate([preroll, window])[-int(PREROLL_S * RATE):]
                # 話し終わり（無音が続いた）を検出したら、その区間を確定する
                while not vad.empty():
                    segment = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    speaking = False
                    if self.streamer is not None:
                        t0 = time.perf_counter()
                        with self._lock:
                            text = clean_text(self.streamer.finish()) if self.streamer.stream is not None else ""
                        ms = (time.perf_counter() - t0) * 1000
                    else:
                        # VAD の区間は話し始めの検出から始まり、最初の音を落とすことがあるので、
                        # 途中経過と同じく検出の前の音（PREROLL_S）から入れる
                        audio = np.concatenate(speech) if speech else segment
                        text, ms = self._run(audio)
                        speech = []
                    # 実際に話し終わってから表示までは、無音が続くのを待つ時間 + 認識の時間
                    self.emit("final", text, ms, len(segment) / RATE, opts.min_silence * 1000 + ms)
            if self.streamer is None and opts.partial and speaking and time.perf_counter() - last_partial >= interval_s:
                text, ms = self._run(np.concatenate(speech))
                last_partial = time.perf_counter()
                if speaking:
                    self.emit("partial", text, ms)

    def _stream_feed(self, audio: np.ndarray) -> None:
        t0 = time.perf_counter()
        with self._lock:
            text = clean_text(self.streamer.feed(audio))
        if text and text != self._last_partial:
            self._last_partial = text
            self.emit("partial", text, (time.perf_counter() - t0) * 1000)

    def _run(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            with self._lock:
                text = self.recognize(audio) if self.recognize else ""
        except Exception as ex:  # noqa: BLE001
            text = f"（エラー: {ex}）"
        return clean_text(text), (time.perf_counter() - t0) * 1000
