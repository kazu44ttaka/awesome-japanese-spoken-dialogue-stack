"""マイクの声をその場で認識して、モデルごとの精度と速さを試す小さなアプリ。

    uv pip install sounddevice      # ほかに使うモデルの分（sherpa-onnx / transformers / funasr）
    python apps/asr_demo.py

- 「VAD で区切る」: 発話区間検出（Silero VAD）で話し終わりを見つけ、区間ごとに認識して確定する
- 「疑似ストリーミング」: 話している途中も、それまでの音声を一定間隔で認識し直して途中経過を出す
  （区間の終わりでは同じく確定する）。モデルが間隔より遅いと、途中経過はその分まばらになる
- 確定した行には、音声の長さ・認識にかかった時間・話し終わりから表示までの時間（無音で区切る秒数 + 認識）を出す

真のストリーミング（音声を少しずつ入れて内部状態を引き継ぐ方式）のモデルは未対応。
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrkit.backends import BACKENDS, create  # noqa: E402
from asrkit.models import ensure_silero_vad  # noqa: E402

RATE = 16000
WINDOW = 512  # Silero VAD の 1 回分（32 ms）
PREROLL_S = 0.3  # 発話の検出前の音も途中経過に入れる


class Engine:
    """マイク → VAD → 認識を別スレッドで回し、結果を events に入れる。"""

    def __init__(self, events: queue.Queue) -> None:
        self.events = events
        self.recognize = None
        self.audio_q: queue.Queue = queue.Queue()
        self.stream = None
        self.worker: threading.Thread | None = None
        self.running = threading.Event()
        self.min_silence = 0.4

    # ---------------------------------------------------------------- model
    def load(self, name: str, device: str, threads: int) -> None:
        def run() -> None:
            self.events.put(("status", f"{BACKENDS[name].label} を読み込み中…（初回はダウンロードに時間がかかります）"))
            try:
                t0 = time.perf_counter()
                recognize = create(name, device=device, threads=threads)
                recognize(np.zeros(RATE // 2, dtype=np.float32))  # ウォームアップ
                self.recognize = recognize
                self.events.put(("loaded", f"読み込み完了（{time.perf_counter() - t0:.1f} 秒）"))
            except Exception as ex:  # noqa: BLE001
                self.events.put(("error", f"読み込めませんでした: {type(ex).__name__}: {ex}"))

        threading.Thread(target=run, daemon=True).start()

    # ---------------------------------------------------------------- run
    def start(self, mic, partial: bool, interval_ms: int, min_silence: float) -> None:
        import sherpa_onnx
        import sounddevice as sd

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(ensure_silero_vad())
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = float(min_silence)
        config.silero_vad.min_speech_duration = 0.2
        config.silero_vad.max_speech_duration = 20.0
        config.sample_rate = RATE
        vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

        self.audio_q = queue.Queue()
        self.min_silence = float(min_silence)
        self.running.set()

        def callback(indata, _frames, _time, _status) -> None:
            self.audio_q.put(indata[:, 0].copy())

        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=WINDOW,
                                     device=mic, callback=callback)
        self.stream.start()
        self.worker = threading.Thread(target=self._loop, args=(vad, partial, interval_ms / 1000.0), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running.clear()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _loop(self, vad, partial: bool, interval_s: float) -> None:
        pending = np.zeros(0, dtype=np.float32)
        preroll = np.zeros(0, dtype=np.float32)
        speech: list[np.ndarray] = []
        speaking = False
        last_partial = 0.0
        while self.running.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.events.put(("level", float(np.sqrt(np.mean(chunk ** 2)) + 1e-9)))
            pending = np.concatenate([pending, chunk])
            while len(pending) >= WINDOW:
                window, pending = pending[:WINDOW], pending[WINDOW:]
                vad.accept_waveform(window)
                if vad.is_speech_detected():
                    if not speaking:
                        speaking = True
                        speech = [preroll.copy()]
                        last_partial = time.perf_counter()
                    speech.append(window)
                else:
                    preroll = np.concatenate([preroll, window])[-int(PREROLL_S * RATE):]
                # 話し終わり（無音が続いた）を検出したら、その区間を確定する
                while not vad.empty():
                    segment = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    speaking = False
                    text, ms = self._run(segment)
                    # 実際に話し終わってから表示までは、無音が続くのを待つ時間 + 認識の時間
                    self.events.put(("final", text, ms, len(segment) / RATE, self.min_silence * 1000 + ms))
            if partial and speaking and time.perf_counter() - last_partial >= interval_s:
                text, ms = self._run(np.concatenate(speech))
                last_partial = time.perf_counter()
                if speaking:
                    self.events.put(("partial", text, ms))

    def _run(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            text = self.recognize(audio) if self.recognize else ""
        except Exception as ex:  # noqa: BLE001
            text = f"（エラー: {ex}）"
        # 単語の間に空白を入れるモデルがあるので、日本語の文字の隣の空白は詰める
        text = re.sub(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])", "", text)
        return text.strip(), (time.perf_counter() - t0) * 1000


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("日本語 ASR デモ")
        root.geometry("900x600")
        self.events: queue.Queue = queue.Queue()
        self.engine = Engine(self.events)

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="モデル").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value=BACKENDS["sherpa-sensevoice"].label)
        self.model_box = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=48,
                                      values=[b.label for b in BACKENDS.values()])
        self.model_box.grid(row=0, column=1, sticky="w", padx=4)
        self.model_box.bind("<<ComboboxSelected>>", lambda _e: self._sync_device())
        ttk.Label(top, text="実行").grid(row=0, column=2, sticky="e")
        self.device_var = tk.StringVar(value="cpu")
        self.device_box = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=6,
                                       values=["cpu", "cuda"])
        self.device_box.grid(row=0, column=3, padx=4)
        self.load_btn = ttk.Button(top, text="読み込み", command=self._load)
        self.load_btn.grid(row=0, column=4, padx=4)

        ttk.Label(top, text="マイク").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.mics = self._list_mics()
        self.mic_var = tk.StringVar(value=self.mics[0][1] if self.mics else "")
        ttk.Combobox(top, textvariable=self.mic_var, state="readonly", width=48,
                     values=[m[1] for m in self.mics]).grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))

        opts = ttk.Frame(root, padding=(8, 0))
        opts.pack(fill="x")
        self.partial_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(opts, text="疑似ストリーミング（途中経過も出す）", variable=self.partial_var, value=True).pack(side="left")
        ttk.Radiobutton(opts, text="VAD で区切る（確定だけ）", variable=self.partial_var, value=False).pack(side="left", padx=8)
        ttk.Label(opts, text="途中経過の間隔(ms)").pack(side="left", padx=(12, 2))
        self.interval_var = tk.IntVar(value=500)
        ttk.Spinbox(opts, from_=100, to=3000, increment=100, width=6, textvariable=self.interval_var).pack(side="left")
        ttk.Label(opts, text="無音で区切る(秒)").pack(side="left", padx=(12, 2))
        self.silence_var = tk.DoubleVar(value=0.4)
        ttk.Spinbox(opts, from_=0.1, to=2.0, increment=0.1, width=5, textvariable=self.silence_var).pack(side="left")

        ctrl = ttk.Frame(root, padding=8)
        ctrl.pack(fill="x")
        self.start_btn = ttk.Button(ctrl, text="開始", command=self._start, state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(ctrl, text="停止", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        ttk.Button(ctrl, text="クリア", command=lambda: self.log.delete("1.0", "end")).pack(side="left")
        ttk.Label(ctrl, text="入力").pack(side="left", padx=(16, 4))
        self.level = ttk.Progressbar(ctrl, length=160, maximum=1.0)
        self.level.pack(side="left")
        self.status = ttk.Label(ctrl, text="モデルを選んで「読み込み」を押してください")
        self.status.pack(side="left", padx=12)

        self.partial = ttk.Label(root, text="", foreground="#777", wraplength=860, padding=(10, 4))
        self.partial.pack(fill="x")
        self.log = ScrolledText(root, font=("Yu Gothic UI", 11), wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.tag_configure("meta", foreground="#888")

        self._sync_device()
        root.after(50, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _list_mics() -> list[tuple[int, str]]:
        try:
            import sounddevice as sd

            default = sd.default.device[0]
            mics = [(i, f"{i}: {d['name']}") for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
            mics.sort(key=lambda m: m[0] != default)  # 既定のマイクを先頭に
            return mics
        except Exception:  # noqa: BLE001
            return []

    def _backend(self) -> str:
        return next(n for n, b in BACKENDS.items() if b.label == self.model_var.get())

    def _sync_device(self) -> None:
        info = BACKENDS[self._backend()]
        if not info.gpu:
            self.device_var.set("cpu")
            self.device_box.configure(state="disabled")
        else:
            self.device_box.configure(state="readonly")
        self.start_btn.configure(state="disabled")

    # ---------------------------------------------------------------- actions
    def _load(self) -> None:
        self._stop()
        self.load_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.engine.load(self._backend(), self.device_var.get(), threads=8)

    def _start(self) -> None:
        mic = next((i for i, label in self.mics if label == self.mic_var.get()), None)
        try:
            self.engine.start(mic, self.partial_var.get(), int(self.interval_var.get()), float(self.silence_var.get()))
        except Exception as ex:  # noqa: BLE001
            self.status.configure(text=f"開始できませんでした: {ex}")
            return
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="聞いています…")

    def _stop(self) -> None:
        self.engine.stop()
        self.stop_btn.configure(state="disabled")
        if self.engine.recognize is not None:
            self.start_btn.configure(state="normal")
        self.partial.configure(text="")
        self.level.configure(value=0)

    def _close(self) -> None:
        self.engine.stop()
        self.root.destroy()

    def _poll(self) -> None:
        try:
            while True:
                kind, *data = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=data[0])
                elif kind == "loaded":
                    self.status.configure(text=data[0])
                    self.load_btn.configure(state="normal")
                    self.start_btn.configure(state="normal")
                elif kind == "error":
                    self.status.configure(text=data[0])
                    self.load_btn.configure(state="normal")
                elif kind == "level":
                    self.level.configure(value=min(1.0, data[0] * 8))
                elif kind == "partial":
                    text, ms = data
                    self.partial.configure(text=f"途中経過（{ms:.0f} ms）: {text}")
                elif kind == "final":
                    text, ms, audio_s, latency = data
                    self.partial.configure(text="")
                    stamp = time.strftime("%H:%M:%S")
                    self.log.insert("end", f"{stamp}  {text or '（認識結果なし）'}\n")
                    self.log.insert("end", f"          音声 {audio_s:.1f} 秒・認識 {ms:.0f} ms・"
                                           f"話し終わりから表示まで 約 {latency:.0f} ms\n", "meta")
                    self.log.see("end")
        except queue.Empty:
            pass
        self.root.after(50, self._poll)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
