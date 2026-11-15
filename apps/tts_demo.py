"""日本語 TTS を選んで文を読み上げ、最初の音が出るまでの時間と速さを試す小さなアプリ。

    python apps/tts_demo.py

エンジンの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピー）。必要なパッケージと入れ方は README を参照。

- AivisSpeech / VOICEVOX はエンジンを起動しておく（入っていれば「エンジンを起動」で起動できる。
  「GPU を使う」で --use_gpu を付ける。すでに起動しているエンジンはそのまま使う）
- 「文ごとに合成して、できた文から再生」を入れると、1 文目ができた時点で再生を始める（音声対話と同じ）。
  最初の音が出るまでの時間（TTFA）が、全文をまとめて合成するときと比べてどれだけ短くなるかが分かる
- 文ごとに、合成にかかった時間・音声の長さ・RTF（合成時間 ÷ 音声の長さ）を出す
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttskit.backends import BACKENDS, VoicevoxCompatible, create, load_config, split_sentences  # noqa: E402

SAMPLES = {
    "あいさつ": "こんにちは。今日はいい天気ですね。何かお手伝いできることはありますか？",
    "数字・日時": "次の会議は8月19日午後3時15分です。売上は1,234万円、気温は31.5度でした。",
    "長めの文": "音声合成の品質を確認しています。東京駅から新宿駅まで電車で移動し、橋の端を、箸を持って歩きました。"
              "固有名詞や数字を含む文章でも、自然な間と正しいアクセントで読み上げられることが重要です。",
    "音声対話の返事": "それは大変でしたね。お昼を食べられなかったなら、今日は少し早めに休んでください。",
}


class Player:
    """合成できた音声を順番に再生する（再生は別スレッド）。"""

    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.stop_flag = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        import sounddevice as sd

        while True:
            audio, rate = self.q.get()
            if self.stop_flag.is_set():
                continue
            sd.play(audio, rate)
            sd.wait()

    def play(self, audio: np.ndarray, rate: int) -> None:
        self.q.put((audio, rate))

    def stop(self) -> None:
        import sounddevice as sd

        self.stop_flag.set()
        while not self.q.empty():
            self.q.get_nowait()
        sd.stop()

    def resume(self) -> None:
        self.stop_flag.clear()


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("日本語 TTS デモ")
        root.geometry("900x640")
        self.events: queue.Queue = queue.Queue()
        self.tts = None
        self.config = load_config()
        self.player = Player()
        self.cancel = threading.Event()
        self.last_audio: list[tuple[np.ndarray, int]] = []

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="エンジン").grid(row=0, column=0, sticky="w")
        self.engine_var = tk.StringVar(value=BACKENDS["aivisspeech"].label)
        engine_box = ttk.Combobox(top, textvariable=self.engine_var, state="readonly", width=44,
                                  values=[b.label for b in BACKENDS.values()])
        engine_box.grid(row=0, column=1, sticky="w", padx=4)
        engine_box.bind("<<ComboboxSelected>>", lambda _e: self._reset())
        self.load_btn = ttk.Button(top, text="読み込み", command=self._load)
        self.load_btn.grid(row=0, column=2, padx=4)
        self.start_engine_btn = ttk.Button(top, text="エンジンを起動", command=self._start_engine)
        self.start_engine_btn.grid(row=0, column=3, padx=4)
        self.gpu_var = tk.BooleanVar(value=True)
        self.gpu_check = ttk.Checkbutton(top, text="GPU を使う", variable=self.gpu_var)
        self.gpu_check.grid(row=0, column=4, padx=4)

        ttk.Label(top, text="声").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.voice_var = tk.StringVar()
        self.voice_box = ttk.Combobox(top, textvariable=self.voice_var, state="readonly", width=44)
        self.voice_box.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(top, text="速さ").grid(row=1, column=2, sticky="e", pady=(6, 0))
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(top, from_=0.5, to=2.0, increment=0.1, width=5, textvariable=self.speed_var).grid(
            row=1, column=3, sticky="w", pady=(6, 0))
        self.voice_ids: list[str] = []

        mid = ttk.Frame(root, padding=(8, 0))
        mid.pack(fill="x")
        ttk.Label(mid, text="例文").pack(side="left")
        sample_box = ttk.Combobox(mid, state="readonly", width=16, values=list(SAMPLES))
        sample_box.pack(side="left", padx=4)
        sample_box.bind("<<ComboboxSelected>>", lambda _e: self._set_text(SAMPLES[sample_box.get()]))
        self.split_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mid, text="文ごとに合成して、できた文から再生（音声対話向け）",
                        variable=self.split_var).pack(side="left", padx=12)

        self.text = tk.Text(root, height=5, font=("Yu Gothic UI", 11), wrap="word")
        self.text.pack(fill="x", padx=8, pady=6)
        self._set_text(SAMPLES["あいさつ"])

        ctrl = ttk.Frame(root, padding=(8, 0))
        ctrl.pack(fill="x")
        self.speak_btn = ttk.Button(ctrl, text="合成して再生", command=self._speak, state="disabled")
        self.speak_btn.pack(side="left")
        ttk.Button(ctrl, text="停止", command=self._stop).pack(side="left", padx=4)
        self.save_btn = ttk.Button(ctrl, text="WAV で保存", command=self._save, state="disabled")
        self.save_btn.pack(side="left")
        self.status = ttk.Label(ctrl, text="エンジンを選んで「読み込み」を押してください")
        self.status.pack(side="left", padx=12)

        self.summary = ttk.Label(root, text="", font=("Yu Gothic UI", 11, "bold"), padding=(10, 6))
        self.summary.pack(fill="x")
        self.log = ScrolledText(root, font=("Yu Gothic UI", 10), wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log.tag_configure("meta", foreground="#888")

        self._reset()
        root.after(50, self._poll)
        root.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- helpers
    def _backend(self) -> str:
        return next(n for n, b in BACKENDS.items() if b.label == self.engine_var.get())

    def _set_text(self, text: str) -> None:
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)

    def _reset(self) -> None:
        self._close_tts()
        self.tts = None
        self.speak_btn.configure(state="disabled")
        info = BACKENDS[self._backend()]
        is_http = self._backend() in ("aivisspeech", "voicevox")
        self.start_engine_btn.configure(state="normal" if is_http else "disabled")
        self.gpu_check.configure(state="normal" if is_http else "disabled")
        self.status.configure(text=info.note)

    # ---------------------------------------------------------------- actions
    def _load(self) -> None:
        name = self._backend()
        self.load_btn.configure(state="disabled")
        self.status.configure(text="読み込み中…（初回はモデルのダウンロードに時間がかかります）")

        def run() -> None:
            try:
                t0 = time.perf_counter()
                tts = create(name, self.config)
                if not tts.available():
                    raise RuntimeError("エンジンにつながりません。エンジンを起動してから読み込んでください")
                voices = tts.voices()
                if not voices:
                    raise RuntimeError("声が見つかりません")
                tts.synthesize("テストです。", voice=voices[0][0])  # ウォームアップ
                self.events.put(("loaded", tts, voices, time.perf_counter() - t0))
            except Exception as ex:  # noqa: BLE001
                self.events.put(("error", f"読み込めませんでした: {type(ex).__name__}: {ex}"))

        threading.Thread(target=run, daemon=True).start()

    def _start_engine(self) -> None:
        name = self._backend()
        self.status.configure(text="エンジンを起動しています…（1 分ほどかかることがあります）")

        def run() -> None:
            engine = create(name, self.config)
            ok = isinstance(engine, VoicevoxCompatible) and engine.start_engine(use_gpu=self.gpu_var.get())
            self.events.put(("status", "エンジンが起動しました。「読み込み」を押してください" if ok
                             else "エンジンを起動できませんでした（既定の場所に入っていないかもしれません）"))

        threading.Thread(target=run, daemon=True).start()

    def _speak(self) -> None:
        text = self.text.get("1.0", "end").strip()
        if not text or self.tts is None:
            return
        idx = self.voice_box.current()
        voice = self.voice_ids[idx if idx >= 0 else 0]
        speed = float(self.speed_var.get())
        sentences = split_sentences(text) if self.split_var.get() else [text]
        self.cancel.clear()
        self.player.resume()
        self.last_audio = []
        self.speak_btn.configure(state="disabled")
        self.summary.configure(text="合成中…")
        tts = self.tts

        def run() -> None:
            t_click = time.perf_counter()
            first = None
            total_audio = 0.0
            for i, sentence in enumerate(sentences, 1):
                if self.cancel.is_set():
                    break
                t0 = time.perf_counter()
                try:
                    audio, rate = tts.synthesize(sentence, voice=voice, speed=speed)
                except Exception as ex:  # noqa: BLE001
                    self.events.put(("error", f"合成に失敗しました: {ex}"))
                    break
                ms = (time.perf_counter() - t0) * 1000
                if first is None:
                    first = (time.perf_counter() - t_click) * 1000
                dur = len(audio) / rate
                total_audio += dur
                self.last_audio.append((audio, rate))
                self.player.play(audio, rate)
                self.events.put(("sentence", i, len(sentences), sentence, ms, dur))
            total = (time.perf_counter() - t_click) * 1000
            self.events.put(("done", first, total, total_audio, len(sentences)))

        threading.Thread(target=run, daemon=True).start()

    def _stop(self) -> None:
        self.cancel.set()
        self.player.stop()

    def _close_tts(self) -> None:
        close = getattr(self.tts, "close", None)  # アプリが起動したサーバーを止める
        if callable(close):
            close()

    def _close(self) -> None:
        self._stop()
        self._close_tts()
        self.root.destroy()

    def _save(self) -> None:
        if not self.last_audio:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not path:
            return
        import soundfile as sf

        rate = self.last_audio[0][1]
        sf.write(path, np.concatenate([a for a, _r in self.last_audio]), rate, subtype="PCM_16")
        self.status.configure(text=f"保存しました: {path}")

    def _poll(self) -> None:
        try:
            while True:
                kind, *data = self.events.get_nowait()
                if kind == "status":
                    self.status.configure(text=data[0])
                elif kind == "loaded":
                    self._close_tts()
                    self.tts, voices, secs = data
                    self.voice_ids = [v for v, _l in voices]
                    self.voice_box.configure(values=[label for _v, label in voices])
                    self.voice_box.current(0)
                    self.status.configure(text=f"読み込み完了（{secs:.1f} 秒）・声 {len(voices)} 種類")
                    self.load_btn.configure(state="normal")
                    self.speak_btn.configure(state="normal")
                elif kind == "error":
                    self.status.configure(text=data[0])
                    self.load_btn.configure(state="normal")
                    if self.tts is not None:
                        self.speak_btn.configure(state="normal")
                elif kind == "sentence":
                    i, n, sentence, ms, dur = data
                    self.log.insert("end", f"[{i}/{n}] {sentence}\n")
                    self.log.insert("end", f"        合成 {ms:.0f} ms・音声 {dur:.1f} 秒・RTF {ms / 1000 / max(dur, 1e-6):.3f}\n", "meta")
                    self.log.see("end")
                elif kind == "done":
                    first, total, audio_s, n = data
                    if first is not None:
                        rtf = total / 1000 / max(audio_s, 1e-6)
                        self.summary.configure(
                            text=f"最初の音まで {first:.0f} ms ・ 全体の合成 {total:.0f} ms ・ 音声 {audio_s:.1f} 秒"
                                 f" ・ RTF {rtf:.3f}（{n} 文）")
                        self.log.insert("end", "─" * 40 + "\n", "meta")
                    self.speak_btn.configure(state="normal")
                    self.save_btn.configure(state="normal" if self.last_audio else "disabled")
        except queue.Empty:
            pass
        self.root.after(50, self._poll)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
