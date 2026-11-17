"""日本語 TTS を選んで文を読み上げ、最初の音が出るまでの時間と速さを試すアプリ（PySide6）。

    python apps/tts_demo.py

ブラウザで使うときは apps/tts_web.py。中身（文ごとの合成）は apps/tts_core.py にある。
エンジンの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピー）。必要なパッケージと入れ方は apps/README.md を参照。

- AivisSpeech / VOICEVOX はエンジンを起動しておく（入っていれば「エンジンを起動」で起動できる。
  「GPU を使う」で --use_gpu を付ける。すでに起動しているエンジンはそのまま使う）
- 「文ごとに合成して、できた文から再生」を入れると、1 文目ができた時点で再生を始める（音声対話と同じ）
- 文ごとに、合成にかかった時間・音声の長さ・RTF（合成時間 ÷ 音声の長さ）を出す
"""

from __future__ import annotations

import queue
import sys
import threading
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_core import DEFAULT_ENGINE, SAMPLES, TtsSession, engine_choices  # noqa: E402

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QFont, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


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


class Bridge(QObject):
    """別スレッドからの通知を GUI のスレッドへ渡す。"""

    event = Signal(tuple)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("日本語 TTS デモ")
        self.resize(960, 680)
        self.player = Player()
        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.session = TtsSession(self._emit)
        self.engines = engine_choices()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        grid = QGridLayout()
        root.addLayout(grid)
        grid.addWidget(QLabel("エンジン"), 0, 0)
        self.engine_box = QComboBox()
        for e in self.engines:
            self.engine_box.addItem(e["label"], e["name"])
        self.engine_box.setCurrentIndex(next(i for i, e in enumerate(self.engines) if e["name"] == DEFAULT_ENGINE))
        self.engine_box.currentIndexChanged.connect(self._reset)
        grid.addWidget(self.engine_box, 0, 1)
        self.load_btn = QPushButton("読み込み")
        self.load_btn.clicked.connect(self._load)
        grid.addWidget(self.load_btn, 0, 2)
        self.start_engine_btn = QPushButton("エンジンを起動")
        self.start_engine_btn.clicked.connect(self._start_engine)
        grid.addWidget(self.start_engine_btn, 0, 3)
        self.gpu_check = QCheckBox("GPU を使う")
        self.gpu_check.setChecked(True)
        grid.addWidget(self.gpu_check, 0, 4)
        grid.addWidget(QLabel("声"), 1, 0)
        self.voice_box = QComboBox()
        grid.addWidget(self.voice_box, 1, 1)
        grid.addWidget(QLabel("速さ"), 1, 2)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        grid.addWidget(self.speed, 1, 3)
        grid.setColumnStretch(1, 1)

        mid = QHBoxLayout()
        root.addLayout(mid)
        mid.addWidget(QLabel("例文"))
        sample_box = QComboBox()
        sample_box.addItems(list(SAMPLES))
        sample_box.textActivated.connect(lambda key: self.text.setPlainText(SAMPLES[key]))
        mid.addWidget(sample_box)
        self.split_check = QCheckBox("文ごとに合成して、できた文から再生（音声対話向け）")
        self.split_check.setChecked(True)
        mid.addWidget(self.split_check)
        mid.addStretch(1)

        self.text = QPlainTextEdit(SAMPLES["あいさつ"])
        self.text.setFont(QFont("Yu Gothic UI", 11))
        self.text.setFixedHeight(110)
        root.addWidget(self.text)

        ctrl = QHBoxLayout()
        root.addLayout(ctrl)
        self.speak_btn = QPushButton("合成して再生")
        self.speak_btn.setEnabled(False)
        self.speak_btn.clicked.connect(self._speak)
        stop_btn = QPushButton("停止")
        stop_btn.clicked.connect(self._stop)
        self.save_btn = QPushButton("WAV で保存")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        ctrl.addWidget(self.speak_btn)
        ctrl.addWidget(stop_btn)
        ctrl.addWidget(self.save_btn)
        self.status = QLabel("エンジンを選んで「読み込み」を押してください")
        self.status.setWordWrap(True)
        ctrl.addWidget(self.status, 1)

        self.summary = QLabel("")
        self.summary.setStyleSheet("font-weight: bold; padding: 6px;")
        root.addWidget(self.summary)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Yu Gothic UI", 10))
        root.addWidget(self.log, 1)

        self._reset()

    # ---------------------------------------------------------------- helpers
    def _emit(self, *e) -> None:
        if e[0] == "audio":  # 再生はすぐに（GUI のスレッドを待たない）
            _kind, _i, audio, rate = e
            self.player.play(audio, rate)
            return
        self.bridge.event.emit(e)

    def _engine(self) -> dict:
        return self.engines[self.engine_box.currentIndex()]

    def _append(self, html: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html + "<br>")
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _reset(self) -> None:
        self.session.close()
        self.speak_btn.setEnabled(False)
        self.voice_box.clear()
        engine = self._engine()
        self.start_engine_btn.setEnabled(engine["http"])
        self.gpu_check.setEnabled(engine["http"])
        self.status.setText(engine["note"])

    # ---------------------------------------------------------------- actions
    def _load(self) -> None:
        self.load_btn.setEnabled(False)
        self.session.load(self._engine()["name"])

    def _start_engine(self) -> None:
        self.session.start_engine(self._engine()["name"], use_gpu=self.gpu_check.isChecked())

    def _speak(self) -> None:
        self.player.resume()
        self.speak_btn.setEnabled(False)
        self.summary.setText("合成中…")
        self.session.speak(self.text.toPlainText(), voice=self.voice_box.currentData(), speed=self.speed.value(),
                           split=self.split_check.isChecked())

    def _stop(self) -> None:
        self.session.stop()
        self.player.stop()

    def _save(self) -> None:
        joined = self.session.joined_audio()
        if joined is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "WAV で保存", "tts.wav", "WAV (*.wav)")
        if not path:
            return
        import soundfile as sf

        audio, rate = joined
        sf.write(path, audio, rate, subtype="PCM_16")
        self.status.setText(f"保存しました: {path}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop()
        self.session.close()
        super().closeEvent(event)

    def _on_event(self, e: tuple) -> None:
        kind, *data = e
        if kind == "status":
            self.status.setText(data[0])
        elif kind == "loaded":
            voices, secs = data
            self.voice_box.clear()
            for vid, label in voices:
                self.voice_box.addItem(label, vid)
            self.status.setText(f"読み込み完了（{secs:.1f} 秒）・声 {len(voices)} 種類")
            self.load_btn.setEnabled(True)
            self.speak_btn.setEnabled(True)
        elif kind == "error":
            self.status.setText(data[0])
            self.load_btn.setEnabled(True)
            self.speak_btn.setEnabled(self.session.ready)
        elif kind == "sentence":
            i, n, sentence, ms, dur = data
            self._append(f"[{i}/{n}] {escape(sentence)}")
            self._append(f"<span style='color:#888'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;合成 {ms:.0f} ms・"
                         f"音声 {dur:.1f} 秒・RTF {ms / 1000 / max(dur, 1e-6):.3f}</span>")
        elif kind == "done":
            first, total, audio_s, n = data
            if first is not None:
                rtf = total / 1000 / max(audio_s, 1e-6)
                self.summary.setText(f"最初の音まで {first:.0f} ms ・ 全体の合成 {total:.0f} ms ・ 音声 {audio_s:.1f} 秒"
                                     f" ・ RTF {rtf:.3f}（{n} 文）")
                self._append("<span style='color:#888'>" + "─" * 40 + "</span>")
            self.speak_btn.setEnabled(self.session.ready)
            self.save_btn.setEnabled(self.session.joined_audio() is not None)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
