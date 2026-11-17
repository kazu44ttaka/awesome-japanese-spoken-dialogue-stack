"""マイクの声をその場で認識して、モデルごとの精度と速さを試すアプリ（PySide6）。

    uv pip install "pyside6-essentials==6.12.0" sounddevice   # ほかに使うモデルの分（sherpa-onnx / transformers / funasr）
    python apps/asr_demo.py

ブラウザで使うときは apps/asr_web.py。中身（VAD で区切って認識する部分）は apps/asr_core.py にある。

- 「疑似ストリーミング」: 話している途中も、それまでの音声を一定間隔で認識し直して途中経過を出す
- 「VAD で区切る」: 発話区間検出（Silero VAD）で話し終わりを見つけ、区間ごとに認識して確定する
- 確定した行には、音声の長さ・認識にかかった時間・話し終わりから表示までの時間（無音で区切る秒数 + 認識）を出す
- 「真のストリーミング」のモデル（Nemotron 3.5 streaming）は、話し始めから音声を少しずつ入れ、
  内部状態を引き継ぎながら途中経過を出す
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_core import DEFAULT_MODEL, AsrSession, ListenOptions, list_mics, model_choices  # noqa: E402

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QFont, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QPushButton, QRadioButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)


class Bridge(QObject):
    """別スレッドからの通知を GUI のスレッドへ渡す。"""

    event = Signal(tuple)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("日本語 ASR デモ")
        self.resize(960, 640)
        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.session = AsrSession(lambda *e: self.bridge.event.emit(e))
        self.models = model_choices()
        self.mics = list_mics()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        grid = QGridLayout()
        root.addLayout(grid)
        grid.addWidget(QLabel("モデル"), 0, 0)
        self.model_box = QComboBox()
        for m in self.models:
            self.model_box.addItem(m["label"], m["name"])
        self.model_box.setCurrentIndex(next(i for i, m in enumerate(self.models) if m["name"] == DEFAULT_MODEL))
        self.model_box.currentIndexChanged.connect(self._sync_device)
        grid.addWidget(self.model_box, 0, 1)
        grid.addWidget(QLabel("実行"), 0, 2)
        self.device_box = QComboBox()
        self.device_box.addItems(["cpu", "cuda"])
        grid.addWidget(self.device_box, 0, 3)
        self.load_btn = QPushButton("読み込み")
        self.load_btn.clicked.connect(self._load)
        grid.addWidget(self.load_btn, 0, 4)
        grid.addWidget(QLabel("マイク"), 1, 0)
        self.mic_box = QComboBox()
        for idx, label in self.mics:
            self.mic_box.addItem(label, idx)
        grid.addWidget(self.mic_box, 1, 1)
        grid.setColumnStretch(1, 1)

        opts = QHBoxLayout()
        root.addLayout(opts)
        self.partial_radio = QRadioButton("疑似ストリーミング（途中経過も出す）")
        self.final_radio = QRadioButton("VAD で区切る（確定だけ）")
        self.partial_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.partial_radio)
        group.addButton(self.final_radio)
        opts.addWidget(self.partial_radio)
        opts.addWidget(self.final_radio)
        opts.addSpacing(12)
        opts.addWidget(QLabel("途中経過の間隔(ms)"))
        self.interval = QSpinBox()
        self.interval.setRange(100, 3000)
        self.interval.setSingleStep(100)
        self.interval.setValue(500)
        opts.addWidget(self.interval)
        opts.addWidget(QLabel("無音で区切る(秒)"))
        self.silence = QDoubleSpinBox()
        self.silence.setRange(0.1, 2.0)
        self.silence.setSingleStep(0.1)
        self.silence.setValue(0.4)
        opts.addWidget(self.silence)
        opts.addStretch(1)

        ctrl = QHBoxLayout()
        root.addLayout(ctrl)
        self.start_btn = QPushButton("開始")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        clear_btn = QPushButton("クリア")
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.stop_btn)
        ctrl.addWidget(clear_btn)
        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("入力"))
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedWidth(160)
        ctrl.addWidget(self.level)
        self.status = QLabel("モデルを選んで「読み込み」を押してください")
        ctrl.addWidget(self.status, 1)

        self.partial = QLabel("")
        self.partial.setWordWrap(True)
        self.partial.setStyleSheet("color: #777; padding: 4px 6px;")
        root.addWidget(self.partial)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Yu Gothic UI", 11))
        root.addWidget(self.log, 1)
        clear_btn.clicked.connect(self.log.clear)

        self._sync_device()

    # ---------------------------------------------------------------- helpers
    def _model(self) -> dict:
        return self.models[self.model_box.currentIndex()]

    def _sync_device(self) -> None:
        gpu = self._model()["gpu"]
        if not gpu:
            self.device_box.setCurrentText("cpu")
        self.device_box.setEnabled(gpu)
        self.start_btn.setEnabled(False)

    def _append(self, html: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html + "<br>")
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    # ---------------------------------------------------------------- actions
    def _load(self) -> None:
        self._stop()
        self.load_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.session.load(self._model()["name"], self.device_box.currentText(), threads=8)

    def _start(self) -> None:
        options = ListenOptions(partial=self.partial_radio.isChecked(), interval_ms=self.interval.value(),
                                min_silence=self.silence.value())
        try:
            self.session.start(options, mic=self.mic_box.currentData())
        except Exception as ex:  # noqa: BLE001
            self.status.setText(f"開始できませんでした: {ex}")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("聞いています…")

    def _stop(self) -> None:
        self.session.stop()
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(self.session.ready)
        self.partial.setText("")
        self.level.setValue(0)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.session.close()
        super().closeEvent(event)

    def _on_event(self, e: tuple) -> None:
        kind, *data = e
        if kind == "status":
            self.status.setText(data[0])
        elif kind == "loaded":
            self.status.setText(data[0])
            self.load_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
        elif kind == "error":
            self.status.setText(data[0])
            self.load_btn.setEnabled(True)
        elif kind == "level":
            if self.session.listening:
                self.level.setValue(int(data[0] * 100))
        elif kind == "partial":
            text, ms = data
            self.partial.setText(f"途中経過（{ms:.0f} ms）: {text}")
        elif kind == "final":
            text, ms, audio_s, latency = data
            self.partial.setText("")
            from html import escape

            stamp = time.strftime("%H:%M:%S")
            self._append(f"{stamp}&nbsp;&nbsp;{escape(text) or '（認識結果なし）'}")
            self._append(f"<span style='color:#888'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;音声 {audio_s:.1f} 秒・"
                         f"認識 {ms:.0f} ms・話し終わりから表示まで 約 {latency:.0f} ms</span>")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
