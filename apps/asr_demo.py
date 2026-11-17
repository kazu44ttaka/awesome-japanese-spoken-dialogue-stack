"""マイクの声をその場で認識して、モデルごとの精度と速さを試すアプリ（PySide6）。

    uv pip install "pyside6-essentials==6.12.0" sounddevice "silero-vad==6.2.1"   # ほかに使うモデルの分も
    python apps/asr_demo.py

ブラウザで使うときは apps/asr_web.py。中身（区切り方と疑似ストリーミング）は apps/asr_core.py にあり、
MaAI-SDS の ASR と同じしくみ・同じ既定値。

1. モデルとデバイスを選んで「起動」（状態が「稼働」になったら使える）
2. マイクを選んで「聞き取り開始」。話し終わると確定した文が下に出る
"""

from __future__ import annotations

import sys
import time

from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_core import (  # noqa: E402
    DEFAULT_MODEL, STATE_COLORS, STATE_LABELS, AsrSession, ListenOptions, list_audio_devices, model_choices,
)

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QFont, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

DEVICES = [("GPU (CUDA)", "cuda"), ("CPU", "cpu")]


class Bridge(QObject):
    """別スレッドからの通知を GUI のスレッドへ渡す。"""

    event = Signal(tuple)


def state_html(state: str) -> str:
    return f"<span style='color:{STATE_COLORS[state]}'>●</span> {STATE_LABELS[state]}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("日本語 ASR デモ")
        self.resize(1000, 720)
        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.session = AsrSession(lambda *e: self.bridge.event.emit(e))
        self.models = model_choices()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- モデル
        model_box = QGroupBox("モデル")
        grid = QGridLayout(model_box)
        grid.addWidget(QLabel("モデル"), 0, 0)
        self.model_combo = QComboBox()
        for m in self.models:
            self.model_combo.addItem(m["label"], m["name"])
        self.model_combo.setCurrentIndex(next(i for i, m in enumerate(self.models) if m["name"] == DEFAULT_MODEL))
        self.model_combo.currentIndexChanged.connect(self._sync_device)
        grid.addWidget(self.model_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("デバイス"), 1, 0)
        self.device_combo = QComboBox()
        for label, value in DEVICES:
            self.device_combo.addItem(label, value)
        grid.addWidget(self.device_combo, 1, 1)
        self.start_model_btn = QPushButton("起動")
        self.start_model_btn.setToolTip("モデルを読み込んで、すぐ認識できる状態にする（初回はダウンロードする）")
        self.start_model_btn.clicked.connect(self._start_model)
        self.stop_model_btn = QPushButton("停止")
        self.stop_model_btn.setToolTip("モデルを閉じてメモリを空ける")
        self.stop_model_btn.clicked.connect(self.session.stop_model)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_model_btn)
        buttons.addWidget(self.stop_model_btn)
        buttons.addStretch(1)
        grid.addLayout(buttons, 1, 2, 1, 2)
        grid.addWidget(QLabel("状態"), 2, 0)
        self.state_label = QLabel(state_html("stopped"))
        grid.addWidget(self.state_label, 2, 1)
        self.state_detail = QLabel("モデルとデバイスを選んで「起動」を押してください")
        self.state_detail.setWordWrap(True)
        self.state_detail.setStyleSheet("color:#777")
        grid.addWidget(self.state_detail, 2, 2, 1, 2)
        grid.setColumnStretch(3, 1)
        root.addWidget(model_box)

        # ---- 区切り方・疑似ストリーミング
        opt_box = QGroupBox("区切り方・疑似ストリーミング（既定値は MaAI-SDS と同じ）")
        opt = QHBoxLayout(opt_box)
        left, right = QFormLayout(), QFormLayout()
        opt.addLayout(left)
        opt.addSpacing(24)
        opt.addLayout(right)
        opt.addStretch(1)
        self.pseudo_check = QCheckBox("疑似ストリーミング（話している途中も認識し直して途中経過を出す）")
        left.addRow(self.pseudo_check)
        self.fields: dict[str, QDoubleSpinBox] = {}

        def spin(key: str, lo: float, hi: float, step: float, form: QFormLayout, label: str, tip: str) -> None:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(2)
            box.setToolTip(tip)
            form.addRow(label, box)
            self.fields[key] = box

        spin("pseudo_chunk_sec", 0.05, 3.0, 0.05, left, "途中経過の間隔（秒）", "音声のこの秒数ごとに、区間の頭から認識し直す")
        spin("pre_roll_sec", 0.0, 2.0, 0.05, left, "プリロール（秒）", "発話の始まりと判定する前の音も、この秒数だけ区間に入れる")
        spin("min_segment_sec", 0.0, 3.0, 0.05, left, "最短の区間（秒）", "これより短い区間は認識せず、直前の途中経過で確定する")
        spin("vad_on_threshold", 0.05, 0.99, 0.05, right, "開始のしきい値", "Silero VAD の確率がこれを超えたら話している")
        spin("vad_on_sec", 0.03, 2.0, 0.05, right, "開始の時間（秒）", "しきい値を超える状態がこの秒数続いたら発話の始まり")
        spin("vad_off_threshold", 0.01, 0.95, 0.05, right, "終了のしきい値", "Silero VAD の確率がこれを下回ったら黙っている")
        spin("vad_off_sec", 0.03, 3.0, 0.05, right, "終了の時間（秒）", "しきい値を下回る状態がこの秒数続いたら発話の終わり")
        reset_btn = QPushButton("既定に戻す")
        reset_btn.clicked.connect(lambda: self._set_options(ListenOptions()))
        right.addRow(reset_btn)
        self._set_options(ListenOptions())
        root.addWidget(opt_box)

        # ---- 聞き取り
        listen_box = QGroupBox("聞き取り")
        lv = QVBoxLayout(listen_box)
        row = QHBoxLayout()
        lv.addLayout(row)
        row.addWidget(QLabel("マイク"))
        self.mic_combo = QComboBox()
        for idx, label in list_audio_devices("input"):
            self.mic_combo.addItem(label, idx)
        row.addWidget(self.mic_combo, 1)
        self.listen_btn = QPushButton("聞き取り開始")
        self.listen_btn.clicked.connect(self._start)
        self.unlisten_btn = QPushButton("聞き取り停止")
        self.unlisten_btn.clicked.connect(self._stop)
        clear_btn = QPushButton("クリア")
        row.addWidget(self.listen_btn)
        row.addWidget(self.unlisten_btn)
        row.addWidget(clear_btn)
        meters = QHBoxLayout()
        lv.addLayout(meters)
        meters.addWidget(QLabel("入力"))
        self.level = self._meter()
        meters.addWidget(self.level)
        meters.addWidget(QLabel("VAD"))
        self.vad = self._meter()
        meters.addWidget(self.vad)
        self.listen_status = QLabel("")
        self.listen_status.setStyleSheet("color:#777")
        meters.addWidget(self.listen_status, 1)
        self.partial = QLabel("")
        self.partial.setWordWrap(True)
        self.partial.setStyleSheet("color: #777; padding: 4px 2px;")
        lv.addWidget(self.partial)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Yu Gothic UI", 11))
        lv.addWidget(self.log, 1)
        clear_btn.clicked.connect(self.log.clear)
        root.addWidget(listen_box, 1)

        self._sync_device()
        self._sync_buttons()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _meter() -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedWidth(140)
        bar.setFixedHeight(12)
        return bar

    def _model(self) -> dict:
        return self.models[self.model_combo.currentIndex()]

    def _sync_device(self) -> None:
        gpu = self._model()["gpu"]
        if not gpu:
            self.device_combo.setCurrentIndex(1)
        self.device_combo.setEnabled(gpu)
        self.device_combo.setToolTip("" if gpu else "このモデルは CPU で動かす")

    def _sync_buttons(self) -> None:
        s = self.session
        self.start_model_btn.setEnabled(s.state != "starting")
        self.stop_model_btn.setEnabled(s.state in ("running", "error"))
        self.listen_btn.setEnabled(s.ready and not s.listening)
        self.unlisten_btn.setEnabled(s.listening)

    def _set_options(self, o: ListenOptions) -> None:
        self.pseudo_check.setChecked(o.pseudo)
        for k, box in self.fields.items():
            box.setValue(getattr(o, k))

    def _options(self) -> ListenOptions:
        return ListenOptions.from_dict({"pseudo": self.pseudo_check.isChecked(),
                                        **{k: b.value() for k, b in self.fields.items()}})

    def _append(self, html: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html + "<br>")
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    # ---------------------------------------------------------------- actions
    def _start_model(self) -> None:
        self._stop()
        self.session.start_model(self._model()["name"], self.device_combo.currentData(), threads=8)

    def _start(self) -> None:
        o = self._options()
        try:
            self.session.start(o, mic=self.mic_combo.currentData())
        except Exception as ex:  # noqa: BLE001
            self.listen_status.setText(f"開始できませんでした: {ex}")
            return
        mode = "真のストリーミング" if self._model_is_streaming() else ("疑似ストリーミング" if o.pseudo else "確定だけ")
        self.listen_status.setText(f"聞いています…（{mode}）")
        self._sync_buttons()

    def _model_is_streaming(self) -> bool:
        return self.session.streamer is not None

    def _stop(self) -> None:
        self.session.stop()
        self.partial.setText("")
        self.level.setValue(0)
        self.vad.setValue(0)
        self.listen_status.setText("")
        self._sync_buttons()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.session.close()
        super().closeEvent(event)

    def _on_event(self, e: tuple) -> None:
        kind, *data = e
        if kind == "state":
            state, detail = data
            self.state_label.setText(state_html(state))
            if detail:
                self.state_detail.setText(detail)
            elif state == "stopped":
                self.state_detail.setText("モデルとデバイスを選んで「起動」を押してください")
            if state != "running":
                self._stop()
            self._sync_buttons()
        elif kind in ("status", "error"):
            self.listen_status.setText(data[0])
        elif kind == "level":
            if self.session.listening:
                self.level.setValue(int(data[0] * 100))
        elif kind == "vad":
            if self.session.listening:
                self.vad.setValue(int(data[0] * 100))
        elif kind == "partial":
            text, ms = data
            self.partial.setText(f"途中経過（{ms:.0f} ms）: {text}")
        elif kind == "final":
            text, ms, audio_s, latency = data
            self.partial.setText("")
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
