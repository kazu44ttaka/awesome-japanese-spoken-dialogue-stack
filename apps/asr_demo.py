"""マイクの声をその場で認識して、モデルごとの精度と速さを試すアプリ（PySide6）。

    uv pip install "pyside6-essentials==6.12.0" sounddevice "silero-vad==6.2.1"   # ほかに使うモデルの分も
    python apps/asr_demo.py

ブラウザで使うときは apps/asr_web.py。中身（区切り方と疑似ストリーミング）は apps/asr_core.py にあり、
MaAI-SDS の ASR と同じしくみ・同じ既定値。

項目の名前は MaAI-SDS の画面に合わせている（ASRエンジン・Device・共通設定の VAD ON threshold など）。

1. ASRエンジンと Device を選んで「起動」（状態が「稼働」になったら使える）
2. マイクデバイスを選んで「聞き取り開始」。話し終わると確定した文が下に出る
"""

from __future__ import annotations

import sys



from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_core import (  # noqa: E402
    DEFAULT_MODEL, STATE_COLORS, STATE_LABELS, AsrSession, ListenOptions, list_audio_devices, model_choices,
)

from PySide6.QtCore import QObject, Signal  # noqa: E402
from qt_util import fill_grouped  # noqa: E402
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

DEVICES = ["cuda", "cpu"]  # MaAI-SDS の ASR の Device と同じ


class Bridge(QObject):
    """別スレッドからの通知を GUI のスレッドへ渡す。"""

    event = Signal(tuple)


def state_html(state: str) -> str:
    """MaAI-SDS のモジュール状態パネルと同じ「ランプ・名前・状態」。"""
    return f"<span style='color:{STATE_COLORS[state]}'>●</span>&nbsp;ASR&nbsp;&nbsp;{STATE_LABELS[state]}"


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

        # ---- ASR エンジン
        model_box = QGroupBox("ASR制御")
        grid = QGridLayout(model_box)
        grid.addWidget(QLabel("ASRエンジン"), 0, 0)
        self.model_combo = QComboBox()
        fill_grouped(self.model_combo, self.models, DEFAULT_MODEL)  # 実行系ごとに見出しを挟む
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        grid.addWidget(self.model_combo, 0, 1, 1, 3)
        grid.addWidget(QLabel("Device"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(DEVICES)
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
        self.state_detail = QLabel("ASRエンジンと Device を選んで「起動」を押してください")
        self.state_detail.setWordWrap(True)
        self.state_detail.setStyleSheet("color:#777")
        grid.addWidget(self.state_detail, 2, 2, 1, 2)
        grid.setColumnStretch(3, 1)
        root.addWidget(model_box)

        # ---- 共通設定（MaAI-SDS の ASR の「共通設定」と同じ項目・並び・既定値）
        opt_box = QGroupBox("共通設定")
        opt = QHBoxLayout(opt_box)
        left, right = QFormLayout(), QFormLayout()
        opt.addLayout(left)
        opt.addSpacing(24)
        opt.addLayout(right)
        opt.addStretch(1)
        self.pseudo_check = QCheckBox("有効")
        self.pseudo_check.setToolTip("疑似ストリーミング: 話している途中も、区間の頭から認識し直して途中経過を出す")
        self.fields: dict[str, QDoubleSpinBox] = {}

        def spin(key: str, lo: float, hi: float, step: float, form: QFormLayout, label: str, tip: str) -> None:
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(2)
            box.setToolTip(tip)
            form.addRow(label, box)
            self.fields[key] = box

        spin("vad_on_threshold", 0.05, 0.99, 0.05, left, "VAD ON threshold", "Silero VAD の確率がこれを超えたら話している")
        spin("vad_on_sec", 0.03, 2.0, 0.05, left, "VAD ON sec", "ON threshold を超える状態がこの秒数続いたら発話の始まり")
        spin("vad_off_threshold", 0.01, 0.95, 0.05, left, "VAD OFF threshold", "Silero VAD の確率がこれを下回ったら黙っている")
        spin("vad_off_sec", 0.03, 3.0, 0.05, left, "VAD OFF sec", "OFF threshold を下回る状態がこの秒数続いたら発話の終わり")
        spin("min_segment_sec", 0.0, 3.0, 0.05, right, "Min segment sec", "これより短い区間は認識せず、直前の途中経過で確定する")
        spin("pre_roll_buffer_sec", 0.0, 2.0, 0.05, right, "Pre-roll buffer sec", "発話の始まりと判定する前の音も、この秒数だけ区間に入れる")
        right.addRow("Pseudo streaming", self.pseudo_check)
        spin("pseudo_chunk_sec", 0.05, 3.0, 0.05, right, "Pseudo chunk sec", "疑似ストリーミングで、音声のこの秒数ごとに区間の頭から認識し直す")
        reset_btn = QPushButton("既定に戻す")
        reset_btn.setToolTip("MaAI-SDS の controller_config.json と同じ値に戻す")
        reset_btn.clicked.connect(lambda: self._set_options(ListenOptions()))
        right.addRow(reset_btn)
        self._set_options(ListenOptions())
        root.addWidget(opt_box)

        # ---- 聞き取り
        listen_box = QGroupBox("聞き取り")
        lv = QVBoxLayout(listen_box)
        row = QHBoxLayout()
        lv.addLayout(row)
        row.addWidget(QLabel("マイクデバイス"))
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
        # 認識結果（MaAI-SDS の ASR の結果欄と同じ書き方。途中経過は最後の行を上書きし、確定したら置き換えて改行）
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Yu Gothic UI", 11))
        lv.addWidget(self.log, 1)
        self._partial_open = False  # 最後の行が途中経過か
        clear_btn.clicked.connect(self._clear)
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
        return next(m for m in self.models if m["name"] == self.model_combo.currentData())

    def _sync_device(self) -> None:
        m = self._model()
        fixed = None if m["gpu"] and m["cpu"] else ("cpu" if not m["gpu"] else "cuda")
        if fixed:
            self.device_combo.setCurrentText(fixed)
        self.device_combo.setEnabled(fixed is None)
        tip = {None: "", "cpu": "このモデルは CPU で動かす", "cuda": "このモデルは GPU でしか動かない"}[fixed]
        if fixed is None and m["name"].startswith("speechcpp-"):
            tip = "cuda を選ぶと Vulkan で GPU を使う"
        self.device_combo.setToolTip(tip)

    def _on_model_changed(self, _index: int) -> None:
        """MaAI-SDS と同じく、ASRエンジンを切り替えたら前のエンジンを止める（新しいものは「起動」で読み込む）。"""
        self._sync_device()
        if self.session.state in ("running", "error"):
            self._stop()
            self.session.stop_model()

    def _sync_buttons(self) -> None:
        s = self.session
        self.model_combo.setEnabled(s.state != "starting")  # 読み込みの途中では切り替えない
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

    def _put_line(self, line: str, final: bool) -> None:
        """途中経過は最後の行を上書きする。確定は途中経過の行を置き換え、次は新しい行から書く。"""
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self._partial_open:
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
        elif not self.log.document().isEmpty():
            cursor.insertBlock()
        fmt = QTextCharFormat()
        if not final:
            fmt.setForeground(QColor("#888888"))  # 途中経過は灰色
        cursor.insertText(line, fmt)
        self._partial_open = not final
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _clear(self) -> None:
        self.log.clear()
        self._partial_open = False


    # ---------------------------------------------------------------- actions
    def _start_model(self) -> None:
        self._stop()
        self.session.start_model(self._model()["name"], self.device_combo.currentText(), threads=8)

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
        if self._partial_open:  # 確定しないまま止めたときは、途中経過の行を残して次は改行する
            self._partial_open = False
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
                self.state_detail.setText("ASRエンジンと Device を選んで「起動」を押してください")
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
            text, _ms, latency = data
            if text:
                self._put_line(f"{text} (latency={latency / 1000:.3f}s)", final=False)
        elif kind == "final":
            text, _ms, _audio_s, latency = data
            if text:  # MaAI-SDS と同じく、空の確定は書かない
                self._put_line(f"{text} (latency={latency / 1000:.3f}s)", final=True)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
