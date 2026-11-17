"""日本語 TTS を選んで文を読み上げ、最初の音が出るまでの時間と速さを試すアプリ（PySide6）。

    python apps/tts_demo.py

ブラウザで使うときは apps/tts_web.py。中身（エンジンの起動と文ごとの合成）は apps/tts_core.py にある。
エンジンの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピー）。必要なパッケージと入れ方は apps/README.md を参照。

項目の名前は MaAI-SDS の画面に合わせている（TTS制御・TTSエンジン・VOICEVOX Core Device・Voice/Speaker・
TTSテストなど）。

1. TTSエンジンと Device を選んで「起動」（状態が「稼働」になったら使える。AivisSpeech / VOICEVOX は
   動いていなければ立ち上げる）
2. Voice/Speaker を選び、テスト発話を入れて「TTSテスト」
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts_core import (  # noqa: E402
    DEFAULT_ENGINE, SAMPLES, STATE_COLORS, STATE_LABELS, TtsSession, engine_choices,
)

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QFont, QTextCursor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)


def list_output_devices() -> list[tuple[int | None, str]]:
    out: list[tuple[int | None, str]] = [(None, "default")]
    try:
        import sounddevice as sd

        api = sd.default.hostapi
        out += [(i, d["name"]) for i, d in enumerate(sd.query_devices())
                if d["max_output_channels"] > 0 and d["hostapi"] == api]
    except Exception:  # noqa: BLE001
        pass
    return out


class Player:
    """合成できた音声を、開いたままの 1 本の出力ストリームで切れ目なく順に再生する。

    - 再生のたびにストリームを開くと、デバイスが動き出すまで（Bluetooth などでは数百 ms）に
      1 文目の頭が消えるので、アプリの起動時から 48 kHz で開いたままにし、黙っている間は無音を流す
    - エンジンごとにサンプリング周波数が違うので、48 kHz に変換してから流す
    - 合成で CPU が忙しくても音が切れないよう、余裕のあるバッファ（latency="high"）で流す
    """

    RATE = 48000

    def __init__(self) -> None:
        self.device: int | None = None
        self.stream = None
        self.buf: deque = deque()
        self.offset = 0
        self.lock = threading.Lock()
        self.error = ""
        self._open()

    def _open(self) -> None:
        import sounddevice as sd

        self._close()
        try:
            self.stream = sd.OutputStream(samplerate=self.RATE, channels=1, dtype="float32", device=self.device,
                                          latency="high", callback=self._callback)
            self.stream.start()
            self.error = ""
        except Exception as ex:  # noqa: BLE001
            self.stream, self.error = None, f"スピーカーを開けませんでした: {ex}"

    def set_device(self, device: int | None) -> None:
        if device != self.device:
            self.device = device
            self.stop()
            self._open()

    def _close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
        self.stream = None

    def _callback(self, outdata, frames, _time, _status) -> None:
        out = outdata[:, 0]
        filled = 0
        with self.lock:
            while filled < frames and self.buf:
                head = self.buf[0]
                n = min(frames - filled, len(head) - self.offset)
                out[filled:filled + n] = head[self.offset:self.offset + n]
                filled += n
                self.offset += n
                if self.offset >= len(head):
                    self.buf.popleft()
                    self.offset = 0
        out[filled:] = 0.0

    @classmethod
    def _resample(cls, audio: np.ndarray, rate: int) -> np.ndarray:
        if rate == cls.RATE or len(audio) == 0:
            return audio
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(rate), cls.RATE)
        return resample_poly(audio, cls.RATE // g, int(rate) // g).astype(np.float32)

    def play(self, audio: np.ndarray, rate: int) -> None:
        if self.stream is None:
            self._open()
        audio = self._resample(np.asarray(audio, dtype=np.float32).reshape(-1), int(rate))
        with self.lock:
            self.buf.append(audio)

    def stop(self) -> None:
        with self.lock:
            self.buf.clear()
            self.offset = 0

    def close(self) -> None:
        self.stop()
        self._close()


class Bridge(QObject):
    """別スレッドからの通知を GUI のスレッドへ渡す。"""

    event = Signal(tuple)


def state_html(state: str) -> str:
    """MaAI-SDS のモジュール状態パネルと同じ「ランプ・名前・状態」。"""
    return f"<span style='color:{STATE_COLORS[state]}'>●</span>&nbsp;TTS&nbsp;&nbsp;{STATE_LABELS[state]}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("日本語 TTS デモ")
        self.resize(1000, 720)
        self.player = Player()
        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.session = TtsSession(self._emit)
        self.engines = engine_choices()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---- TTS制御
        engine_box = QGroupBox("TTS制御")
        grid = QGridLayout(engine_box)
        grid.addWidget(QLabel("TTSエンジン"), 0, 0)
        self.engine_combo = QComboBox()
        for e in self.engines:
            self.engine_combo.addItem(e["label"], e["name"])
        self.engine_combo.setCurrentIndex(next(i for i, e in enumerate(self.engines) if e["name"] == DEFAULT_ENGINE))
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        grid.addWidget(self.engine_combo, 0, 1, 1, 3)
        self.device_label = QLabel("Device")
        grid.addWidget(self.device_label, 1, 0)
        self.device_combo = QComboBox()
        grid.addWidget(self.device_combo, 1, 1)
        self.start_btn = QPushButton("起動")
        self.start_btn.setToolTip("エンジンを使える状態にする。AivisSpeech / VOICEVOX は動いていなければ立ち上げる")
        self.start_btn.clicked.connect(self._start_engine)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setToolTip("エンジンを止める（このアプリが立ち上げたエンジンだけ。前から動いていたものはそのまま）")
        self.stop_btn.clicked.connect(self._stop_engine)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch(1)
        grid.addLayout(buttons, 1, 2, 1, 2)
        grid.addWidget(QLabel("状態"), 2, 0)
        self.state_label = QLabel(state_html("stopped"))
        grid.addWidget(self.state_label, 2, 1)
        self.state_detail = QLabel("")
        self.state_detail.setWordWrap(True)
        self.state_detail.setStyleSheet("color:#777")
        grid.addWidget(self.state_detail, 2, 2, 1, 2)
        grid.setColumnStretch(3, 1)
        root.addWidget(engine_box)

        # ---- TTSテスト
        speak_box = QGroupBox("TTSテスト")
        sv = QVBoxLayout(speak_box)
        g2 = QGridLayout()
        sv.addLayout(g2)
        g2.addWidget(QLabel("Voice/Speaker"), 0, 0)
        self.voice_combo = QComboBox()
        g2.addWidget(self.voice_combo, 0, 1)
        g2.addWidget(QLabel("話速（1.00が標準）"), 0, 2)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        self.speed.setDecimals(2)
        g2.addWidget(self.speed, 0, 3)
        g2.addWidget(QLabel("スピーカーデバイス"), 1, 0)
        self.out_combo = QComboBox()
        for idx, label in list_output_devices():
            self.out_combo.addItem(label, idx)
        self.out_combo.currentIndexChanged.connect(self._set_output)
        g2.addWidget(self.out_combo, 1, 1)
        g2.addWidget(QLabel("例文"), 1, 2)
        sample_combo = QComboBox()
        sample_combo.addItems(list(SAMPLES))
        sample_combo.textActivated.connect(lambda key: self.text.setPlainText(SAMPLES[key]))
        g2.addWidget(sample_combo, 1, 3)
        g2.setColumnStretch(1, 1)
        self.split_check = QCheckBox("文ごとに合成して、できた文から再生（音声対話向け）")
        self.split_check.setChecked(True)
        sv.addWidget(self.split_check)
        sv.addWidget(QLabel("テスト発話"))
        self.text = QPlainTextEdit(SAMPLES["あいさつ"])
        self.text.setFont(QFont("Yu Gothic UI", 11))
        self.text.setFixedHeight(100)
        sv.addWidget(self.text)
        ctrl = QHBoxLayout()
        sv.addLayout(ctrl)
        self.speak_btn = QPushButton("TTSテスト")
        self.speak_btn.setToolTip("テスト発話を合成して再生する")
        self.speak_btn.clicked.connect(self._speak)
        self.halt_btn = QPushButton("再生を止める")
        self.halt_btn.clicked.connect(self._halt)
        self.save_btn = QPushButton("WAV で保存")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        ctrl.addWidget(self.speak_btn)
        ctrl.addWidget(self.halt_btn)
        ctrl.addWidget(self.save_btn)
        self.status = QLabel(self.player.error)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#777")
        ctrl.addWidget(self.status, 1)
        self.summary = QLabel("")
        self.summary.setStyleSheet("font-weight: bold; padding: 4px 2px;")
        sv.addWidget(self.summary)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Yu Gothic UI", 10))
        sv.addWidget(self.log, 1)
        root.addWidget(speak_box, 1)

        self._sync_engine()

    # ---------------------------------------------------------------- helpers
    def _emit(self, *e) -> None:
        if e[0] == "audio":  # 再生はすぐに（GUI のスレッドを待たない）
            _kind, _i, audio, rate = e
            self.player.play(audio, rate)
            return
        self.bridge.event.emit(e)

    def _set_output(self, _index: int) -> None:
        self.player.set_device(self.out_combo.currentData())
        self.status.setText(self.player.error)

    def _engine(self) -> dict:
        return self.engines[self.engine_combo.currentIndex()]

    def _sync_engine(self) -> None:
        e = self._engine()
        self.device_combo.clear()
        self.device_label.setText(e["device_label"])
        if e["devices"]:
            self.device_combo.addItems(e["devices"])
            self.device_combo.setCurrentText(e["default_device"])
            self.device_combo.setEnabled(True)
            self.device_combo.setToolTip("AivisSpeech / VOICEVOX は、このアプリが立ち上げるときだけ効く" if e["http"] else "")
        else:
            self.device_combo.addItem(e["fixed_device"])
            self.device_combo.setEnabled(False)
            self.device_combo.setToolTip("このエンジンはデバイスを選べない")
        if self.session.state in ("stopped", "error"):
            self.state_detail.setText(e["note"])
        self._sync_buttons()

    def _on_engine_changed(self, _index: int) -> None:
        """MaAI-SDS と同じく、TTSエンジンを切り替えたら前のエンジンを止める（新しいものは「起動」で起動する）。"""
        if self.session.state in ("running", "error"):
            self._halt()
            self.voice_combo.clear()
            self.session.stop_engine()
        self._sync_engine()

    def _sync_buttons(self) -> None:
        s = self.session
        self.engine_combo.setEnabled(s.state != "starting")  # 起動の途中では切り替えない
        self.start_btn.setEnabled(s.state != "starting")
        self.stop_btn.setEnabled(s.state in ("running", "error"))
        self.speak_btn.setEnabled(s.ready and not s.speaking)

    def _append(self, html: str) -> None:
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertHtml(html + "<br>")
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    # ---------------------------------------------------------------- actions
    def _start_engine(self) -> None:
        self._halt()
        e = self._engine()
        self.session.start(e["name"], self.device_combo.currentText() if e["devices"] else None)

    def _stop_engine(self) -> None:
        self._halt()
        self.session.stop_engine()

    def _speak(self) -> None:
        self.player.stop()
        self.speak_btn.setEnabled(False)
        self.summary.setText("合成中…")
        self.status.setText("")
        self.session.speak(self.text.toPlainText(), voice=self.voice_combo.currentData(), speed=self.speed.value(),
                           split=self.split_check.isChecked())

    def _halt(self) -> None:
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
        self._halt()
        self.session.close()
        self.player.close()
        super().closeEvent(event)

    def _on_event(self, e: tuple) -> None:
        kind, *data = e
        if kind == "state":
            state, detail = data
            self.state_label.setText(state_html(state))
            self.state_detail.setText(detail or self._engine()["note"])
        elif kind == "voices":
            self.voice_combo.clear()
            for vid, label in data[0]:
                self.voice_combo.addItem(label, vid)
        elif kind in ("status", "error"):
            self.status.setText(data[0])
        elif kind == "sentence":
            i, n, sentence, ms, dur = data
            self._append(f"[{i}/{n}] {escape(sentence)}")
            self._append(f"<span style='color:#888'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;合成 {ms:.0f} ms・"
                         f"音声 {dur:.1f} 秒・RTF {ms / 1000 / max(dur, 1e-6):.3f}</span>")
        elif kind == "done":
            first, total, audio_s, n = data
            if first is not None:
                rtf = total / 1000 / max(audio_s, 1e-6)
                self.summary.setText(f"再生開始まで {first:.0f} ms ・ 全体の合成 {total:.0f} ms ・ 音声 {audio_s:.1f} 秒"
                                     f" ・ RTF {rtf:.3f}（{n} 文）")
                self._append("<span style='color:#888'>" + "─" * 40 + "</span>")
            else:
                self.summary.setText("")
            self.save_btn.setEnabled(self.session.joined_audio() is not None)
        self._sync_buttons()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
