"""TTS デモの中身（PySide6 版 tts_demo.py と Web 版 tts_web.py が共通で使う）。

エンジンの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピー）。

- 「文ごとに合成」を入れると、文に分けて 1 文ずつ合成し、できた文から渡す（音声対話と同じ）。
  最初の音が出るまでの時間（TTFA）が、全文をまとめて合成するときと比べてどれだけ短くなるかが分かる
- 文ごとに、合成にかかった時間・音声の長さ・RTF（合成時間 ÷ 音声の長さ）を出す

結果は emit(kind, ...) で知らせる（別のスレッドから呼ばれる）:
    ("status", 文)  ("loaded", voices, 秒)  ("error", 文)
    ("audio", 文の番号, audio, rate)  ("sentence", i, n, 文, 合成 ms, 音声の秒数)
    ("done", 最初の音までの ms | None, 全体の ms, 音声の秒数, 文の数)
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ttskit.backends import BACKENDS, VoicevoxCompatible, create, load_config, split_sentences  # noqa: E402

DEFAULT_ENGINE = "aivisspeech"
HTTP_ENGINES = ("aivisspeech", "voicevox")  # アプリからエンジンを起動できるもの

SAMPLES = {
    "あいさつ": "こんにちは。今日はいい天気ですね。何かお手伝いできることはありますか？",
    "数字・日時": "次の会議は8月19日午後3時15分です。売上は1,234万円、気温は31.5度でした。",
    "長めの文": "音声合成の品質を確認しています。東京駅から新宿駅まで電車で移動し、橋の端を、箸を持って歩きました。"
              "固有名詞や数字を含む文章でも、自然な間と正しいアクセントで読み上げられることが重要です。",
    "音声対話の返事": "それは大変でしたね。お昼を食べられなかったなら、今日は少し早めに休んでください。",
}

Emit = Callable[..., None]


def engine_choices() -> list[dict]:
    """[{name, label, note, http}]。GUI と Web のエンジンの一覧。"""
    return [{"name": b.name, "label": b.label, "note": b.note, "http": b.name in HTTP_ENGINES} for b in BACKENDS.values()]


def to_wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TtsSession:
    """エンジンの読み込みと、文ごとの合成（別スレッド）。"""

    def __init__(self, emit: Emit, config: dict | None = None) -> None:
        self.emit = emit
        self.config = config if config is not None else load_config()
        self.tts = None
        self.engine_name: str | None = None
        self.voices: list[tuple[str, str]] = []
        self.loading = False
        self.speaking = False
        self.cancel = threading.Event()
        self.last_audio: list[tuple[np.ndarray, int]] = []
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.tts is not None and not self.loading

    # ---------------------------------------------------------------- engine
    def load(self, name: str) -> None:
        if name not in BACKENDS:
            self.emit("error", f"知らないエンジンです: {name}")
            return
        self.stop()
        self.loading = True
        self.emit("status", "読み込み中…（初回はモデルのダウンロードに時間がかかります）")

        def run() -> None:
            try:
                with self._lock:
                    self._close_engine()
                    t0 = time.perf_counter()
                    tts = create(name, self.config)
                    if not tts.available():
                        raise RuntimeError("エンジンにつながりません。エンジンを起動してから読み込んでください")
                    voices = [tuple(v) for v in tts.voices()]
                    if not voices:
                        raise RuntimeError("声が見つかりません")
                    tts.synthesize("テストです。", voice=voices[0][0])  # ウォームアップ
                    self.tts, self.engine_name, self.voices = tts, name, voices
                self.loading = False
                self.emit("loaded", voices, time.perf_counter() - t0)
            except Exception as ex:  # noqa: BLE001
                self.loading = False
                self.emit("error", f"読み込めませんでした: {type(ex).__name__}: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def start_engine(self, name: str, use_gpu: bool = True) -> None:
        """AivisSpeech / VOICEVOX のエンジンを起動する（入っていれば）。"""
        self.emit("status", "エンジンを起動しています…（1 分ほどかかることがあります）")

        def run() -> None:
            try:
                engine = create(name, self.config)
                ok = isinstance(engine, VoicevoxCompatible) and engine.start_engine(use_gpu=use_gpu)
            except Exception:  # noqa: BLE001
                ok = False
            self.emit("status", "エンジンが起動しました。「読み込み」を押してください" if ok
                      else "エンジンを起動できませんでした（既定の場所に入っていないかもしれません）")

        threading.Thread(target=run, daemon=True).start()

    def _close_engine(self) -> None:
        close = getattr(self.tts, "close", None)  # アプリが起動したサーバーや子プロセスを止める
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        self.tts, self.engine_name, self.voices = None, None, []

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._close_engine()

    # ---------------------------------------------------------------- speak
    def speak(self, text: str, voice: str | None = None, speed: float = 1.0, split: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self.ready:
            self.emit("error", "先にエンジンを読み込んでください")
            return
        voice = voice or self.voices[0][0]
        sentences = split_sentences(text) if split else [text]
        self.cancel.clear()
        self.last_audio = []
        self.speaking = True

        def run() -> None:
            t_click = time.perf_counter()
            first = None
            total_audio = 0.0
            try:
                for i, sentence in enumerate(sentences, 1):
                    if self.cancel.is_set():
                        break
                    t0 = time.perf_counter()
                    try:
                        with self._lock:
                            audio, rate = self.tts.synthesize(sentence, voice=voice, speed=speed)
                    except Exception as ex:  # noqa: BLE001
                        self.emit("error", f"合成に失敗しました: {ex}")
                        break
                    ms = (time.perf_counter() - t0) * 1000
                    if first is None:
                        first = (time.perf_counter() - t_click) * 1000
                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                    dur = len(audio) / rate if rate else 0.0
                    total_audio += dur
                    self.last_audio.append((audio, rate))
                    self.emit("audio", i, audio, rate)
                    self.emit("sentence", i, len(sentences), sentence, ms, dur)
            finally:
                self.speaking = False
                self.emit("done", first, (time.perf_counter() - t_click) * 1000, total_audio, len(sentences))

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        self.cancel.set()

    def joined_audio(self) -> tuple[np.ndarray, int] | None:
        """最後に合成した音声をつないだもの。"""
        if not self.last_audio:
            return None
        rate = self.last_audio[0][1]
        return np.concatenate([a for a, _r in self.last_audio]), rate
