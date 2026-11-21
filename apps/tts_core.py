"""TTS デモの中身（PySide6 版 tts_demo.py と Web 版 tts_web.py が共通で使う）。

エンジンの場所（モデル・実行ファイル・辞書など）は apps/tts_engines.toml に書く
（apps/tts_engines.example.toml をコピー）。

- 「起動」でエンジンを使えるようにする。AivisSpeech / VOICEVOX はエンジンが動いていなければ立ち上げ、
  どのエンジンも声の一覧を取って 1 回試しに合成する（ウォームアップ）。「停止」で止める
  （このアプリが立ち上げたエンジンだけ止め、前から動いていたエンジンはそのまま）
- 状態は MaAI-SDS と同じ 4 つ: 停止（stopped）・起動中（starting）・稼働（running）・異常（error）
- 「デバイス」は、エンジンが GPU / CPU を選べるときだけ選べる（設定の device / acceleration / --use_gpu に入れる）
- 「文ごとに合成」を入れると、文に分けて 1 文ずつ合成し、できた文から渡す（音声対話と同じ）

結果は emit(kind, ...) で知らせる（別のスレッドから呼ばれる）:
    ("state", state, 説明)  ("voices", [(id, 表示名), ...])  ("status", 文)  ("error", 文)
    ("audio", 文の番号, audio, rate)  ("sentence", i, n, 文, 合成 ms, 音声の秒数)
    ("done", 最初の音までの ms | None, 全体の ms, 音声の秒数, 文の数)
"""

from __future__ import annotations

import copy
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
HTTP_ENGINES = ("aivisspeech", "voicevox")  # 別に動くエンジンへ HTTP でつなぐもの（なければ立ち上げる）

# エンジンごとのデバイスの項目名と選択肢（既定は default）。名前と選択肢は MaAI-SDS の画面に合わせる
# （VOICEVOX Core Device は AUTO / CPU / GPU (CUDA)、PyTorch のエンジンは cuda / cpu）。
# ここに無いエンジンは選べない（固定の説明だけ出す）
DEVICE_SPECS: dict[str, dict] = {
    "aivisspeech": {"label": "AivisSpeech Device", "choices": ["GPU", "CPU"], "default": "GPU"},
    "voicevox": {"label": "VOICEVOX Device", "choices": ["GPU", "CPU"], "default": "GPU"},
    "voicevox_core": {"label": "VOICEVOX Core Device", "choices": ["AUTO", "CPU", "GPU (CUDA)"], "default": "GPU (CUDA)"},
    "irodori": {"label": "Irodori Device", "choices": ["cuda", "cpu"], "default": "cuda"},
    "kitten": {"label": "Kitten Device", "choices": ["cuda", "cpu"], "default": "cuda"},
    "moss_nano": {"label": "MOSS-TTS-Nano Device", "choices": ["cuda", "cpu"], "default": "cpu"},
}
# ggml 系の実行系の構成（crispasr-* / audiocpp-* / speechcpp-*）は GPU と CPU を選べる（speech.cpp の GPU は Vulkan）
for _name in BACKENDS:
    _runtime = _name.partition("-")[0]
    if _runtime in ("crispasr", "audiocpp", "speechcpp"):
        DEVICE_SPECS[_name] = {"label": "Device（cuda は Vulkan で GPU を使う）" if _runtime == "speechcpp" else "Device",
                               "choices": ["cuda", "cpu"], "default": "cuda"}
DEVICE_CHOICES = {k: v["choices"] for k, v in DEVICE_SPECS.items()}
FIXED_DEVICE: dict[str, str] = {
    "supertonic": "CPU",
    "supertonic_cpp": "GPU（CUDA のサーバー）",
    "mio": "サーバーのビルドによる",
    "zonos3": "GPU（CUDA のサーバー）",
    "qwen3_tts_cpp": "GPU（CUDA のサーバー）",
    "lfm_audio": "GPU",
    "qwen3-tts": "GPU",
    "tsukasa": "GPU",
    "kokoro": "既定",
    **{n: "GPU（tts_engines.toml の device）" for n in ("qwen3-tts-0.6b-base", "qwen3-tts-0.6b-customvoice", "qwen3-tts-1.7b-base",
                                                     "qwen3-tts-1.7b-customvoice", "omnivoice", "cosyvoice3",
                                                     "style-bert-vits2", "gpt-sovits")},
}

STATE_LABELS = {"stopped": "停止", "starting": "起動中", "running": "稼働", "error": "異常"}
STATE_COLORS = {"stopped": "#9e9e9e", "starting": "#f0b429", "running": "#43a047", "error": "#e53935"}

SAMPLES = {
    "あいさつ": "こんにちは。今日はいい天気ですね。何かお手伝いできることはありますか？",
    "数字・日時": "次の会議は8月19日午後3時15分です。売上は1,234万円、気温は31.5度でした。",
    "長めの文": "音声合成の品質を確認しています。東京駅から新宿駅まで電車で移動し、橋の端を、箸を持って歩きました。"
              "固有名詞や数字を含む文章でも、自然な間と正しいアクセントで読み上げられることが重要です。",
    "音声対話の返事": "それは大変でしたね。お昼を食べられなかったなら、今日は少し早めに休んでください。",
}

Emit = Callable[..., None]


# 一覧は実行系ごとにまとめて出す（この順）
ENGINE_GROUPS = ["エンジン・元の実装", "CrispASR", "audio.cpp", "speech.cpp"]
_PREFIX_GROUP = {"crispasr": "CrispASR", "audiocpp": "audio.cpp", "speechcpp": "speech.cpp"}
_AUDIOCPP_ENGINES = {"supertonic_cpp", "qwen3_tts_cpp"}  # audio.cpp のサーバーを MaAI-SDS と同じ設定で動かすもの


def _engine_group(name: str) -> str:
    if name in _AUDIOCPP_ENGINES:
        return "audio.cpp"
    return _PREFIX_GROUP.get(name.partition("-")[0], ENGINE_GROUPS[0])


def engine_choices() -> list[dict]:
    """[{name, label, note, http, device_label, devices, default_device, fixed_device, group}]。
    GUI と Web のエンジンの一覧（実行系ごとにまとめた順）。"""
    out = []
    for b in BACKENDS.values():
        spec = DEVICE_SPECS.get(b.name, {})
        out.append({"name": b.name, "label": b.label, "note": b.note, "http": b.name in HTTP_ENGINES,
                    "device_label": spec.get("label", "Device"), "devices": spec.get("choices", []),
                    "default_device": spec.get("default"), "fixed_device": FIXED_DEVICE.get(b.name, "既定"),
                    "group": _engine_group(b.name)})
    return sorted(out, key=lambda e: ENGINE_GROUPS.index(e["group"]))


def to_wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _with_device(config: dict, name: str, device: str | None) -> dict:
    """デバイスの選択を、そのエンジンの設定の書き方に直して入れる。"""
    cfg = copy.deepcopy(config)
    if not device or name not in DEVICE_CHOICES or name in HTTP_ENGINES:
        return cfg
    sec = cfg.setdefault(name, {})
    if name == "voicevox_core":
        sec["acceleration"] = {"AUTO": "AUTO", "CPU": "CPU", "GPU (CUDA)": "GPU"}.get(device, "GPU")
    else:
        sec["device"] = device  # cuda / cpu
    return cfg


class TtsSession:
    """エンジンの起動・停止と、文ごとの合成（別スレッド）。"""

    def __init__(self, emit: Emit, config: dict | None = None) -> None:
        self.emit = emit
        self.config = config if config is not None else load_config()
        self.tts = None
        self.engine_name: str | None = None
        self.device: str | None = None
        self.voices: list[tuple[str, str]] = []
        self.state = "stopped"
        self.state_detail = ""
        self.speaking = False
        self.cancel = threading.Event()
        self.last_audio: list[tuple[np.ndarray, int]] = []
        self._lock = threading.Lock()
        self._watch_stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self.tts is not None and self.state == "running"

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state, self.state_detail = state, detail
        self.emit("state", state, detail)

    # ---------------------------------------------------------------- engine
    def start(self, name: str, device: str | None = None) -> None:
        """エンジンを使えるようにする（HTTP のエンジンは動いていなければ立ち上げる）。"""
        if name not in BACKENDS:
            self.emit("error", f"知らないエンジンです: {name}")
            return
        if self.state == "starting":
            return
        self.cancel.set()
        spec = DEVICE_SPECS.get(name)
        device = (device if device in spec["choices"] else spec["default"]) if spec else None
        self._set_state("starting", f"{BACKENDS[name].label} を起動しています…")

        def run() -> None:
            try:
                with self._lock:
                    self._close_engine()
                    t0 = time.perf_counter()
                    tts = create(name, _with_device(self.config, name, device))
                    if not tts.available():
                        if isinstance(tts, VoicevoxCompatible):
                            self.emit("status", "エンジンが動いていないので立ち上げています…（1 分ほどかかることがあります）")
                            if not tts.start_engine(use_gpu=(device or "GPU") == "GPU"):
                                raise RuntimeError("エンジンを立ち上げられませんでした（既定の場所に入っていないかもしれません）")
                        else:
                            raise RuntimeError("エンジンにつながりません（apps/tts_engines.toml の設定を確かめてください）")
                    self.emit("status", "声の一覧を取って、試しに合成しています…（初回はモデルのダウンロードに時間がかかります）")
                    voices = [tuple(v) for v in tts.voices()]
                    if not voices:
                        raise RuntimeError("声が見つかりません")
                    tts.synthesize("テストです。", voice=voices[0][0])  # ウォームアップ
                    self.tts, self.engine_name, self.device, self.voices = tts, name, device, voices
                self.emit("voices", voices)
                where = device or FIXED_DEVICE.get(name, "既定")
                if name.startswith("speechcpp-") and where == "cuda":
                    where = "vulkan"  # speech.cpp の GPU は Vulkan（ASR の表示と同じ）
                note = ""
                if isinstance(tts, VoicevoxCompatible) and tts.process is None:
                    note = "・前から動いていたエンジンを使用（デバイスはそのエンジンの起動のしかたのまま）"
                self._set_state("running", f"{BACKENDS[name].label}・{where}・声 {len(voices)} 種類"
                                           f"（{time.perf_counter() - t0:.1f} 秒）{note}")
            except Exception as ex:  # noqa: BLE001
                self._set_state("error", f"起動できませんでした: {type(ex).__name__}: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def stop_engine(self) -> None:
        """エンジンを止める（このアプリが立ち上げたサーバーや子プロセスも止める）。"""
        self.cancel.set()

        def run() -> None:
            with self._lock:
                self._close_engine()
            self._set_state("stopped", "")
            self.emit("voices", [])

        threading.Thread(target=run, daemon=True).start()

    def _close_engine(self) -> None:
        close = getattr(self.tts, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        self.tts, self.engine_name, self.device, self.voices = None, None, None, []

    def _watch(self) -> None:
        """HTTP のエンジンが外で止められたら「異常」にする。"""
        while not self._watch_stop.wait(3.0):
            tts = self.tts
            if self.state == "running" and isinstance(tts, VoicevoxCompatible) and not self.speaking:
                if not tts.available():
                    self._set_state("error", "エンジンが応答しません（止まったかもしれません）。もう一度「起動」してください")

    def close(self) -> None:
        self.cancel.set()
        self._watch_stop.set()
        with self._lock:
            self._close_engine()

    # ---------------------------------------------------------------- speak
    def speak(self, text: str, voice: str | None = None, speed: float = 1.0, split: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not self.ready:
            self.emit("error", "先にエンジンを「起動」してください")
            return
        voice = voice or self.voices[0][0]
        sentences = split_sentences(text) if split else [text]
        self.cancel.clear()
        self.last_audio = []
        self.speaking = True
        tts = self.tts

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
                            audio, rate = tts.synthesize(sentence, voice=voice, speed=speed)
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
        """合成を途中でやめる（エンジンは止めない）。"""
        self.cancel.set()

    def joined_audio(self) -> tuple[np.ndarray, int] | None:
        """最後に合成した音声をつないだもの。"""
        if not self.last_audio:
            return None
        rate = self.last_audio[0][1]
        return np.concatenate([a for a, _r in self.last_audio]), rate
