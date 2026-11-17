"""ASR デモの中身（PySide6 版 asr_demo.py と Web 版 asr_web.py が共通で使う）。

マイクの音（このアプリが sounddevice で取る、またはブラウザから送られてくる 16 kHz の float32）を
発話区間検出で区切り、区間ごとに認識して確定する。区切り方と疑似ストリーミングは MaAI-SDS の
ASR（src/maai_sds/asr/service.py）と同じ:

- Silero VAD（silero-vad の ONNX）の確率を 512 サンプル（32 ms）ごとに出す
- 確率が「開始のしきい値」を超える状態が「開始の時間」続いたら発話の始まり。その前の「プリロール」秒も区間に入れる
- 確率が「終了のしきい値」を下回る状態が「終了の時間」続いたら発話の終わり。区間を認識して確定する
- 区間が「最短の長さ」より短いときは認識せず、直前の途中経過をそのまま確定にする
- 疑似ストリーミング: 発話の始まりと、その後は音声「間隔」秒ごとに、区間の頭からの音声を認識し直して途中経過を出す。
  認識が間に合わないときは古い依頼を捨てて、いちばん新しい音声だけを認識する（遅れがたまらない）
- 「真のストリーミング」のモデル（Nemotron 3.5 streaming）は、発話の始まりから音声を少しずつ入れ、
  内部状態を引き継ぎながら途中経過を出す（疑似ストリーミングの設定は使わない）

モデルの状態は MaAI-SDS と同じ 4 つ: 停止（stopped）・起動中（starting）・稼働（running）・異常（error）。

結果は emit(kind, ...) で知らせる（別のスレッドから呼ばれる）:
    ("state", state, 説明)  ("status", 文)  ("error", 文)  ("level", 0〜1)  ("vad", 確率)
    ("partial", text, ms)  ("final", text, 認識 ms, 音声の秒数, 話し終わりから表示までの ms)
"""

from __future__ import annotations

import math
import queue
import re
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asrkit.backends import BACKENDS, create, create_streaming  # noqa: E402

RATE = 16000
WINDOW = 512  # Silero VAD の 1 回分（32 ms）
DEFAULT_MODEL = "sherpa-sensevoice"

STATE_LABELS = {"stopped": "停止", "starting": "起動中", "running": "稼働", "error": "異常"}
STATE_COLORS = {"stopped": "#9e9e9e", "starting": "#f0b429", "running": "#43a047", "error": "#e53935"}

Emit = Callable[..., None]


@dataclass
class ListenOptions:
    """区切り方と疑似ストリーミングの設定。既定値は MaAI-SDS の controller_config.json と同じ。"""

    pseudo: bool = True  # 疑似ストリーミング（途中経過も出す）
    pseudo_chunk_sec: float = 0.25  # 途中経過を出す間隔（音声の秒数）
    vad_on_threshold: float = 0.7  # 発話の始まりとみなす確率
    vad_on_sec: float = 0.1  # 始まりとみなすまでに続く時間
    vad_off_threshold: float = 0.3  # 発話の終わりとみなす確率
    vad_off_sec: float = 0.1  # 終わりとみなすまでに続く時間
    pre_roll_sec: float = 0.2  # 始まりの前から区間に入れる長さ
    min_segment_sec: float = 0.4  # これより短い区間は認識しない（直前の途中経過で確定）

    @classmethod
    def from_dict(cls, d: dict) -> "ListenOptions":
        base = asdict(cls())
        out = {}
        for k, v in base.items():
            if k in d and d[k] is not None:
                out[k] = bool(d[k]) if isinstance(v, bool) else float(d[k])
        return cls(**{**base, **out})


def clean_text(text: str) -> str:
    """単語の間に空白を入れるモデルがあるので、日本語の文字の隣の空白は詰める。"""
    return re.sub(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])", "", text or "").strip()


def model_choices() -> list[dict]:
    """[{name, label, gpu, streaming}]。GUI と Web のモデルの一覧。"""
    return [{"name": b.name, "label": b.label, "gpu": b.gpu, "streaming": b.streaming} for b in BACKENDS.values()]


def list_audio_devices(kind: str) -> list[tuple[int | None, str]]:
    """マイク（kind="input"）かスピーカー（"output"）。既定の API の機器だけ、先頭は「既定」。"""
    out: list[tuple[int | None, str]] = [(None, "既定のデバイス")]
    try:
        import sounddevice as sd

        api = sd.default.hostapi
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        for i, d in enumerate(sd.query_devices()):
            if d[key] > 0 and d["hostapi"] == api:
                out.append((i, d["name"]))
    except Exception:  # noqa: BLE001
        pass
    return out


class SileroScorer:
    """silero-vad（ONNX）の確率を 512 サンプルごとに出す（MaAI-SDS の vad_turn.py と同じ読み方）。"""

    def __init__(self) -> None:
        import torch

        threads = torch.get_num_threads()
        import silero_vad  # import すると torch のスレッド数が 1 になるので戻す

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = silero_vad.load_silero_vad(onnx=True)

    def reset(self) -> None:
        self.model.reset_states()

    def __call__(self, window: np.ndarray) -> float:
        return float(self.model(self.torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)), RATE).item())


class AsrSession:
    """モデルの起動・停止と、音声 → VAD → 認識のループ（別スレッド）。"""

    def __init__(self, emit: Emit) -> None:
        self.emit = emit
        self.recognize = None
        self.streamer = None  # 真のストリーミングのモデル
        self.model_name: str | None = None
        self.device: str | None = None
        self.state = "stopped"
        self.state_detail = ""
        self.audio_q: queue.Queue = queue.Queue()
        self.stream = None  # sounddevice の入力（マイクを直接使うとき）
        self.worker: threading.Thread | None = None
        self.infer_worker: threading.Thread | None = None
        self.running = threading.Event()
        self.options = ListenOptions()
        self._vad: SileroScorer | None = None
        self._lock = threading.Lock()  # モデルを同時に使わない（起動と認識も）
        self._jobs: deque = deque()  # 認識の依頼（確定は順に、途中経過は最新の 1 つだけ）
        self._jobs_cv = threading.Condition()
        self._last_partial = ""

    @property
    def ready(self) -> bool:
        return self.recognize is not None and self.state == "running"

    @property
    def listening(self) -> bool:
        return self.running.is_set()

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state, self.state_detail = state, detail
        self.emit("state", state, detail)

    # ---------------------------------------------------------------- model
    def start_model(self, name: str, device: str = "cpu", threads: int = 8) -> None:
        if name not in BACKENDS:
            self.emit("error", f"知らないモデルです: {name}")
            return
        if self.state == "starting":
            return
        if not BACKENDS[name].gpu:
            device = "cpu"
        self.stop()
        self._set_state("starting", f"{BACKENDS[name].label} を読み込み中…（初回はダウンロードに時間がかかります）")

        def run() -> None:
            try:
                with self._lock:
                    self._close_model()
                    t0 = time.perf_counter()
                    if self._vad is None:
                        self._vad = SileroScorer()
                    if BACKENDS[name].streaming:
                        streamer = create_streaming(name, threads=4, device=device)
                        streamer(np.zeros(RATE // 2, dtype=np.float32))  # ウォームアップ
                        self.streamer, self.recognize = streamer, streamer
                    else:
                        recognize = create(name, device=device, threads=threads)
                        recognize(np.zeros(RATE // 2, dtype=np.float32))  # ウォームアップ
                        self.streamer, self.recognize = None, recognize
                    self.model_name, self.device = name, getattr(self.recognize, "device", device)
                self._set_state("running", f"{BACKENDS[name].label}・{self.device}（{time.perf_counter() - t0:.1f} 秒）")
            except Exception as ex:  # noqa: BLE001
                self._set_state("error", f"読み込めませんでした: {type(ex).__name__}: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def stop_model(self) -> None:
        self.stop()

        def run() -> None:
            with self._lock:
                self._close_model()
            self._set_state("stopped", "")

        threading.Thread(target=run, daemon=True).start()

    def _close_model(self) -> None:
        close = getattr(self.recognize, "close", None)  # 前のモデルが起動したサーバーや子プロセスを止める
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        self.recognize = self.streamer = None
        self.model_name = self.device = None

    # ---------------------------------------------------------------- listen
    def start(self, options: ListenOptions, mic: int | None = None, use_local_mic: bool = True) -> None:
        """聞き始める。use_local_mic=False なら、音声は feed() で渡す（ブラウザのマイク）。"""
        if not self.ready:
            raise RuntimeError("先にモデルを「起動」してください")
        self.stop()
        self.options = options
        self.audio_q = queue.Queue()
        with self._jobs_cv:
            self._jobs.clear()
        self._vad.reset()
        self.running.set()
        if use_local_mic:
            import sounddevice as sd

            def callback(indata, _frames, _time, _status) -> None:
                self.audio_q.put(indata[:, 0].copy())

            self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=WINDOW,
                                         device=mic, callback=callback)
            self.stream.start()
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.infer_worker = threading.Thread(target=self._infer_loop, daemon=True)
        self.worker.start()
        self.infer_worker.start()

    def feed(self, chunk: np.ndarray) -> None:
        """ブラウザなどから受け取った 16 kHz・float32 の音声を入れる。"""
        if self.running.is_set():
            self.audio_q.put(np.asarray(chunk, dtype=np.float32))

    def stop(self) -> None:
        self.running.clear()
        with self._jobs_cv:
            self._jobs_cv.notify_all()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:  # noqa: BLE001
                pass
            self.stream = None
        for t in (self.worker, self.infer_worker):
            if t is not None and t is not threading.current_thread():
                t.join(timeout=5)
        self.worker = self.infer_worker = None

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._close_model()

    # ---------------------------------------------------------------- loop
    def _submit(self, job: dict) -> None:
        with self._jobs_cv:
            if job["mode"] == "partial":
                # まだ認識していない途中経過は捨てて、新しいものだけ残す（MaAI-SDS の drop_oldest と同じ）
                self._jobs = deque(j for j in self._jobs if j["mode"] != "partial")
            self._jobs.append(job)
            self._jobs_cv.notify()

    def _loop(self) -> None:
        o = self.options
        frame_hz = RATE / WINDOW
        on_frames = max(1, math.ceil(o.vad_on_sec * frame_hz))
        off_frames = max(1, math.ceil(o.vad_off_sec * frame_hz))
        pre_frames = max(1, math.ceil(o.pre_roll_sec * frame_hz))
        chunk_samples = max(1, int(RATE * o.pseudo_chunk_sec))
        pre: deque = deque(maxlen=pre_frames)
        segment: list[np.ndarray] = []
        active = False
        on_count = off_count = 0
        samples = 0  # 音声の時計（サンプル数）
        next_partial = 0
        pending = np.zeros(0, dtype=np.float32)
        while self.running.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            # 発話中でないときに処理が遅れてたまった音声は捨てて、実時間へ追いつく（MaAI-SDS と同じ）
            if not active:
                while self.audio_q.qsize() > max(2, int(frame_hz * 0.5)):
                    try:
                        self.audio_q.get_nowait()
                    except queue.Empty:
                        break
            self.emit("level", min(1.0, float(np.sqrt(np.mean(chunk ** 2)) + 1e-9) * 8))
            pending = np.concatenate([pending, chunk])
            while len(pending) >= WINDOW:
                window, pending = pending[:WINDOW], pending[WINDOW:]
                samples += WINDOW
                pre.append(window)
                score = self._vad(window)
                self.emit("vad", score)
                if not active:
                    if score > o.vad_on_threshold:
                        on_count += 1
                        if on_count >= on_frames:
                            active, off_count = True, 0
                            segment = list(pre)
                            self._last_partial = ""
                            if self.streamer is not None:
                                self._submit({"mode": "stream_start", "audio": np.concatenate(segment)})
                            elif o.pseudo:
                                self._submit({"mode": "partial", "audio": np.concatenate(segment)})
                            next_partial = samples + chunk_samples
                    else:
                        on_count = 0
                    continue
                segment.append(window)
                if self.streamer is not None:
                    self._submit({"mode": "stream_feed", "audio": window})
                elif o.pseudo and samples >= next_partial:
                    self._submit({"mode": "partial", "audio": np.concatenate(segment)})
                    while next_partial <= samples:
                        next_partial += chunk_samples
                if score < o.vad_off_threshold:
                    off_count += 1
                    if off_count >= off_frames:
                        audio = np.concatenate(segment)
                        self._submit({"mode": "final", "audio": audio, "short": len(audio) < RATE * o.min_segment_sec,
                                      "wait_ms": o.vad_off_sec * 1000})
                        active, on_count, off_count = False, 0, 0
                        segment = []
                        pre.clear()
                else:
                    off_count = 0

    def _infer_loop(self) -> None:
        while self.running.is_set():
            with self._jobs_cv:
                while not self._jobs and self.running.is_set():
                    self._jobs_cv.wait(timeout=0.2)
                if not self.running.is_set():
                    return
                job = self._jobs.popleft()
            mode = job["mode"]
            if self.streamer is not None:
                t0 = time.perf_counter()
                with self._lock:
                    if mode == "stream_start":
                        self.streamer.start()
                        text = clean_text(self.streamer.feed(job["audio"]))
                    elif mode == "stream_feed":
                        text = clean_text(self.streamer.feed(job["audio"]))
                    else:
                        text = clean_text(self.streamer.finish()) if self.streamer.stream is not None else ""
                ms = (time.perf_counter() - t0) * 1000
                if mode == "final":
                    self.emit("final", text, ms, len(job["audio"]) / RATE, job["wait_ms"] + ms)
                elif text and text != self._last_partial:
                    self._last_partial = text
                    self.emit("partial", text, ms)
                continue
            if mode == "final" and job.get("short"):
                # 短すぎる区間は認識しない。直前の途中経過があれば、それを確定にする
                if self._last_partial:
                    self.emit("final", self._last_partial, 0.0, len(job["audio"]) / RATE, job["wait_ms"])
                self._last_partial = ""
                continue
            text, ms = self._run(job["audio"])
            if mode == "partial":
                self._last_partial = text
                self.emit("partial", text, ms)
            else:
                self._last_partial = ""
                self.emit("final", text, ms, len(job["audio"]) / RATE, job["wait_ms"] + ms)

    def _run(self, audio: np.ndarray) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            with self._lock:
                text = self.recognize(audio) if self.recognize else ""
        except Exception as ex:  # noqa: BLE001
            text = f"（エラー: {ex}）"
        return clean_text(text), (time.perf_counter() - t0) * 1000
