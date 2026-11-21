"""ASR に音声を実時間で少しずつ（80 ms ずつ）送り、途中の文字が出るまでと、話し終えてから確定するまでを測る。

    python benchmarks/asr/scripts/stream_asr.py --backend audiocpp-nemotron-streaming
    python benchmarks/asr/scripts/stream_asr.py --backend nemotron-80ms --device cpu   # sherpa-onnx（この python の中）

- first_partial_s: 音声の頭から、最初の途中の文字が届くまで（音声の時刻で数える。話している最中に出れば発話の長さより短い）
- final_lag_ms: 最後の 80 ms を送れる時刻（= 話し終えた時刻）から、確定した文字が届くまで
- CER は確定した文字で採点する（eval_asr.py と同じ scoring.py）
`timing100` から 3 発話おきの 34 発話（ReazonSpeech・Common Voice・FLEURS が混ざる）。結果は data/results/asr_stream/<名前>.json。
比べる相手として、ストリーミングでないモデルを話し終えてから 1 回で認識したときの時間は timing_gpu の mean_ms。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_asr import load_manifest, read_wav  # noqa: E402
from scoring import score_pair, summarize  # noqa: E402

CHUNK = 1280  # 80 ms（16 kHz）
TRIM_SEC = 0.3  # 音素の時刻がある評価データで、最後の音素のあとに残す無音


class _LocalStream:
    """asrkit の真のストリーミング（sherpa-onnx の Nemotron・NeMo の Nemotron）を同じ形で包む。"""

    def __init__(self, name: str, device: str, threads: int) -> None:
        from asrkit.backends import create_streaming

        self.s = create_streaming(name, threads=threads, device=device)
        self.device = getattr(self.s, "device", device)

    def start(self) -> None:
        self.events, self.t_final = [], None
        self.s.start()

    def feed(self, audio: np.ndarray) -> None:
        text = (self.s.feed(audio) or "").strip()
        if text and (not self.events or self.events[-1][1] != text):
            self.events.append((time.perf_counter(), text))

    def finish(self) -> str:
        text = self.s.finish()
        self.t_final = time.perf_counter()
        return (text or "").strip()

    def close(self) -> None:
        close = getattr(self.s, "close", None)
        if callable(close):
            close()


class _AppStream:
    """デモアプリ（apps/asr_core.py。MaAI-SDS の ASR と同じしくみ）の疑似ストリーミングを、そのまま動かす:
    Silero VAD で区切り、音声 pseudo_chunk_sec 秒ごとに区間の頭から認識し直して途中経過を出し、VAD が話し終わりを
    判定したら区間全体を認識して確定する。音声はマイクの代わりに feed で入れる。話し終えたあとも無音を実時間で
    送り続け（tail_sec）、VAD が話し終わりを判定できるようにする。区間が複数に分かれたら確定をつなげる。"""

    tail_sec = 1.5

    def __init__(self, name: str, device: str, threads: int, interval: float, pre_roll: float | None = None) -> None:
        import threading

        sys.path.insert(0, str(REPO / "apps"))
        from asr_core import AsrSession, ListenOptions

        self._lock = threading.Lock()
        self.finals: list[str] = []
        self.partial = ""
        self.events: list = []
        self.t_final = None
        self.session = AsrSession(self._emit)
        from asr_core import BACKENDS, SileroScorer

        if name in BACKENDS:  # アプリで選べるモデルは、アプリと同じ読み込み方で
            self.session.start_model(name, device=device, threads=threads)
            t = time.monotonic()
            while self.session.state in ("stopped", "starting") and time.monotonic() - t < 3600:
                time.sleep(0.2)
            if self.session.state != "running":
                raise RuntimeError(f"モデルを起動できませんでした: {self.session.state_detail}")
        else:  # 実行系の構成（crispasr-・speechcpp-・audiocpp-）は、認識するモデルだけ差し替える
            from asrkit.backends import create

            rec = create(name, device=device, threads=threads)
            rec(np.zeros(8000, dtype=np.float32))
            self.session._vad = SileroScorer()
            self.session.recognize, self.session.streamer = rec, None
            self.session.model_name, self.session.device = name, getattr(rec, "device", device)
            self.session.state = "running"
        self.device = self.session.device or device
        # マイクのように聞き続ける（発話ごとに止めると、VAD が頭から立ち上がって発話の始まりを落とす）
        opts = {"pseudo_chunk_sec": interval, **({"pre_roll_buffer_sec": pre_roll} if pre_roll is not None else {})}
        self.session.start(ListenOptions(**opts), use_local_mic=False)

    def _emit(self, kind: str, *args) -> None:
        if kind not in ("partial", "final"):
            return
        now = time.perf_counter()
        with self._lock:
            text = (args[0] or "").strip()
            if kind == "final":
                if text:
                    self.finals.append(text)
                self.partial = ""
                self.t_final = now
            else:
                self.partial = text
            shown = "".join(self.finals) + self.partial
            if shown and (not self.events or self.events[-1][1] != shown):
                self.events.append((now, shown))

    def start(self) -> None:
        self.session.feed(np.zeros(8000, dtype=np.float32))  # 発話の前の 0.5 秒の無音
        time.sleep(0.3)
        with self._lock:
            self.finals, self.partial, self.events, self.t_final = [], "", [], None

    def feed(self, audio: np.ndarray) -> None:
        self.session.feed(audio)

    def finish(self) -> str:
        with self._lock:
            if self.t_final is None:
                self.t_final = time.perf_counter()
            return "".join(self.finals)

    def close(self) -> None:
        self.session.stop()
        self.session.stop_model()
        time.sleep(1)


PSEUDO = 0.0  # --pseudo で決める（秒）
PRE_ROLL: float | None = None  # --pre-roll（省くとアプリの既定 0.2 秒）


def make(name: str, device: str, threads: int):
    if PSEUDO > 0:
        return _AppStream(name, device, threads, PSEUDO, PRE_ROLL)
    runtime, _, key = name.partition("-")
    if runtime in ("crispasr", "speechcpp", "audiocpp"):
        from runtimes.streaming import create_stream_asr

        return create_stream_asr(runtime, key, device, threads)
    return _LocalStream(name, device, threads)


def run_one(s, audio: np.ndarray) -> dict:
    s.start()
    n = (len(audio) + CHUNK - 1) // CHUNK
    t0 = time.perf_counter()
    for k in range(n):
        wait = t0 + (k + 1) * CHUNK / 16000 - time.perf_counter()  # その 80 ms を話し終えた時刻まで待つ
        if wait > 0:
            time.sleep(wait)
        s.feed(audio[k * CHUNK:(k + 1) * CHUNK])
    t_end = max(time.perf_counter(), t0 + n * CHUNK / 16000)
    tail = int(getattr(s, "tail_sec", 0) * 16000 / CHUNK)  # VAD で区切る方式は、話し終えたあとの無音も実時間で送る
    for k in range(n, n + tail):
        wait = t0 + (k + 1) * CHUNK / 16000 - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        s.feed(np.zeros(CHUNK, dtype=np.float32))
    try:
        text = s.finish()
    except TimeoutError:  # 確定した文字が返らなかった（CrispASR の Qwen3-ASR は、短い発話で返さないことがある）
        first = s.events[0][0] - t0 if s.events else None
        return {"text": s.events[-1][1] if s.events else "", "final_lag_ms": None, "timeout": True,
                "first_partial_s": None if first is None else round(first, 3), "partials": len(s.events),
                "send_late_ms": 0.0, "events": [(round(t - t0, 3), x) for t, x in s.events], "t_final_s": None}
    first = s.events[0][0] - t0 if s.events else None
    return {"text": text, "final_lag_ms": round((s.t_final - t_end) * 1000, 1),
            "first_partial_s": None if first is None else round(first, 3), "partials": len(s.events),
            "send_late_ms": round((t_end - (t0 + n * CHUNK / 16000)) * 1000, 1),
            # 途中の文字の変わり目（音声の頭からの秒, その時点の文字）と、確定した時刻。文字ごとの遅れを後で出すため
            "events": [(round(t - t0, 3), x) for t, x in s.events], "t_final_s": round(s.t_final - t0, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--manifest", type=Path, default=REPO / "data" / "eval" / "timing100" / "manifest.json")
    ap.add_argument("--step", type=int, default=3, help="何発話おきに使うか")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pseudo", type=float, default=0.0,
                    help="デモアプリ（apps/asr_core.py）の疑似ストリーミングで動かし、途中経過を音声のこの秒数ごとに出す（アプリの既定 0.25）")
    ap.add_argument("--pre-roll", type=float, default=None, help="--pseudo のとき、発話の始まりの前から区間に入れる秒数")
    ap.add_argument("--out-dir", default="asr_stream", help="data/results/ の下のフォルダー（JSUT の音素の時刻で測るときは asr_stream_jsut）")
    args = ap.parse_args()
    name = args.name or args.backend
    items = load_manifest(args.manifest, None)[:: args.step]
    if args.limit:
        items = items[: args.limit]
    audios = [read_wav(it["wav"]) for it in items]
    # 音素の時刻がある評価データ（JSUT）は、最後の音素の終わりから TRIM_SEC 秒で切る。ファイルの後ろの無音の長さは
    # 発話ごとに違い、「終わり」を送る方式（実行系のストリーミング）ではその長さがそのまま確定の遅れに入ってしまうため
    phones = {Path(r["wav"]).stem: r["phones"] for r in json.loads(args.manifest.read_text(encoding="utf-8"))
              if r.get("phones")}
    if phones:
        audios = [a[: int((phones[it["id"]][-1][2] + TRIM_SEC) * 16000)] if it["id"] in phones else a
                  for it, a in zip(items, audios)]

    t0 = time.perf_counter()
    global PSEUDO, PRE_ROLL
    PSEUDO, PRE_ROLL = args.pseudo, args.pre_roll
    s = make(args.backend, args.device, args.threads)
    load_s = time.perf_counter() - t0
    rows = []
    try:
        mid = sorted(range(len(audios)), key=lambda i: len(audios[i]))[len(audios) // 2]
        run_one(s, audios[mid])  # ウォームアップ（長さが中くらいの発話）
        for it, audio in zip(items, audios):
            r = run_one(s, audio)
            if r["text"].startswith("[error]"):  # 実行系がエラーを返した（結果は保存しない）
                raise RuntimeError(r["text"][:300])
            row = {"utt": it["id"], "ref": it["ref"], "hyp": r.pop("text"), "audio_s": round(len(audio) / 16000, 2), **r}
            row.update(score_pair(it["ref"], row["hyp"]))
            rows.append(row)
            print(it["id"], row["final_lag_ms"], row["first_partial_s"], row["audio_s"], row["hyp"][:30], flush=True)
    finally:
        s.close()

    lags = [r["final_lag_ms"] for r in rows if r["final_lag_ms"] is not None]
    firsts = [r["first_partial_s"] for r in rows if r["first_partial_s"] is not None]
    # 途中の文字が話し終える前に出た割合（出なければストリーミングになっていない）
    early = [r for r in rows if r["first_partial_s"] is not None and r["first_partial_s"] < r["audio_s"]]
    summary = {"backend": args.backend, "name": name, "device": getattr(s, "device", args.device),
               "load_s": round(load_s, 1), **summarize(rows),
               "mean_final_lag_ms": round(statistics.mean(lags), 1), "p50_final_lag_ms": round(statistics.median(lags), 1),
               "p95_final_lag_ms": round(sorted(lags)[max(0, int(len(lags) * 0.95 + 0.5) - 1)], 1),
               "mean_first_partial_s": round(statistics.mean(firsts), 3) if firsts else None,
               "partial_before_end": round(len(early) / len(rows), 3),
               "mean_audio_s": round(statistics.mean(r["audio_s"] for r in rows), 2),
               "max_send_late_ms": max(r["send_late_ms"] for r in rows),
               "timeouts": sum(1 for r in rows if r.get("timeout"))}
    out = REPO / "data" / "results" / args.out_dir / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
