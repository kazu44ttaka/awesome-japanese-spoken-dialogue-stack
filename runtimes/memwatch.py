"""測定中のメモリ使用量を見張る（GPU のメモリと、この処理と子プロセスのメモリ）。

    watch = MemWatch().start()      # モデルを読み込む前に始める（GPU はここでの使用量を基準にする）
    ...                             # 読み込みと推論
    watch.stop()  # → {"gpu_mem_mb": 読み込み前からの増え方の最大, "ram_mb": 処理と子プロセスの合計の最大}

GPU は nvidia-smi で GPU 全体の使用量を 0.5 秒ごとに見るので、ほかのアプリが同時に GPU のメモリを増減させると
その分もずれる。サーバー（CrispASR など）や別の環境の子プロセスで動くモデルも、子プロセスとして数える。
"""

from __future__ import annotations

import subprocess
import threading


def _gpu_used_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=5,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def _tree_rss_mb() -> float:
    import psutil

    me = psutil.Process()
    total = 0
    for p in [me, *me.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total / 2**20


class MemWatch:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.base_gpu: float | None = None
        self.peak_gpu: float | None = None
        self.peak_ram = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        gpu = _gpu_used_mb()
        if gpu is not None:
            self.peak_gpu = gpu if self.peak_gpu is None else max(self.peak_gpu, gpu)
        try:
            self.peak_ram = max(self.peak_ram, _tree_rss_mb())
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def start(self) -> "MemWatch":
        self.base_gpu = _gpu_used_mb()
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._sample()
        gpu = (None if self.base_gpu is None or self.peak_gpu is None
               else round(max(0.0, self.peak_gpu - self.base_gpu)))
        return {"gpu_mem_mb": gpu, "ram_mb": round(self.peak_ram)}
