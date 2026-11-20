"""同じ GGUF を、LM Studio と公式の llama.cpp（llama-server の CUDA 版・Vulkan 版）で動かして比べる。

    python benchmarks/llm/scripts/bench_runtimes.py
    python benchmarks/llm/scripts/bench_runtimes.py --models lfm2.5-1.2b-jp-202606 --runtimes llamacpp-cuda

測り方は bench_llm.py と同じ（短い返事の TTFT と tok/s、長めの返事の tok/s）。llama-server は全層を GPU に載せ、
コンテキスト 4096、ほかは既定のまま。LM Studio は lms で同じモデルを載せる（LM Studio の中の llama.cpp）。
llama.cpp の実行ファイルは models/src/llamacpp-bin/{cuda,vulkan}（公式のリリースの zip を展開したもの）。
GGUF は LM Studio のダウンロード先をそのまま読む（--gguf-root）。結果は data/results/llm_rt/<モデル>__<実行系>.json。
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent / "bench_llm.py"
BIN = REPO / "models" / "src" / "llamacpp-bin"
# LM Studio での名前 → GGUF（--gguf-root からの場所）
MODELS = {
    "lfm2.5-1.2b-jp-202606": "LiquidAI/LFM2.5-1.2B-JP-202606-GGUF/LFM2.5-1.2B-JP-202606-Q8_0.gguf",
    "elyza-shortcut-1.0-qwen-7b": "mmnga/ELYZA-Shortcut-1.0-Qwen-7B-gguf/ELYZA-Shortcut-1.0-Qwen-7B-Q4_K_S.gguf",
    "llm-jp-4-8b-instruct": "mmnga-o/llm-jp-4-8b-instruct-gguf/llm-jp-4-8b-instruct-Q4_K_M.gguf",
    "qwen3.5-9b": "lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf",
    "gemma-4-26b-a4b-it-japanese-64e": "JMingo/gemma-4-26B-A4B-it-Japanese-64e-GGUF/gemma-4-26B-A4B-it-Japanese-64e-Q4_K_S.gguf",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lms(*args: str) -> None:
    exe = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    subprocess.run([str(exe), *args], check=False, capture_output=True)


def start_llamacpp(kind: str, gguf: Path, context: int) -> tuple[subprocess.Popen, str]:
    port = free_port()
    exe = BIN / kind / "llama-server.exe"
    log = open(REPO / "data" / f"runtime_llamacpp_{kind}.log", "ab")  # noqa: SIM115
    p = subprocess.Popen([str(exe), "-m", str(gguf), "-ngl", "999", "-c", str(context), "--host", "127.0.0.1",
                          "--port", str(port), "--jinja"], stdout=log, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    base = f"http://127.0.0.1:{port}"
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError(f"llama-server が終了しました（exit {p.returncode}）")
        try:
            if json.loads(urllib.request.urlopen(base + "/health", timeout=2).read()).get("status") == "ok":
                return p, base + "/v1"
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    p.kill()
    raise RuntimeError("llama-server の起動がタイムアウトしました")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--runtimes", nargs="*", default=["lmstudio", "llamacpp-cuda", "llamacpp-vulkan"])
    ap.add_argument("--gguf-root", type=Path, default=Path("D:/lmstudio_models"))
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    out_dir = REPO / "data" / "results" / "llm_rt"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for rt in args.runtimes:
            out = out_dir / f"{model}__{rt}.json"
            if out.is_file():
                continue
            print(f"== {model} {rt}", flush=True)
            proc = None
            try:
                if rt == "lmstudio":
                    lms("server", "start")
                    cmd = [sys.executable, str(BENCH), "--model", model, "--lms-load", "--runs", str(args.runs),
                           "--context", str(args.context), "--out", str(out)]
                else:
                    lms("unload", "--all")  # GPU を空ける
                    proc, base = start_llamacpp(rt.split("-", 1)[1], args.gguf_root / MODELS[model], args.context)
                    cmd = [sys.executable, str(BENCH), "--model", model, "--base", base, "--runs", str(args.runs),
                           "--context", str(args.context), "--out", str(out)]
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                print((r.stdout or "")[-400:], (r.stderr or "")[-400:], flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"  failed: {type(ex).__name__}: {ex}", flush=True)
            finally:
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
    lms("unload", "--all")


if __name__ == "__main__":
    main()
