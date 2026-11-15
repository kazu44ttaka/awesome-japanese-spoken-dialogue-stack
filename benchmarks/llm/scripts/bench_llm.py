"""音声対話向けに、LLM の「最初の文字が出るまでの時間」と「文字の出る速さ」を測る。

OpenAI 互換 API（LM Studio / llama.cpp server / vLLM など）にストリーミングで投げる。

    python benchmarks/llm/scripts/bench_llm.py --model gemma-4-e4b-it-japanese@q4_k_m --lms-load

- 音声対話のような短い返事（max_tokens 128）を、毎回ちがう利用者の発話で 1 回のウォームアップ + N 回投げる
- TTFT: 送ってから最初の「見える文字」（content）が届くまで。考える（reasoning）モデルは、
  考えている間は返事が始まらないので、その時間も含めて数える（reasoning_ms として別にも出す）
- tok/s: 最初の文字から最後までの、出力トークン数 ÷ 時間（短い返事は誤差が大きいので、
  長めの返事 2 回でも測る: tok_s_long_median）
- 考える（thinking）モードは既定で切る（--think で切らずに測る）
- --lms-load: LM Studio の lms コマンドで、ほかのモデルを外してからこのモデルだけを GPU に載せる
標準ライブラリだけで動く。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

SYSTEM = (
    "あなたは音声対話のアシスタントです。話し言葉で、1〜2 文の短い日本語で返事をしてください。"
    "箇条書きや記号、絵文字は使わないでください。"
)
UTTERANCES = [
    "こんにちは。今日はちょっと疲れちゃった。",
    "最近、夜なかなか眠れないんだよね。",
    "週末にどこか出かけたいんだけど、おすすめある？",
    "昨日、友達とラーメンを食べに行ったよ。",
    "仕事が忙しくて、お昼も食べられなかったんだ。",
    "明日は雨が降るみたいだね。",
]
# 文字の出る速さ（tok/s）は短い返事だと誤差が大きいので、長めの返事でも測る
LONG_PROMPT = "音声対話システムの仕組みを、200 文字くらいで説明してください。"


def lms_path() -> str | None:
    found = shutil.which("lms")
    if found:
        return found
    candidate = Path.home() / ".lmstudio" / "bin" / "lms.exe"
    return str(candidate) if candidate.is_file() else None


def lms_load(model: str, context: int) -> None:
    lms = lms_path()
    if lms is None:
        raise SystemExit("lms コマンドが見つかりません（LM Studio の CLI）")
    subprocess.run([lms, "unload", "--all"], check=False, capture_output=True)
    subprocess.run([lms, "load", model, "--gpu", "max", "-c", str(context), "-y"], check=True, capture_output=True)


def stream_once(base: str, model: str, user: str, max_tokens: int, timeout: float, think: bool = False) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not think:
        # 音声対話では考える（thinking）モードを切る。LM Studio は reasoning_effort、
        # vLLM / llama.cpp は chat_template_kwargs を見る（知らない方は無視される）
        body["reasoning_effort"] = "none"
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_content = first_reasoning = None
    chunks = 0
    text: list[str] = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    if first_reasoning is None:
                        first_reasoning = time.perf_counter()
                if delta.get("content"):
                    if first_content is None:
                        first_content = time.perf_counter()
                    chunks += 1
                    text.append(delta["content"])
    t_end = time.perf_counter()
    out_tokens = (usage or {}).get("completion_tokens") or chunks
    ttft = (first_content - t0) * 1000 if first_content else None
    gen_s = (t_end - first_content) if first_content else 0.0
    return {
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "reasoning_ms": round((first_content - first_reasoning) * 1000, 1) if first_reasoning and first_content else 0.0,
        "total_ms": round((t_end - t0) * 1000, 1),
        "out_tokens": out_tokens,
        "tok_s": round((out_tokens - 1) / gen_s, 1) if gen_s > 0 and out_tokens > 1 else None,
        "chars": len("".join(text)),
        "text": "".join(text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:1234/v1")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--lms-load", action="store_true")
    ap.add_argument("--think", action="store_true", help="考える（thinking）モードを切らずに測る")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load_s = None
    if args.lms_load:
        t = time.perf_counter()
        lms_load(args.model, args.context)
        load_s = round(time.perf_counter() - t, 1)

    stream_once(args.base, args.model, UTTERANCES[-1], args.max_tokens, args.timeout, args.think)  # ウォームアップ
    runs = [stream_once(args.base, args.model, UTTERANCES[i % (len(UTTERANCES) - 1)], args.max_tokens, args.timeout,
                        args.think)
            for i in range(args.runs)]
    longs = [stream_once(args.base, args.model, LONG_PROMPT, 320, args.timeout, args.think) for _ in range(2)]
    ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
    speeds = [r["tok_s"] for r in runs if r["tok_s"]]
    summary = {
        "model": args.model,
        "think": args.think,
        "runs": len(runs),
        "load_s": load_s,
        "ttft_median_ms": round(statistics.median(ttfts), 1) if ttfts else None,
        "ttft_worst_ms": max(ttfts) if ttfts else None,
        "reasoning_median_ms": round(statistics.median(r["reasoning_ms"] for r in runs), 1),
        "tok_s_median": round(statistics.median(speeds), 1) if speeds else None,
        "out_tokens_median": statistics.median(r["out_tokens"] for r in runs),
        "tok_s_long_median": round(statistics.median(r["tok_s"] for r in longs if r["tok_s"]), 1)
        if any(r["tok_s"] for r in longs) else None,
        "no_content_runs": sum(1 for r in runs if r["ttft_ms"] is None),
        "sample": runs[0]["text"][:80],
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, "detail": runs, "long": longs}, ensure_ascii=False, indent=2),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
