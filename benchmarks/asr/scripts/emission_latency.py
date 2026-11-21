"""ストリーミング ASR で、音素が話し終わってから、その音素にあたる文字が出るまでの遅れを出す。

    python benchmarks/asr/scripts/emission_latency.py audiocpp-nemotron-streaming ...   # 省くと jsut-align のすべて

stream_asr.py を音素の時刻付きの評価データ（make_jsut_align.py の jsut-align-60）で動かした結果
（data/results/asr_stream_jsut/<名前>.json の途中の文字とその時刻）を使う:
1. 途中の文字（と確定した文字）を pyopenjtalk で音素にし、正解の音素の並び（jsut-lab の時刻付き）と編集距離で対応づける
2. 正解の各音素について
   - first: 対応する音素が初めて出た時刻
   - stable: 出たあと最後まで消えなくなった時刻（あとで書き換わるモデルのため）
   から、その音素が話し終わった時刻を引いたものが遅れ
3. 確定までに一度も対応がつかなかった音素（認識の誤り・表記の揺れ）は数えない
結果は data/results/emission/<名前>.json。表記を音素に直すので、数字や英字の読み方の違いはずれとして残る。
"""

from __future__ import annotations

import json
import re
import statistics
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RES = REPO / "data" / "results" / "asr_stream_jsut"
OUT = REPO / "data" / "results" / "emission"
MANIFEST = REPO / "data" / "eval" / "jsut-align-60" / "manifest.json"
_PUNCT = re.compile(r"[\s、。，．,.!?！？「」『』（）()\[\]【】・：:；;\"'“”‘’…〜~]")


@lru_cache(maxsize=4096)
def to_phones(text: str) -> tuple[str, ...]:
    import pyopenjtalk

    t = _PUNCT.sub("", unicodedata.normalize("NFKC", text or ""))
    if not t:
        return ()
    return tuple(p.lower() for p in pyopenjtalk.g2p(t).split() if p not in ("pau", "sil"))


def matched(ref: list[str], hyp: tuple[str, ...]) -> set[int]:
    """編集距離の最短の対応で、hyp と同じ音素に対応した ref の位置。"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    # 途中の文字は話した分の頭だけなので、正解の後ろの残り（まだ話していない分）は数えずに、正解の頭と対応づける
    i = min(range(n + 1), key=lambda k: (d[k][m], -k))
    out, j = set(), m
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] == hyp[j - 1]:
                out.add(i - 1)
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def one(name: str, man: dict) -> dict | None:
    src = RES / f"{name}.json"
    if not src.is_file():
        return None
    data = json.loads(src.read_text(encoding="utf-8"))
    first_d, stable_d, covered, total, after = [], [], 0, 0, []
    for r in data["rows"]:
        ph = man[r["utt"]]
        ref = [p[0].lower() for p in ph]
        ends = [p[2] for p in ph]
        if r.get("t_final_s") is None:  # 確定した文字が返らなかった発話は数えない
            continue
        events = [(t, x) for t, x in r.get("events", [])] + [(r["t_final_s"], r["hyp"])]
        sets = [matched(ref, to_phones(x)) for _, x in events]
        total += len(ref)
        after.append(r["t_final_s"] - ends[-1])  # 最後の音素を話し終えてから確定まで（音声の後ろの無音を含めない）
        final = sets[-1]
        for i in range(len(ref)):
            hits = [k for k, s in enumerate(sets) if i in s]
            if not hits:
                continue
            covered += 1
            first_d.append(events[hits[0]][0] - ends[i])
            if i in final:  # 確定まで続けて出ている最初の時点（確定で消えた音素は stable に数えない）
                k = len(sets) - 1
                while k > 0 and i in sets[k - 1]:
                    k -= 1
                stable_d.append(events[k][0] - ends[i])
    if not first_d:
        return None
    ms = lambda v: round(v * 1000)  # noqa: E731
    return {"name": name, "utterances": len(data["rows"]), "phones": total, "matched": round(covered / total, 3),
            "first_p50_ms": ms(pct(first_d, 0.5)), "first_p90_ms": ms(pct(first_d, 0.9)),
            "first_mean_ms": ms(statistics.mean(first_d)),
            "stable_p50_ms": ms(pct(stable_d, 0.5)), "stable_p90_ms": ms(pct(stable_d, 0.9)),
            "stable_mean_ms": ms(statistics.mean(stable_d)), "corpus_cer_kana": data.get("corpus_cer_kana"),
            "final_after_speech_p50_ms": ms(pct(after, 0.5)), "final_after_speech_p90_ms": ms(pct(after, 0.9))}


def main() -> None:
    man = {Path(r["wav"]).stem: r["phones"] for r in json.loads(MANIFEST.read_text(encoding="utf-8"))}
    names = sys.argv[1:] or sorted(p.stem for p in RES.glob("*.json"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        res = one(name, man)
        if res:
            (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
