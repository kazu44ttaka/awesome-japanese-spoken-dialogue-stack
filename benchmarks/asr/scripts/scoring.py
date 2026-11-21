"""日本語 ASR の採点（3 種類の CER と、ブートストラップによる 95% 信頼区間）。

- cer         : NFKC 正規化と空白の除去だけ（句読点あり。2026-08 の表と同じ）
- cer_nopunct : さらに句読点・記号を除く（句読点を出さないモデルも公平に比べる）
- cer_kana    : 句読点を除き、正解と結果の両方をカタカナの読みに直して比べる
                （「昔々」と「むかしむかし」のような表記の揺れを打ち消す。fugashi + unidic-lite）

2 通りの集計を出す:
- mean_*   : 発話ごとの CER の平均（2026-08 の表と同じ）
- corpus_* : 全体の CER（誤りの合計 ÷ 正解の文字数の合計）。論文で一般的。正解が音声より
             短い発話（テレビ音声など）で値が 1 を超えても、平均ほど引きずられない
どちらも、発話を復元抽出して集計し直すことを繰り返し、2.5 / 97.5 パーセンタイルを 95% 信頼区間にする。
"""

from __future__ import annotations

import random
import re
import unicodedata
from functools import lru_cache

_PUNCT = r"[\s、。，．,.!?！？「」『』（）()\[\]【】・：:；;\"'“”‘’…〜~]"


def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").strip())


def norm_nopunct(text: str) -> str:
    # 長音符「ー」は語の一部なので残す
    return re.sub(_PUNCT, "", norm(text))


@lru_cache(maxsize=1)
def _tagger():
    import fugashi

    return fugashi.Tagger()


def _hira_to_kata(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


_DIGITS = "〇一二三四五六七八九"
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _int_to_kanji(s: str) -> str:
    """"2026" → "二千二十六"。頭が 0 の数字列（"007" など）は 1 桁ずつ読む。"""
    if len(s) > 1 and s[0] == "0" or len(s) > 16:
        return "".join(_DIGITS[int(c)] for c in s)
    n = int(s)
    if n == 0:
        return "ゼロ"
    out = ""
    for big, unit in ((10**12, "兆"), (10**8, "億"), (10**4, "万"), (1, "")):
        part, n = divmod(n, big)
        if not part:
            continue
        small = ""
        for v, u in ((1000, "千"), (100, "百"), (10, "十")):
            d, part = divmod(part, v)
            if d:
                small += ("" if d == 1 else _DIGITS[d]) + u
        if part:
            small += _DIGITS[part]
        out += small + unit
    return out


def _num_to_kanji(m: re.Match) -> str:
    whole, _, frac = m.group(0).replace(",", "").partition(".")
    return _int_to_kanji(whole) + ("点" + "".join(_DIGITS[int(c)] for c in frac) if frac else "")


def norm_kana(text: str) -> str:
    """句読点を除いてカタカナの読みにする。辞書に読みが無い語（英字など）は表記のまま。

    算用数字は、句読点を除く前に漢数字に直してから読む（「72 時間」と「七十二時間」を同じ読みにする）。
    """
    text = _NUM.sub(_num_to_kanji, unicodedata.normalize("NFKC", text or "")).replace("%", "パーセント")
    plain = norm_nopunct(text)
    if not plain:
        return ""
    out = []
    for word in _tagger()(plain):
        kana = getattr(word.feature, "kana", None)
        out.append(kana if kana and kana != "*" else word.surface)
    return _hira_to_kata("".join(out))


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return lev(ref, hyp) / max(1, len(ref))


def score_pair(ref: str, hyp: str) -> dict[str, float]:
    """3 種類の CER と、全体の CER を出すための誤り数・正解の文字数。"""
    out: dict[str, float] = {}
    for key, fn in (("cer", norm), ("cer_nopunct", norm_nopunct), ("cer_kana", norm_kana)):
        r, h = fn(ref), fn(hyp)
        edits = lev(r, h)
        out[key] = edits / max(1, len(r))
        out[f"{key}_edits"] = edits
        out[f"{key}_len"] = len(r)
    return out


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """発話ごとの値の平均について、95% 信頼区間を返す。"""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return (means[int(0.025 * iters)], means[int(0.975 * iters) - 1])


def bootstrap_corpus_ci(edits: list[int], lens: list[int], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """全体の CER（誤りの合計 ÷ 文字数の合計）について、95% 信頼区間を返す。"""
    if not edits:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(edits)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(sum(edits[i] for i in idx) / max(1, sum(lens[i] for i in idx)))
    vals.sort()
    return (vals[int(0.025 * iters)], vals[int(0.975 * iters) - 1])


def summarize(rows: list[dict]) -> dict:
    """rows は score_pair の結果を含む dict のリスト。平均・全体の CER と 95% 信頼区間をまとめる。"""
    out: dict = {"count": len(rows)}
    for key in ("cer", "cer_nopunct", "cer_kana"):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        out[f"mean_{key}"] = round(sum(vals) / len(vals), 4)
        out[f"ci95_{key}"] = [round(lo, 4), round(hi, 4)]
        edits = [r[f"{key}_edits"] for r in rows if f"{key}_edits" in r]
        lens = [r[f"{key}_len"] for r in rows if f"{key}_len" in r]
        if edits:
            clo, chi = bootstrap_corpus_ci(edits, lens)
            out[f"corpus_{key}"] = round(sum(edits) / max(1, sum(lens)), 4)
            out[f"ci95_corpus_{key}"] = [round(clo, 4), round(chi, 4)]
    return out
