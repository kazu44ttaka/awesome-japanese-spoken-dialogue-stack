"""bench_tts.py で作った音声に、自然さの推定（UTMOS）と、参照音声との話者の近さを付ける。

    python benchmarks/tts/scripts/quality_tts.py                 # data/results/tts/*.json をすべて
    python benchmarks/tts/scripts/quality_tts.py audiocpp-omnivoice cosyvoice3

- utmos: UTMOS22 strong（SpeechMOS の torch.hub 版、1〜5）。英語の読み上げ（BVCC）で学習した推定器なので、
  日本語では値そのものより同じ文どうしの比べ合いに使う。20 文の平均
- spk_sim: 参照音声 data/voices/ref_ja.wav との話者ベクトルのコサイン類似度（microsoft/wavlm-base-plus-sv）。
  参照音声で声をクローンする構成だけ意味がある
結果は data/results/tts_quality/<名前>.json。モデルは models/hf と models/torch に置く。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
os.environ.setdefault("TORCH_HOME", str(REPO / "models" / "torch"))
os.environ.setdefault("HF_HUB_CACHE", str(REPO / "models" / "hf"))
REF = REPO / "data" / "voices" / "ref_ja.wav"


def load16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    a, rate = sf.read(str(path), dtype="float32", always_2d=True)
    a = a.mean(axis=1)
    if rate != 16000:
        a = F.resample(torch.from_numpy(np.ascontiguousarray(a)), rate, 16000).numpy()
    return np.ascontiguousarray(a, dtype=np.float32)


class Scorer:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch, self.device = torch, device
        self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
        self.ref = self.embed(load16k(REF))

    def utmos(self, a: np.ndarray) -> float:
        with self.torch.no_grad():
            return float(self.mos(self.torch.from_numpy(a)[None].to(self.device), 16000).item())

    def embed(self, a: np.ndarray):
        x = self.fe(a, sampling_rate=16000, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            e = self.sv(**x).embeddings[0]
        return self.torch.nn.functional.normalize(e, dim=-1)

    def spk_sim(self, a: np.ndarray) -> float:
        return float((self.embed(a) * self.ref).sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res_dir = REPO / "data" / "results" / "tts"
    out_dir = REPO / "data" / "results" / "tts_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.names or sorted(p.stem for p in res_dir.glob("*.json"))
    sc = Scorer(args.device)
    for name in names:
        out = out_dir / f"{name}.json"
        src = res_dir / f"{name}.json"
        if (out.is_file() and not args.force) or not src.is_file():
            continue
        rows = json.loads(src.read_text(encoding="utf-8")).get("rows", [])
        res = []
        for r in rows:
            wav = REPO / r["wav"] if r.get("wav") else None
            if not wav or not wav.is_file():
                continue
            a = load16k(wav)
            if len(a) < 1600:
                continue
            res.append({"i": r["i"], "utmos": round(sc.utmos(a), 3), "spk_sim": round(sc.spk_sim(a), 3)})
        if not res:
            continue
        summary = {"name": name, "count": len(res),
                   "mean_utmos": round(statistics.mean(x["utmos"] for x in res), 3),
                   "mean_spk_sim": round(statistics.mean(x["spk_sim"] for x in res), 3)}
        out.write_text(json.dumps({**summary, "rows": res}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    # 比べる目安：参照音声そのものと、同じ人の別の発話（Common Voice の同じ話者）は用意していないので、参照音声だけ
    ref = load16k(REF)
    (out_dir / "_reference.json").write_text(json.dumps({"name": "ref_ja.wav", "mean_utmos": round(sc.utmos(ref), 3)}),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
