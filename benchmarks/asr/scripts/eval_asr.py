"""日本語 ASR の CER と 1 発話あたりの処理時間を測る。

    python benchmarks/asr/scripts/eval_asr.py --backend granite --manifest <manifest.json> --wav-dir <dir>

manifest は [{"wav": "jsut_00000.wav", "ref": "正解文"}, ...] の JSON（16-bit PCM の WAV。16 kHz 以外は読み込み時に変換する）。
評価用の音声と正解文はライセンス上リポジトリに入れない（benchmarks/README.md 参照）。

- CER（句読点あり）: NFKC 正規化と空白の除去だけをした文字誤り率（2026-08 の表と同じ）
- CER（句読点なし）: さらに句読点・記号を除いた値（句読点を出さないモデルも公平に比べるため）
- 時間: 1 発話目で一度ウォームアップしてから全発話を測る。平均と最悪を出す

結果は --out に JSON で書く（発話ごとの CER と時間。認識結果の文は --keep-text のときだけ）。
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np



def norm(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").strip())


def norm_nopunct(text: str) -> str:
    # 長音符「ー」は語の一部なので残す
    return re.sub(r"[\s、。，．,.!?！？「」『』（）()\[\]【】・：:；;\"'“”‘’…〜~]", "", norm(text))


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


def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM の WAV を読み、mono 16 kHz の float32 にする（JSUT は 48 kHz）。"""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: 16-bit PCM の WAV が必要です")
        channels, rate = w.getnchannels(), w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        import torch
        import torchaudio.functional as F

        data = F.resample(torch.from_numpy(data), rate, 16000).numpy()
    return np.ascontiguousarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------- backends
class GraniteCTC:
    """IBM Granite Speech 5.0 TurboCTC（transformers>=5.16、AutoModelForCTC）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", device=self.device)
        inputs = inputs.to(self.device, dtype=self.model.dtype)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


class Moonshine:
    """Moonshine（transformers の MoonshineForConditionalGeneration）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import AutoProcessor, MoonshineForConditionalGeneration

        self.torch = torch
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = MoonshineForConditionalGeneration.from_pretrained(model_id).to(device, self.dtype).eval()
        self.device = device

    def __call__(self, audio: np.ndarray) -> str:
        sr = self.processor.feature_extractor.sampling_rate
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device, self.dtype)
        # 公式の例と同じく、音声 1 秒あたり 13 トークンを上限にする（繰り返しの暴走を防ぐ）
        max_length = int((inputs.attention_mask.sum(dim=-1) * (13 / sr)).max().item())
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_length=max(8, max_length))
        return self.processor.decode(ids[0], skip_special_tokens=True)


class FunASRNano:
    """Fun-ASR-Nano / Fun-ASR-MLT-Nano（funasr の AutoModel + モデル同梱の model.py）。"""

    def __init__(self, model_id: str, device: str, language: str) -> None:
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(model_id))
        kwargs = {"trust_remote_code": True, "device": device, "disable_update": True}
        if (local / "model.py").is_file():
            kwargs["remote_code"] = str(local / "model.py")
        self.model = AutoModel(model=str(local), **kwargs)
        self.language = language

    def __call__(self, audio: np.ndarray) -> str:
        import torch

        # funasr の Fun-ASR-Nano は音声を torch.Tensor かファイルパスでしか受け取らない
        res = self.model.generate(
            input=[torch.from_numpy(audio)], cache={}, batch_size=1, language=self.language, itn=True,
            llm_kwargs={"do_sample": False},
        )
        text = str(res[0].get("text", "")) if res else ""
        return re.sub(r"<\|[^|]+\|>", "", text)


BACKENDS = {
    "granite": lambda dev: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc", dev),
    "granite-nc": lambda dev: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc-nc", dev),
    "moonshine-tiny-ja": lambda dev: Moonshine("UsefulSensors/moonshine-tiny-ja", dev),
    "fun-asr-mlt-nano": lambda dev: FunASRNano("FunAudioLLM/Fun-ASR-MLT-Nano-2512", dev, "日文"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--wav-dir", type=Path, default=None, help="manifest の wav の場所（省略時は manifest と同じフォルダー）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-text", action="store_true", help="認識結果の文も JSON に残す")
    args = ap.parse_args()

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    wav_dir = args.wav_dir or args.manifest.parent
    audios = [read_wav(wav_dir / r["wav"]) for r in rows]

    t_load = time.perf_counter()
    recognize = BACKENDS[args.backend](args.device)
    load_s = time.perf_counter() - t_load
    recognize(audios[0])  # ウォームアップ

    results = []
    for row, audio in zip(rows, audios):
        t0 = time.perf_counter()
        hyp = recognize(audio)
        ms = (time.perf_counter() - t0) * 1000
        ref = row.get("ref") or row.get("reference") or row.get("text") or ""
        item = {
            "utt": Path(row["wav"]).stem,
            "cer": cer(norm(ref), norm(hyp)),
            "cer_nopunct": cer(norm_nopunct(ref), norm_nopunct(hyp)),
            "ms": round(ms, 1),
            "audio_s": round(len(audio) / 16000, 3),
        }
        if args.keep_text:
            item["ref"], item["hyp"] = ref, hyp
        results.append(item)
        print(f"{item['utt']} CER={item['cer']:.3f} nopunct={item['cer_nopunct']:.3f} {ms:.0f} ms | {hyp}", flush=True)

    n = len(results)
    summary = {
        "backend": args.backend,
        "device": args.device,
        "count": n,
        "load_s": round(load_s, 1),
        "mean_cer": round(sum(r["cer"] for r in results) / n, 4),
        "mean_cer_nopunct": round(sum(r["cer_nopunct"] for r in results) / n, 4),
        "mean_ms": round(sum(r["ms"] for r in results) / n, 1),
        "worst_ms": max(r["ms"] for r in results),
        "rtf": round(sum(r["ms"] for r in results) / 1000 / sum(r["audio_s"] for r in results), 4),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
