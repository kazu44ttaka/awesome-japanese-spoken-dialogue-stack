"""日本語 ASR の CER（句読点あり・なし・読み）と 1 発話あたりの処理時間を測る。

    # 手元の manifest（[{"wav": "...", "ref": "正解文"}, ...]）で測る
    python benchmarks/asr/scripts/eval_asr.py --backend moonshine-tiny-ja --manifest data/eval/jsut20/manifest.json
    # FLEURS 日本語テスト（初回に data/eval/fleurs_ja へ取得する）
    python benchmarks/asr/scripts/eval_asr.py --backend granite --dataset fleurs-ja

評価用の音声と正解文はリポジトリに入れない（data/eval/ と data/results/ は .gitignore）。
採点は scoring.py（3 種類の CER と 95% 信頼区間）。時間は 1 発話目で一度ウォームアップしてから測る。
sherpa-onnx のモデルは --model-dir に k2-fsa の配布物のフォルダーを渡す。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import score_pair, summarize  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------- audio / datasets
def read_wav(path: Path) -> np.ndarray:
    """WAV を読み、mono 16 kHz の float32 にする（JSUT は 48 kHz の 16-bit、FLEURS は float32）。"""
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if rate != 16000:
        import torch
        import torchaudio.functional as F

        data = F.resample(torch.from_numpy(np.ascontiguousarray(data)), rate, 16000).numpy()
    return np.ascontiguousarray(data, dtype=np.float32)


def load_manifest(path: Path, wav_dir: Path | None) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    base = wav_dir or path.parent
    return [{"id": Path(r["wav"]).stem, "wav": base / r["wav"],
             "ref": r.get("ref") or r.get("reference") or r.get("text") or ""} for r in rows]


def load_fleurs_ja(split: str = "test") -> list[dict]:
    """google/fleurs の ja_jp を data/eval/fleurs_ja に取得して使う（CC-BY-4.0）。"""
    from huggingface_hub import hf_hub_download

    root = REPO / "data" / "eval" / "fleurs_ja"
    wav_dir = root / split
    tsv = Path(hf_hub_download("google/fleurs", f"data/ja_jp/{split}.tsv", repo_type="dataset"))
    if not wav_dir.is_dir() or not any(wav_dir.glob("*.wav")):
        tar = Path(hf_hub_download("google/fleurs", f"data/ja_jp/audio/{split}.tar.gz", repo_type="dataset"))
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            for member in t.getmembers():
                if member.isfile() and member.name.endswith(".wav"):
                    member.name = Path(member.name).name
                    t.extract(member, wav_dir)
    rows = []
    with tsv.open(encoding="utf-8") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            # id, ファイル名, 元の文（句読点あり）, 正規化した文, 音素, 長さ, 性別
            if len(rec) < 3 or not (wav_dir / rec[1]).is_file():
                continue
            rows.append({"id": Path(rec[1]).stem, "wav": wav_dir / rec[1], "ref": rec[2]})
    return rows


# Hugging Face の japanese-asr が置いている評価セット（Kotoba-Whisper などのモデルカードと同じもの）。
# 音声は初回に data/eval/<name>/ へ 16 kHz の WAV で書き出し、2 回目からはそれを使う。
HF_SETS = {
    # ReazonSpeech の test（テレビ番組の音声。CDLA-Sharing-1.0、著作権法 30 条の 4 の目的に限る）
    "reazonspeech-test": "japanese-asr/ja_asr.reazonspeech_test",
    # Common Voice 8.0 日本語 test（CC0）
    "cv8-ja": "japanese-asr/ja_asr.common_voice_8_0",
}


def load_hf_set(name: str, subset: int, seed: int = 0) -> list[dict]:
    """parquet を取得し、subset 件を乱数（seed 固定）で選んで WAV に書き出す。"""
    import io
    import random

    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    repo = HF_SETS[name]
    out_dir = REPO / "data" / "eval" / f"{name}-{subset or 'all'}-seed{seed}"
    manifest = out_dir / "manifest.json"
    if manifest.is_file():
        return load_manifest(manifest, None)
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
    table_rows = []
    for f in files:
        table = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
        table_rows.extend(table.to_pylist())
    order = list(range(len(table_rows)))
    if subset and subset < len(order):
        order = sorted(random.Random(seed).sample(order, subset))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in order:
        rec = table_rows[i]
        audio, rate = sf.read(io.BytesIO(rec["audio"]["bytes"]), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16000:
            import torch
            import torchaudio.functional as F

            audio = F.resample(torch.from_numpy(np.ascontiguousarray(audio)), rate, 16000).numpy()
        wav = f"{i:05d}.wav"
        sf.write(str(out_dir / wav), audio, 16000, subtype="PCM_16")
        rows.append({"wav": wav, "ref": rec["transcription"]})
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return load_manifest(manifest, None)


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
    """Fun-ASR-Nano / Fun-ASR-MLT-Nano（funasr の AutoModel）。"""

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


class WhisperHF:
    """Whisper 系（transformers の pipeline。日本語・書き起こしを指定）。"""

    def __init__(self, model_id: str, device: str) -> None:
        import torch
        from transformers import pipeline

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype, device=device)

    def __call__(self, audio: np.ndarray) -> str:
        out = self.pipe({"raw": audio, "sampling_rate": 16000},
                        generate_kwargs={"language": "ja", "task": "transcribe"})
        return str(out.get("text", ""))


class Sherpa:
    """sherpa-onnx のオフライン認識（k2-fsa の配布物。CPU）。"""

    def __init__(self, kind: str, model_dir: Path, threads: int) -> None:
        import sherpa_onnx

        d = Path(model_dir)
        R = sherpa_onnx.OfflineRecognizer
        if kind == "sensevoice":
            self.rec = R.from_sense_voice(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                          num_threads=threads, language="ja", use_itn=True)
        elif kind == "parakeet-ctc":
            self.rec = R.from_nemo_ctc(model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"),
                                       num_threads=threads)
        elif kind == "reazonspeech-k2":
            self.rec = R.from_transducer(
                encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"), decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
                joiner=str(d / "joiner-epoch-99-avg-1.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads)
        elif kind == "cohere":
            self.rec = R.from_cohere_transcribe(encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
                                                tokens=str(d / "tokens.txt"), num_threads=threads, language="ja",
                                                use_punct=True, use_itn=True)
        elif kind == "funasr-nano":
            self.rec = R.from_funasr_nano(encoder_adaptor=str(d / "encoder_adaptor.int8.onnx"), llm=str(d / "llm.int8.onnx"),
                                          embedding=str(d / "embedding.int8.onnx"), tokenizer=str(d / "Qwen3-0.6B"),
                                          num_threads=threads, language="ja", itn=True, max_new_tokens=128)
        else:
            raise ValueError(kind)

    def __call__(self, audio: np.ndarray) -> str:
        stream = self.rec.create_stream()
        stream.accept_waveform(16000, audio)
        self.rec.decode_stream(stream)
        return re.sub(r"<\|[^|]+\|>", "", stream.result.text)


def make_backend(name: str, args: argparse.Namespace):
    dev = args.device
    table = {
        "granite": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc", dev),
        "granite-nc": lambda: GraniteCTC("ibm-granite/granite-speech-5.0-470m-turboctc-nc", dev),
        "moonshine-tiny-ja": lambda: Moonshine("UsefulSensors/moonshine-tiny-ja", dev),
        "fun-asr-mlt-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-MLT-Nano-2512", dev, "日文"),
        "fun-asr-nano": lambda: FunASRNano("FunAudioLLM/Fun-ASR-Nano-2512", dev, "日文"),
        "whisper-large-v3-turbo": lambda: WhisperHF("openai/whisper-large-v3-turbo", dev),
        "kotoba-whisper-v2.0": lambda: WhisperHF("kotoba-tech/kotoba-whisper-v2.0", dev),
    }
    if name in table:
        return table[name]()
    if name.startswith("sherpa-"):
        if not args.model_dir:
            raise SystemExit(f"{name} には --model-dir（k2-fsa の配布物のフォルダー）が必要です")
        return Sherpa(name[len("sherpa-"):], args.model_dir, args.threads)
    raise SystemExit(f"unknown backend: {name}")


BACKENDS = [
    "granite", "granite-nc", "moonshine-tiny-ja", "fun-asr-mlt-nano", "fun-asr-nano",
    "whisper-large-v3-turbo", "kotoba-whisper-v2.0",
    "sherpa-sensevoice", "sherpa-parakeet-ctc", "sherpa-reazonspeech-k2", "sherpa-cohere", "sherpa-funasr-nano",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path)
    src.add_argument("--dataset", choices=["fleurs-ja", *HF_SETS])
    ap.add_argument("--subset", type=int, default=1000, help="reazonspeech-test / cv8-ja から乱数で選ぶ件数（0 で全部）")
    ap.add_argument("--wav-dir", type=Path, default=None, help="manifest の wav の場所（省略時は manifest と同じフォルダー）")
    ap.add_argument("--model-dir", type=Path, default=None, help="sherpa-onnx のモデルのフォルダー")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8, help="sherpa-onnx の CPU スレッド数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="結果の JSON（正解文と認識結果を含むので公開しない場所へ）")
    args = ap.parse_args()

    if args.manifest:
        rows = load_manifest(args.manifest, args.wav_dir)
    elif args.dataset == "fleurs-ja":
        rows = load_fleurs_ja()
    else:
        rows = load_hf_set(args.dataset, args.subset)
    skipped = sum(1 for r in rows if not r["ref"].strip())
    rows = [r for r in rows if r["ref"].strip()]
    if skipped:
        print(f"正解文が空の {skipped} 発話はとばしました（自由発話は書き起こしてから使う）", flush=True)
    if args.limit:
        rows = rows[: args.limit]
    audios = [read_wav(r["wav"]) for r in rows]

    t_load = time.perf_counter()
    recognize = make_backend(args.backend, args)
    load_s = time.perf_counter() - t_load
    recognize(audios[0])  # ウォームアップ

    results = []
    for row, audio in zip(rows, audios):
        t0 = time.perf_counter()
        hyp = recognize(audio)
        ms = (time.perf_counter() - t0) * 1000
        item = {"utt": row["id"], **score_pair(row["ref"], hyp), "ms": round(ms, 1),
                "audio_s": round(len(audio) / 16000, 3), "ref": row["ref"], "hyp": hyp}
        results.append(item)
        print(f"{item['utt']} CER={item['cer']:.3f} kana={item['cer_kana']:.3f} {ms:.0f} ms | {hyp}", flush=True)

    ms_sorted = sorted(r["ms"] for r in results)
    summary = {
        "backend": args.backend,
        "dataset": args.dataset or str(args.manifest),
        "device": "cpu" if args.backend.startswith("sherpa-") else args.device,
        "load_s": round(load_s, 1),
        **summarize(results),
        "mean_ms": round(sum(ms_sorted) / len(ms_sorted), 1),
        "p95_ms": ms_sorted[min(len(ms_sorted) - 1, int(0.95 * len(ms_sorted)))],
        "worst_ms": ms_sorted[-1],
        "rtf": round(sum(ms_sorted) / 1000 / sum(r["audio_s"] for r in results), 4),
        "mean_audio_s": round(sum(r["audio_s"] for r in results) / len(results), 2),
    }
    print(json.dumps(summary, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, "rows": results}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
