"""実行系どうしの比較の表を、測定結果の JSON から作り、benchmarks/2026-11-runtimes.md の印の間に書き込む。

    python benchmarks/make_runtime_page.py

使う結果（data/results/ の下。無いところは「—」）
- ASR：timing_gpu・timing_cpu（timing100 の 100 発話）、fleurs-ja・reazonspeech-test・cv8-ja（3 つの評価データ）、
  mem_gpu（最長 10 発話でのメモリと読み込み時間）
- TTS：tts（20 文。CPU は <構成>-cpu）、mem_tts（3 文でのメモリと読み込み時間）
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "data" / "results"
PAGE = REPO / "benchmarks" / "2026-11-runtimes.md"

RT = {"crispasr": "CrispASR", "speechcpp": "speech.cpp（Vulkan）", "audiocpp": "audio.cpp"}

# (モデル, [(構成, 実行系の表示, 注)])
ASR_GROUPS = [
    ("Haas-ja v1.0", [("haas-ja", "haas-asr（PyTorch・NeMo）", ""), ("speechcpp-haas-ja", "", ""),
                      ("audiocpp-haas-ja", "", ""), ("crispasr-haas-ja", "", "※1")]),
    ("ReazonSpeech NeMo v2", [("reazonspeech-nemo-v2", "NeMo", ""), ("speechcpp-reazonspeech-nemo-v2", "", ""),
                              ("crispasr-reazonspeech-nemo-v2", "", "※1")]),
    ("Parakeet TDT-CTC 0.6B ja", [("parakeet-tdt-ctc-0.6b-ja", "NeMo", ""),
                                  ("sherpa-parakeet-ctc", "sherpa-onnx（int8）", ""),
                                  ("speechcpp-parakeet-tdt-ctc-0.6b-ja", "", ""),
                                  ("crispasr-parakeet-tdt-ctc-0.6b-ja", "", "※1")]),
    ("Parakeet CTC 1.1B ja", [("parakeet-ctc-1.1b-ja", "NeMo", ""), ("crispasr-parakeet-ctc-1.1b-ja", "", "※2")]),
    ("Qwen3-ASR 0.6B", [("qwen3-asr-0.6b", "qwen-asr（PyTorch）", ""), ("sherpa-qwen3-asr", "sherpa-onnx（int8）", ""),
                        ("speechcpp-qwen3-asr-0.6b", "", ""), ("audiocpp-qwen3-asr-0.6b", "", ""),
                        ("crispasr-qwen3-asr-0.6b", "", "")]),
    ("Qwen3-ASR 1.7B", [("qwen3-asr-1.7b", "qwen-asr（PyTorch）", ""), ("speechcpp-qwen3-asr-1.7b", "", ""),
                        ("audiocpp-qwen3-asr-1.7b", "", ""), ("crispasr-qwen3-asr-1.7b", "", "")]),
    ("Qwen3-ASR 1.7B ja-anime", [("crispasr-qwen3-asr-1.7b-ja-anime", "", "")]),
    ("Fun-ASR-Nano-2512", [("fun-asr-nano", "funasr（PyTorch）", ""), ("sherpa-funasr-nano", "sherpa-onnx", ""),
                           ("audiocpp-fun-asr-nano", "", ""), ("crispasr-fun-asr-nano", "", "")]),
    ("Fun-ASR-MLT-Nano-2512", [("fun-asr-mlt-nano", "funasr（PyTorch）", ""), ("crispasr-fun-asr-mlt-nano", "", "3 データとメモリの測定では最初の発話でサーバーが落ちた")]),
    ("SenseVoiceSmall", [("sherpa-sensevoice", "sherpa-onnx", ""), ("audiocpp-sensevoice", "", ""),
                         ("crispasr-sensevoice", "", "")]),
    ("Cohere Transcribe", [("sherpa-cohere", "sherpa-onnx（int8）", ""), ("audiocpp-cohere-transcribe", "", ""),
                           ("crispasr-cohere-transcribe", "", "Q4_K")]),
    ("Cohere ASR ja v0.1", [("crispasr-cohere-asr-ja-v0.1", "", "")]),
    ("Granite 4.0 1B Speech", [("granite-4.0-1b", "transformers", ""), ("crispasr-granite-4.0-1b", "", "Q4_K")]),
    ("Granite 4.1 2B Speech", [("crispasr-granite-4.1-2b", "", "Q4_K・※3")]),
    ("Voxtral Mini 4B Realtime", [("voxtral-realtime", "transformers", ""), ("audiocpp-voxtral-realtime", "", ""),
                                  ("crispasr-voxtral-realtime", "", "")]),
    ("Nemotron 3.5 ASR streaming 0.6B", [("nemotron-nemo", "NeMo", ""), ("nemotron-560ms", "sherpa-onnx（int8・560 ms）", ""),
                                         ("audiocpp-nemotron-streaming", "", ""),
                                         ("crispasr-nemotron-streaming", "", "F16")]),
    ("Whisper large-v3-turbo", [("whisper-large-v3-turbo", "transformers", ""),
                                ("faster-whisper", "faster-whisper（CTranslate2）", ""),
                                ("crispasr-whisper-large-v3-turbo", "", "whisper.cpp の ggml")]),
    ("LFM2.5-Audio-1.5B-JP", [("crispasr-lfm2.5-audio-jp", "", "Q5_K・※3")]),
]

TTS_GROUPS = [
    ("Supertonic 3", [("supertonic", "supertonic（ONNX・CPU 4 スレッド）", ""), ("audiocpp-supertonic-3", "", ""),
                      ("crispasr-supertonic-3", "", "")]),
    ("Irodori-TTS v4 Small", [("speechcpp-irodori-tts-v4-small", "", ""), ("audiocpp-irodori-tts-v4-small", "", "")]),
    ("Irodori-TTS v4.1 Small", [("irodori", "Irodori-TTS（PyTorch）", "キャプションで声を決める")]),
    ("Irodori-TTS v4.1 Small MF", [("speechcpp-irodori-tts-v4-small-mf", "", ""),
                                   ("audiocpp-irodori-tts-v4-small-mf", "", "")]),
    ("Irodori-TTS v4.1 Anime", [("audiocpp-irodori-tts-v4-anime", "", "")]),
    ("Irodori-TTS Large", [("speechcpp-irodori-tts-large", "", "")]),
    ("Irodori-TTS 500M v3", [("audiocpp-irodori-tts-500m-v3", "", ""), ("crispasr-irodori-tts-500m-v3", "", "")]),
    ("Irodori-TTS 600M v3 VoiceDesign", [("audiocpp-irodori-tts-600m-v3-voicedesign", "", ""),
                                         ("crispasr-irodori-tts-600m-v3-voicedesign", "", "Q4_K")]),
    ("Qwen3-TTS 0.6B Base（クローン）", [("qwen3-tts-0.6b-base", "qwen-tts（PyTorch）", "flash-attn なし"),
                                     ("audiocpp-qwen3-tts-0.6b-base", "", ""), ("crispasr-qwen3-tts-0.6b-base", "", "")]),
    ("Qwen3-TTS 0.6B CustomVoice（Ono_Anna）", [("qwen3-tts-0.6b-customvoice", "qwen-tts（PyTorch）", "flash-attn なし"),
                                             ("speechcpp-qwen3-tts-0.6b-customvoice", "", ""),
                                             ("crispasr-qwen3-tts-0.6b-customvoice", "", "")]),
    ("Qwen3-TTS 1.7B Base（クローン）", [("qwen3-tts-1.7b-base", "qwen-tts（PyTorch）", "flash-attn なし"),
                                     ("audiocpp-qwen3-tts-1.7b-base", "", ""), ("crispasr-qwen3-tts-1.7b-base", "", "")]),
    ("Qwen3-TTS 1.7B CustomVoice（Ono_Anna）", [("qwen3-tts-1.7b-customvoice", "qwen-tts（PyTorch）", "flash-attn なし"),
                                             ("speechcpp-qwen3-tts-1.7b-customvoice", "", ""),
                                             ("audiocpp-qwen3-tts-1.7b-customvoice", "", ""),
                                             ("crispasr-qwen3-tts-1.7b-customvoice", "", "")]),
    ("OmniVoice", [("omnivoice", "omnivoice（PyTorch）", ""), ("audiocpp-omnivoice", "", ""),
                   ("crispasr-omnivoice", "", "")]),
    ("CosyVoice3 0.5B（クローン・カタカナで入力）", [("cosyvoice3", "CosyVoice（公式・PyTorch）", ""),
                                             ("crispasr-cosyvoice3", "", "LLM は Q4_K")]),
    ("MOSS-TTS-Nano（クローン）", [("moss_nano_gpu", "MOSS-TTS-Nano（PyTorch）", ""), ("audiocpp-moss-tts-nano", "", "")]),
    ("MOSS-TTS-Local v1.5（クローン）", [("audiocpp-moss-tts-local", "", "GPU メモリが 12 GB にほぼ届く"), ("crispasr-moss-tts-local", "", "")]),
    ("VoxCPM2（クローン）", [("audiocpp-voxcpm2", "", ""), ("crispasr-voxcpm2", "", "")]),
    ("MioTTS", [("mio", "mio-tts-cpp（0.1B）", ""), ("crispasr-miotts-0.6b", "", "0.6B・温度 0.8"),
                ("audiocpp-miotts-1.7b", "", "1.7B・クローン")]),
    ("Confucius4-TTS（クローン）", [("audiocpp-confucius4-tts", "", ""), ("crispasr-confucius4-tts", "", "")]),
    ("Fish Audio S2 Pro（クローン）", [("audiocpp-fish-s2-pro", "", "")]),
    ("Higgs Audio TTS 3 4B（クローン）", [("audiocpp-higgs-audio-v3-4b", "", "")]),
    ("OuteTTS 1.0 1B（クローン）", [("audiocpp-outetts-1.0-1b", "", "")]),
]


# GPU では測っていない構成 → CPU の結果の名前（表の CPU の欄に入れる）
TTS_CPU_NAME = {"supertonic": "supertonic_t4", "moss_nano_gpu": "moss_nano"}
TTS_CPU_ONLY = {"supertonic"}


def load(*parts: str) -> dict | None:
    p = R.joinpath(*parts)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def runtime_label(name: str, label: str) -> str:
    if label:
        return label
    return RT.get(name.split("-", 1)[0], name)


def ms(d: dict | None, key: str = "mean_ms") -> str:
    return "—" if not d or d.get(key) is None else f"{d[key]:.0f}"


def cer(d: dict | None, key: str = "corpus_cer_kana") -> str:
    return "—" if not d or d.get(key) is None else f"{d[key]:.3f}"


def mem(d: dict | None, key: str) -> str:
    return "—" if not d or d.get(key) is None else f"{d[key] / 1024:.1f}"


def load_s(d: dict | None) -> str:
    return "—" if not d or d.get("load_s") is None else f"{d['load_s']:.1f}"


def asr_table() -> str:
    out = []
    for model, rows in ASR_GROUPS:
        out.append(f"### {model}\n")
        out.append("| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice "
                   "| 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for name, label, note in rows:
            g, c = load("timing_gpu", f"{name}.json"), load("timing_cpu", f"{name}.json")
            m = load("mem_gpu", f"{name}.json") or g
            base = g or c
            sets = [cer(load(ds, f"{name}.json")) for ds in ("fleurs-ja", "reazonspeech-test", "cv8-ja")]
            out.append(f"| {runtime_label(name, label)} | {ms(g)} | {ms(c)} | {cer(base)} | {' | '.join(sets)} | "
                       f"{load_s(m)} | {mem(m, 'gpu_mem_mb')} | {mem(m, 'ram_mb')} | {note} |")
        out.append("")
    return "\n".join(out)


def tts_table() -> str:
    out = []
    for model, rows in TTS_GROUPS:
        out.append(f"### {model}\n")
        out.append("| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for name, label, note in rows:
            d = None if name in TTS_CPU_ONLY else load("tts", f"{name}.json")
            c = load("tts", f"{TTS_CPU_NAME.get(name, name + '-cpu')}.json")
            if c is not None and d is None:  # CPU でだけ測ったものは、読みの CER も CPU の結果から
                d_cer = c
            else:
                d_cer = d
            m = load("mem_tts", f"{name}.json")
            rtf = "—" if not d or d.get("rtf") is None else f"{d['rtf']:.3f}"
            out.append(f"| {runtime_label(name, label)} | {ms(d, 'mean_synth_ms')} | {rtf} | {cer(d_cer)} | "
                       f"{ms(c, 'mean_synth_ms')} | {load_s(m)} | {mem(m, 'gpu_mem_mb')} | {mem(m, 'ram_mb')} | {note} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for tag, body in (("asr", asr_table()), ("tts", tts_table())):
        start, end = f"<!-- {tag}-tables:start -->", f"<!-- {tag}-tables:end -->"
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body}\n{end}{tail}"
    PAGE.write_text(text, encoding="utf-8")
    print("wrote", PAGE.relative_to(REPO))


if __name__ == "__main__":
    main()
