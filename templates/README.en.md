# awesome-japanese-speech

A curated list of **Japanese speech recognition (ASR), speech synthesis (TTS) and LLMs**, plus the models around spoken dialogue (voice activity detection, turn-taking, backchannels).

[日本語](README.md)

- Focused on models you can run locally, with **Japanese support, streaming, license (commercial use) and runtime** side by side.
- **Compare the same model across runtimes.** The original implementation (PyTorch etc.), sherpa-onnx, faster-whisper and the ggml-based CrispASR, audio.cpp and speech.cpp,
  measured for speed, accuracy and memory on the same machine ([runtime comparison](benchmarks/2026-11-runtimes.md), Japanese). The same model can be several times faster, or less accurate, depending on the runtime.
- **Broad coverage**: most of the main ASR and TTS models usable for Japanese have **numbers measured on the same machine under the same conditions** (character error rate, latency, RTF, memory) together with the scripts, in [benchmarks/](benchmarks/).
- Written with spoken dialogue systems in mind, so **latency matters as much as accuracy**.

> Licenses were checked against each official page (model card, repository and license text) on 2026-11-21. They can change, so always check the official page before use. "conditional" means commercial use is allowed under conditions such as a revenue cap; "unknown" means no license is stated.
> The "Commercial" column is a hint, not legal advice.

## Contents

- [ASR](#asr)
- [TTS](#tts)
- [LLM](#llm)
- [Spoken dialogue (VAD, turn-taking, backchannels)](#spoken-dialogue-vad-turn-taking-backchannels)
- [Try it](#try-it)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)

## ASR

"Streaming: yes" means text appears while the user is still speaking; "pseudo" means short chunks are re-recognized each time.

<!-- table:asr -->

### Not yet evaluated

<!-- table:asr_todo -->

## TTS

"Streaming: yes" means audio can start before the whole sentence is synthesized.

<!-- table:tts -->

## LLM

For Japanese knowledge and reasoning, see the existing leaderboards:

- [Nejumi LLM Leaderboard](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA)
- [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html)

We measure what matters for dialogue — **time to first token and tokens per second** — on a local GPU ([benchmarks/llm/2026-11-lmstudio.md](benchmarks/llm/2026-11-lmstudio.md)).

<!-- table:llm -->

## Spoken dialogue (VAD, turn-taking, backchannels)

<!-- table:dialog -->

## Try it

- **Minimal inference examples**: [examples/asr/](examples/asr/) — one standalone script per model that transcribes a WAV file
- **Microphone demo app**: [apps/asr_demo.py](apps/asr_demo.py) (PySide6) and [apps/asr_web.py](apps/asr_web.py) (browser) — pick a model and speak; utterances are segmented with Silero VAD and finalized, with optional interim results (pseudo-streaming). Shows recognition time and the delay from the end of speech to the text. Nemotron 3.5 streaming runs as true streaming (incremental input with carried-over state)
- **TTS demo app**: [apps/tts_demo.py](apps/tts_demo.py) (PySide6) and [apps/tts_web.py](apps/tts_web.py) (browser) — pick an engine (AivisSpeech, VOICEVOX, VOICEVOX Core, Irodori-TTS, Supertonic 3, mio-tts-cpp, Qwen3-TTS, Tsukasa, Kokoro, ZONOS3-Mini, Kitten TTS 2, LFM2.5-Audio, MOSS-TTS-Nano) and a voice and read text aloud. "Synthesize per sentence and play as soon as each is ready" shows the time to first audio, which is what matters in dialogue. AivisSpeech / VOICEVOX can be launched from the app if installed. Setup: [apps/README.md](apps/README.md)

```
uv venv --python 3.11 .venv
uv pip install sherpa-onnx sounddevice soundfile numpy
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps "silero-vad==6.2.1"               # voice activity detection (same as MaAI-SDS)
uv pip install "transformers>=5.16" funasr                 # for transformers / funasr models
uv pip install "pyside6-essentials==6.12.0" fastapi "uvicorn[standard]"   # desktop UI (PySide6) and web version
python apps/asr_demo.py      # web version: python apps/asr_web.py (http://127.0.0.1:8765)
.venv-tts/Scripts/python apps/tts_demo.py   # TTS uses its own environment (apps/README.md); web version: apps/tts_web.py
```

## Benchmarks

| Target | What | Results |
|---|---|---|
| ASR, TTS | The same model on different runtimes (original implementation, sherpa-onnx, CrispASR, audio.cpp, speech.cpp, ...): speed, accuracy and memory (18 ASR and 22 TTS models) | [2026-11 (runtimes)](benchmarks/2026-11-runtimes.md) (Japanese) |
| ASR | Reading-based CER on three public test sets (FLEURS, ReazonSpeech, Common Voice; 23 configurations) | [2026-11 (three test sets)](benchmarks/asr/2026-11-3sets.md) (Japanese) |
| ASR | Per-utterance latency on the same 100 utterances, measured separately on GPU and CPU (8 threads) | [2026-11 (latency)](benchmarks/asr/2026-11-timing.md) (Japanese) |
| ASR | Character error rate (CER) and per-utterance latency on 20 JSUT utterances | [2026-08](benchmarks/asr/2026-08-jsut20.md), [2026-11 (new models, CER without punctuation)](benchmarks/asr/2026-11-jsut20.md) (Japanese) |
| TTS | Synthesis time and RTF on 20 short-to-long sentences, plus reading-based CER from three ASR models (9 engines in 18 configurations, GPU and CPU separately) | [2026-11](benchmarks/tts/2026-11-tts.md) (Japanese) |
| TTS | Long-text RTF (synthesis time / audio length), memory, ASR round-trip check | [2026-08](benchmarks/tts/2026-08-19.md) (Japanese) |
| LLM | Time to first token and tokens/s for dialogue replies (LM Studio, 10 models) | [2026-11](benchmarks/llm/2026-11-lmstudio.md) (Japanese) |

Method and test machine: [benchmarks/README.md](benchmarks/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The tables are generated from `data/models.toml`; do not edit the README directly.

## License

This repository (the list, benchmark results and code) is MIT-licensed ([LICENSE](LICENSE)). The listed models and datasets follow their own licenses.

## Related

- [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) — a spoken dialogue system where many of these models are integrated and tested
