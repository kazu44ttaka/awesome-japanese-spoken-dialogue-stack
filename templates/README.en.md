# Awesome Japanese Spoken Dialogue Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ASR, TTS, LLMs and related models for Japanese spoken dialogue systems. [日本語](README.md)

- Speed, accuracy and memory of the same model across runtimes, all measured on one PC ([runtime comparison](benchmarks/2026-11-runtimes.md))
- Latency that matters in dialogue: end of speech to text, time to first audio
- Apps to try every measured configuration with your microphone or speakers ([apps/](apps/README.md))

**Updates**: 2026-11-22 CPU measurements and missing configurations / 2026-11-21 conversational ASR, streaming latency, UTMOS, LLM runtimes ([changelog](CHANGELOG.md))

[Picks](#quick-picks) · [ASR](#asr) · [TTS](#tts) · [LLM](#llm) · [Dialogue](#spoken-dialogue) · [Runtimes](#runtimes) · [Try it](#try-it) · [Benchmarks](#benchmarks)

## Quick picks

| Goal | Pick | Numbers |
|---|---|---|
| Fast, accurate ASR on GPU | Haas-ja v1.0, Parakeet TDT-CTC 0.6B ja | ~40 ms per utterance, FLEURS CER 0.025 / 0.038 |
| ASR on CPU only | ReazonSpeech k2-v2 | ~90 ms per utterance, CER 0.036 |
| Casual conversation | Qwen3-ASR 1.7B | Conversation CER 0.089-0.091 |
| Text while the user speaks | Pseudo-streaming Parakeet TDT-CTC or Haas-ja | ~60-130 ms from phoneme to text |
| Fast TTS | VOICEVOX (voicevox_core), Supertonic 3 (audio.cpp) | ~50 ms / ~90 ms per sentence |
| Voice cloning | Qwen3-TTS (audio.cpp, speech.cpp) | ~0.6 s per sentence, 32-92 ms to first audio |
| Voice from a description | Qwen3-TTS 1.7B VoiceDesign (audio.cpp) | ~0.7 s per sentence, 25 ms to first audio |
| Fast first LLM reply | LFM2.5 1.2B JP, ELYZA-Shortcut 7B | 39 ms / 45 ms to first token |

Times are on GPU. CER is computed on katakana readings; lower is better. "—" means not measured. Developer, license and the measured configuration are in [model details](docs/models.en.md). Benchmark pages are in Japanese.

## ASR

CER on four test sets: read speech, TV, crowd-sourced read speech and free two-person conversation ([details](benchmarks/asr/2026-11-3sets.md)). **†** = trained on that test set's source (favourable); bold = best without †. Pseudo streaming re-recognises short segments.

<!-- table:asr -->

### Not yet evaluated

<!-- table:asr_todo -->

## TTS

ms/sentence is the mean over 20 sentences; Reading CER is the ASR error on the synthesised audio; UTMOS estimates naturalness (1-5, a rough guide); First audio ms is the time to the first audio when streaming ([details](benchmarks/tts/2026-11-tts.md)).

<!-- table:tts -->

## LLM

For knowledge and reasoning see the [Nejumi](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) and [Swallow](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) leaderboards. Here we measure time to first token and generation speed.

<!-- table:llm -->

## Spoken dialogue

Voice activity detection, turn-taking, backchannels and more.

<!-- table:dialog -->

## Runtimes

| Runtime | What it is | For |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | ONNX Runtime; many fast int8 builds for CPU | ASR, TTS, VAD |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Fast Whisper on CTranslate2 | ASR |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | C++ runtime and server on ggml | ASR, TTS |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | C++ server on ggml, with streaming | ASR, TTS |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | C++ runtime on ggml; Vulkan GPU on Windows | ASR, TTS |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), [LM Studio](https://lmstudio.ai) | GGUF LLMs | LLM |

Setup: [docs/runtimes.md](docs/runtimes.md).

## Try it

- [apps/](apps/README.md): an ASR app for your microphone and a TTS app (desktop and browser). Every measured configuration can be selected
- [examples/asr/](examples/asr/): minimal per-model scripts that transcribe one WAV file

## Benchmarks

- [Runtime comparison](benchmarks/2026-11-runtimes.md): speed, accuracy, memory, streaming latency, TTS naturalness, LLM runtimes
- [ASR accuracy](benchmarks/asr/2026-11-3sets.md) · [ASR latency](benchmarks/asr/2026-11-timing.md) · [TTS](benchmarks/tts/2026-11-tts.md) · [LLM](benchmarks/llm/2026-11-lmstudio.md)
- Methods and test machine: [benchmarks/README.md](benchmarks/README.md)

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) (tables are generated from `data/models.toml`; do not edit the README). This repository is [MIT](LICENSE); listed models and datasets follow their own licenses. Licenses were checked on the official pages on 2026-11-21; the commercial column is a guide, not legal advice.

Related: [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS), a spoken dialogue system integrating many of these models.
