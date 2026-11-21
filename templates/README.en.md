# Awesome Japanese Spoken Dialogue Stack [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ASR, TTS, LLMs and the surrounding models for building Japanese spoken dialogue systems, with speed and accuracy measured per runtime on the same machine.

[日本語](README.md)

- **List**: Japanese support, streaming, license and runtimes for each model.
- **Runtime comparison**: the same model run with its original implementation, sherpa-onnx, CrispASR, audio.cpp and speech.cpp, measured for speed, accuracy and memory. The same model can be several times faster depending on the runtime.
- **Latency too**: time from the end of speech to the text, and time to the first audio, which matter in spoken dialogue.

## Latest updates

- **2026-11-21** Added ASR on free two-person conversation, streaming latency, TTS naturalness (UTMOS) and LLM runtimes. Fixed how numerals are scored and rescored every result. Checked every model license.
- **2026-11-18** Added a [page](benchmarks/2026-11-runtimes.md) comparing the same model across runtimes.
- **2026-11-17** Added CrispASR, audio.cpp and speech.cpp as runtimes; ASR latency measured separately on GPU and CPU.

[Full changelog](CHANGELOG.md)

## Contents

- [Quick picks](#quick-picks)
- [ASR](#asr)
- [TTS](#tts)
- [LLM](#llm)
- [Spoken dialogue (VAD, turn-taking, backchannels)](#spoken-dialogue-vad-turn-taking-backchannels)
- [Try it](#try-it)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Quick picks

First choices based on our measurements. Check the license column below before commercial use.

| Goal | Pick | Why |
|---|---|---|
| Fast, accurate ASR on GPU | Haas-ja v1.0, Parakeet TDT-CTC 0.6B ja | ~40 ms per utterance; Haas has the best average rank on our three test sets ([ASR accuracy](benchmarks/asr/2026-11-3sets.md)) |
| ASR on CPU only | ReazonSpeech k2-v2 | ~90 ms per utterance, CER 0.024-0.036 ([latency](benchmarks/asr/2026-11-timing.md)) |
| Casual conversation | Qwen3-ASR 1.7B (speech.cpp) | Best CER on free two-person conversation (0.091) ([conversation](benchmarks/asr/2026-11-3sets.md#会話magichub-の自由会話)) |
| Text while the user is still speaking | Pseudo-streaming Parakeet TDT-CTC or Haas-ja | Text appears ~60-130 ms after each phoneme; true-streaming Nemotron takes 0.5-0.8 s ([streaming](benchmarks/2026-11-runtimes.md#ストリーミング)) |
| Fast TTS | VOICEVOX (voicevox_core), Supertonic 3 (audio.cpp) | ~50 ms and ~90 ms per sentence on GPU ([TTS](benchmarks/tts/2026-11-tts.md)) |
| Voice cloning | Qwen3-TTS (audio.cpp, speech.cpp) | ~0.6 s per sentence, 32-92 ms to first audio when streaming ([runtimes](benchmarks/2026-11-runtimes.md)) |
| Voice from a text description | Irodori-TTS v4.1 (audio.cpp) | ~0.4 s per sentence ([runtimes](benchmarks/2026-11-runtimes.md)) |
| Fast first reply from the LLM | LFM2.5 1.2B JP, ELYZA-Shortcut 7B | 39 ms and 45 ms to the first token ([LLM](benchmarks/llm/2026-11-lmstudio.md)) |

**Reading the tables**: CER is the reading-based character error rate (both sides converted to katakana readings; lower is better). Times are on GPU unless noted, per utterance for ASR and per sentence for TTS. The benchmark pages are in Japanese.

## ASR

"Streaming" yes means text can appear while the user is speaking; pseudo means short segments are re-recognised repeatedly.

<!-- table:asr -->

### Not yet evaluated

<!-- table:asr_todo -->

## TTS

"Streaming" yes means audio can start before the whole sentence is synthesised.

<!-- table:tts -->

## LLM

For knowledge and reasoning in Japanese, see the [Nejumi LLM Leaderboard](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) and the [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html). Here we measure what matters in spoken dialogue: time to the first token and generation speed.

<!-- table:llm -->

## Spoken dialogue (VAD, turn-taking, backchannels)

<!-- table:dialog -->

## Try it

- [examples/asr/](examples/asr/) - Minimal per-model scripts that transcribe one WAV file.
- [apps/](apps/README.md) - An ASR app that recognises your microphone and a TTS app that reads text with a chosen engine and voice (desktop and browser versions), with install steps.

## Benchmarks

| What | Page |
|---|---|
| Speed, accuracy and memory of the same model across runtimes, streaming latency, TTS naturalness, LLM runtimes | [Runtime comparison](benchmarks/2026-11-runtimes.md) |
| ASR accuracy (FLEURS, ReazonSpeech, Common Voice, free conversation) | [ASR accuracy](benchmarks/asr/2026-11-3sets.md) |
| ASR latency (the same 100 utterances on GPU and CPU) | [ASR latency](benchmarks/asr/2026-11-timing.md) |
| TTS synthesis time and reading accuracy (20 sentences) | [TTS](benchmarks/tts/2026-11-tts.md) |
| LLM reply latency (LM Studio) | [LLM](benchmarks/llm/2026-11-lmstudio.md) |

Methods, the test machine and earlier results are in [benchmarks/README.md](benchmarks/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The tables are generated from `data/models.toml`; do not edit the README directly.

## License

This repository (the list, benchmark results and code) is [MIT](LICENSE). The listed models and datasets follow their own licenses.

## Footnotes

- Licenses were checked against each official page (model card, repository and license text) on 2026-11-21. They can change, so check the official page before use. The commercial column is a guide, not legal advice. "conditional" means commercial use is allowed under conditions such as a revenue cap; "unknown" means no license is stated.
- Related: [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) - A spoken dialogue system where many of these models are integrated and tested.
