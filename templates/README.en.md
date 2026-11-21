# Awesome Japanese Spoken Dialogue Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ASR, TTS, LLMs and the surrounding models for building Japanese spoken dialogue systems, with speed and accuracy measured per runtime on the same machine, and demo apps to try the models locally.

[日本語](README.md)

- **List**: Japanese support, streaming, license and runtimes for each model.
- **Runtime comparison**: the same model run with its original implementation, [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [CrispASR](https://github.com/CrispStrobe/CrispASR), [audio.cpp](https://github.com/0xShug0/audio.cpp) and [speech.cpp](https://github.com/nyosegawa/speech.cpp), measured for speed, accuracy and memory. The same model can be several times faster depending on the runtime.
- **Latency too**: time from the end of speech to the text, and time to the first audio, which matter in spoken dialogue.
- **Try them locally**: every measured configuration can be tried in an ASR app that recognises your microphone and a TTS app that reads text aloud, including the choice of runtime, with the latency shown as you go ([apps/](apps/README.md)).

## Latest updates

- **2026-11-22** Measured on CPU the models only measured on GPU and filled the remaining gaps (conversation and more). Added configurations the runtimes support but we had not set up (LFM2.5-Audio, Kitten and ZONOS3 on audio.cpp, Qwen3-TTS VoiceDesign, Moonshine on CrispASR and more).
- **2026-11-21** Added ASR on free two-person conversation, streaming latency, TTS naturalness (UTMOS) and LLM runtimes. Fixed how numerals are scored and rescored every result. Checked every model license.
- **2026-11-18** Added a [page](benchmarks/2026-11-runtimes.md) comparing the same model across runtimes.

[Full changelog](CHANGELOG.md)

## Contents

- [Quick picks](#quick-picks)
- [ASR](#asr)
- [TTS](#tts)
- [LLM](#llm)
- [Spoken dialogue (VAD, turn-taking, backchannels)](#spoken-dialogue-vad-turn-taking-backchannels)
- [Runtimes](#runtimes)
- [Try it](#try-it)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)
- [License](#license)

## Quick picks

First choices based on our measurements. Check the license column below before commercial use.

| Goal | Pick | Why |
|---|---|---|
| Fast, accurate ASR on GPU | Haas-ja v1.0, Parakeet TDT-CTC 0.6B ja | ~40 ms per utterance; on held-out FLEURS, Haas 0.025 and Parakeet 0.038 ([ASR accuracy](benchmarks/asr/2026-11-3sets.md)) |
| ASR on CPU only | ReazonSpeech k2-v2 | ~90 ms per utterance, CER 0.036 on FLEURS ([latency](benchmarks/asr/2026-11-timing.md)) |
| Casual conversation | Qwen3-ASR 1.7B (speech.cpp) | Best CER on free two-person conversation (0.091) ([conversation](benchmarks/asr/2026-11-3sets.md#会話magichub-の自由会話)) |
| Text while the user is still speaking | Pseudo-streaming Parakeet TDT-CTC or Haas-ja | Text appears ~60-130 ms after each phoneme; true-streaming Nemotron takes 0.5-0.8 s ([streaming](benchmarks/2026-11-runtimes.md#ストリーミング)) |
| Fast TTS | VOICEVOX (voicevox_core), Supertonic 3 (audio.cpp) | ~50 ms and ~90 ms per sentence on GPU ([TTS](benchmarks/tts/2026-11-tts.md)) |
| Voice cloning | Qwen3-TTS (audio.cpp, speech.cpp) | ~0.6 s per sentence, 32-92 ms to first audio when streaming ([runtimes](benchmarks/2026-11-runtimes.md)) |
| Voice from a text description | Qwen3-TTS 1.7B VoiceDesign (audio.cpp), Irodori-TTS (audio.cpp) | Qwen3-TTS ~0.7 s per sentence with 25 ms to first audio; Irodori ~0.4 s per sentence ([runtimes](benchmarks/2026-11-runtimes.md)) |
| Fast first reply from the LLM | LFM2.5 1.2B JP, ELYZA-Shortcut 7B | 39 ms and 45 ms to the first token ([LLM](benchmarks/llm/2026-11-lmstudio.md)) |

**Reading the tables**: CER is the reading-based character error rate (both sides converted to katakana readings; lower is better). Times are on GPU unless noted, per utterance for ASR and per sentence for TTS. The numbers come from the configuration shown as "values:" in the notes; "—" means not measured. The benchmark pages are in Japanese.

## ASR

The FLEURS, ReazonSpeech, CV and Conversation columns are the reading-based CER on each test set (read speech, TV broadcasts, crowd-sourced read speech and free two-person conversation; [details](benchmarks/asr/2026-11-3sets.md)). **†** means the model card lists that test set (or data from the same source) in training, so the score is favourable. Bold marks the best value without †. Conversation segments are padded with 0.5 s of silence on both ends, which helps some models and hurts others. "Streaming" yes means text can appear while the user is speaking; pseudo means short segments are re-recognised repeatedly.

<!-- table:asr -->

### Not yet evaluated

<!-- table:asr_todo -->

## TTS

ms/sentence is the mean synthesis time over 20 sentences; Reading CER is the median over three ASR transcriptions of the audio; UTMOS estimates naturalness (1-5; trained on English, so treat it as a guide; it reflects the chosen voice more than the model, and cloning models all use the same reference, whose own recording scores 3.52); First audio ms is the median time to the first audio when streaming ([details](benchmarks/2026-11-runtimes.md#tts)). Bold marks the best value in each column. "Streaming" yes means audio can start before the whole sentence is synthesised.

<!-- table:tts -->

## LLM

For knowledge and reasoning in Japanese, see the [Nejumi LLM Leaderboard](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) and the [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html). Here we measure what matters in spoken dialogue: time to the first token and generation speed.

<!-- table:llm -->

## Spoken dialogue (VAD, turn-taking, backchannels)

<!-- table:dialog -->

## Runtimes

The runtimes compared in the benchmarks. The same model can differ a lot in speed and memory depending on the runtime ([runtime comparison](benchmarks/2026-11-runtimes.md)).

| Runtime | What it is | For |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Speech recognition and synthesis on ONNX Runtime, with many fast int8 builds for CPU | ASR, TTS, VAD |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Fast Whisper-family inference on CTranslate2 | ASR |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | C++ runtime and server running many ASR and TTS models on ggml | ASR, TTS |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | C++ server running ASR and TTS on ggml, with streaming | ASR, TTS |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | C++ runtime for ASR and TTS on ggml; uses Vulkan for the GPU on Windows | ASR, TTS |
| [llama.cpp](https://github.com/ggml-org/llama.cpp), [LM Studio](https://lmstudio.ai) | Run GGUF LLMs | LLM |

Setup steps are in [docs/runtimes.md](docs/runtimes.md).

## Try it

- [examples/asr/](examples/asr/) - Minimal per-model scripts that transcribe one WAV file.
- [apps/](apps/README.md) - An ASR app that recognises your microphone and a TTS app that reads text with a chosen engine and voice (desktop and browser versions). Every benchmarked configuration, including each runtime, can be selected. Install steps are there too.

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
