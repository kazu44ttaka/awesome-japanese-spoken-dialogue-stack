# awesome-japanese-speech

A curated list of **Japanese speech recognition (ASR), speech synthesis (TTS) and LLMs**, plus the models around spoken dialogue (voice activity detection, turn-taking, backchannels).

[日本語](README.md)

- Focused on models you can run locally, with **Japanese support, streaming, license (commercial use) and runtime** side by side.
- Some models have **numbers measured on the same machine under the same conditions** (character error rate, latency, RTF) together with the scripts, in [benchmarks/](benchmarks/).
- Written with spoken dialogue systems in mind, so **latency matters as much as accuracy**.

> Always check the official license of each model. "TBD" means we have not verified it yet.
> The "Commercial" column is a hint, not legal advice.

## Contents

- [ASR](#asr)
- [TTS](#tts)
- [LLM](#llm)
- [Spoken dialogue (VAD, turn-taking, backchannels)](#spoken-dialogue-vad-turn-taking-backchannels)
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

## Benchmarks

| Target | What | Results |
|---|---|---|
| ASR | Character error rate (CER) and per-utterance latency on 20 JSUT utterances | [2026-08](benchmarks/asr/2026-08-jsut20.md), [2026-11 (new models, CER without punctuation)](benchmarks/asr/2026-11-jsut20.md) (Japanese) |
| TTS | Long-text RTF (synthesis time / audio length), memory, ASR round-trip check | [benchmarks/tts/2026-08-19.md](benchmarks/tts/2026-08-19.md) (Japanese) |
| LLM | Time to first token and tokens/s for dialogue replies (LM Studio, 10 models) | [2026-11](benchmarks/llm/2026-11-lmstudio.md) (Japanese) |

Method and test machine: [benchmarks/README.md](benchmarks/README.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The tables are generated from `data/models.toml`; do not edit the README directly.

## Related

- [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) — a spoken dialogue system where many of these models are integrated and tested
