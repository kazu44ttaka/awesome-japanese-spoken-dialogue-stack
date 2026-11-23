# Awesome Japanese Spoken Dialogue Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

日本語の音声対話システムに使える ASR・TTS・LLM などの一覧です。[English](README.en.md)

- 同じ PC で、同じモデルを実行系ごとに測った速さ・精度・メモリの比較（[実行系の比較](benchmarks/2026-11-runtimes.md)）
- 話し終わってから文字が出るまで、最初の声が出るまでなど、対話で効く遅延も測定
- 測った構成をマイクや読み上げでその場で試せるアプリ（[apps/](apps/README.md)）

**更新**：2026-11-22 CPU での測定と、空いていた構成を追加 ／ 2026-11-21 会話の ASR・ストリーミングの遅れ・UTMOS・LLM の実行系を追加（[履歴](CHANGELOG.md)）

[おすすめ](#目的別のおすすめ) ・ [ASR](#asr音声認識) ・ [TTS](#tts音声合成) ・ [LLM](#llm) ・ [対話の周辺](#音声対話の周辺) ・ [実行系](#実行系) ・ [試す](#動かして試す) ・ [ベンチマーク](#ベンチマーク)

## 目的別のおすすめ

| やりたいこと | 候補 | 目安 |
|---|---|---|
| GPU で速く正確に認識 | Haas-ja v1.0、Parakeet TDT-CTC 0.6B ja | 1 発話 約 40 ms、FLEURS の CER 0.025・0.038 |
| CPU だけで認識 | ReazonSpeech k2-v2 | 1 発話 約 90 ms、CER 0.036 |
| くだけた会話を認識 | Qwen3-ASR 1.7B | 会話の CER 0.089〜0.091 |
| 話している途中から文字を出す | Parakeet TDT-CTC・Haas-ja の疑似ストリーミング | 音素から文字まで 約 60〜130 ms |
| 速く読み上げ | VOICEVOX（voicevox_core）、Supertonic 3（audio.cpp） | 1 文 約 50 ms・約 90 ms |
| 声を真似て読み上げ | Qwen3-TTS（audio.cpp・speech.cpp） | 1 文 約 0.6 秒、最初の声まで 32〜92 ms |
| 声を文章で指定 | Qwen3-TTS 1.7B VoiceDesign（audio.cpp） | 1 文 約 0.7 秒、最初の声まで 25 ms |
| LLM の返事を速く始める | LFM2.5 1.2B JP、ELYZA-Shortcut 7B | 最初の文字まで 39 ms・45 ms |

時間は GPU。CER は読み（カタカナ）に直した文字誤り率で、小さいほどよい。「—」は未測定。開発元・ライセンス・値を測った構成は [モデルの詳細](docs/models.md) に。

## ASR（音声認識）

列は 4 つの評価データでの CER（読み上げ・テレビ・マイクの読み上げ・2 人の自由会話。[詳しく](benchmarks/asr/2026-11-3sets.md)）。**†** はその評価データで学習していて有利なもの、太字は † を除いた最良。ストリーミングの △ は短く区切って認識し直す疑似ストリーミング。

<!-- table:asr -->

### 未評価の候補

<!-- table:asr_todo -->

## TTS（音声合成）

1 文 ms は 20 文の平均、読みの CER は合成音声を ASR で書き起こした誤り、UTMOS は自然さの目安（1〜5）、最初の声 ms はストリーミングで最初の音が出るまで（[詳しく](benchmarks/tts/2026-11-tts.md)）。

<!-- table:tts -->

## LLM

知識や推論は [Nejumi](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) や [Swallow](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) のリーダーボードを。ここでは最初の文字までの時間と生成の速さを測っています。

<!-- table:llm -->

## 音声対話の周辺

発話区間検出・ターンテイキング・相槌など。

<!-- table:dialog -->

## 実行系

| 実行系 | 内容 | 対象 |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | ONNX Runtime。CPU で速い int8 の配布物が多い | ASR・TTS・VAD |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 で Whisper 系を速く | ASR |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | ggml の C++ 実行系とサーバー | ASR・TTS |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | ggml の C++ サーバー。ストリーミング対応 | ASR・TTS |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | ggml の C++ 実行系。Windows の GPU は Vulkan | ASR・TTS |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)・[LM Studio](https://lmstudio.ai) | GGUF の LLM | LLM |

入れ方は [docs/runtimes.md](docs/runtimes.md)。

## 動かして試す

- [apps/](apps/README.md)：マイクで話して認識する ASR アプリと、読み上げる TTS アプリ（デスクトップ版・ブラウザ版）。測った構成を全部選べます
- [examples/asr/](examples/asr/)：WAV を 1 つ書き起こす、モデルごとの最小の例

## ベンチマーク

- [実行系の比較](benchmarks/2026-11-runtimes.md)：速さ・精度・メモリ、ストリーミングの遅れ、TTS の自然さ、LLM の実行系
- [ASR の精度](benchmarks/asr/2026-11-3sets.md) ・ [ASR の処理時間](benchmarks/asr/2026-11-timing.md) ・ [TTS](benchmarks/tts/2026-11-tts.md) ・ [LLM](benchmarks/llm/2026-11-lmstudio.md)
- 測り方と測定環境：[benchmarks/README.md](benchmarks/README.md)

## 追加・修正・ライセンス

追加や修正は [CONTRIBUTING.md](CONTRIBUTING.md) へ（表は `data/models.toml` から作るので README は直接編集しない）。このリポジトリは [MIT](LICENSE)、載せたモデルとデータはそれぞれのライセンスに従います。ライセンスは 2026-11-21 に公式ページで確かめたもので、商用の欄は目安です（法的な判断ではありません）。

関連：[MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS)（ここのモデルの多くを組み込んだ音声対話システム）
