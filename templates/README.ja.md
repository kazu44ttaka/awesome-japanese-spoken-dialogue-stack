# Awesome Japanese Spoken Dialogue Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

日本語の音声対話システムを作るための ASR・TTS・LLM と周辺モデルの一覧です。同じ PC で実行系ごとに測った速さと精度の比較と、手元でモデルを試せるデモアプリもあります。

[English](README.en.md)

- **一覧**：日本語への対応・ストリーミング・ライセンス・動かし方を、モデルごとに並べています。
- **実行系の比較**：同じモデルを元の実装・[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)・[CrispASR](https://github.com/CrispStrobe/CrispASR)・[audio.cpp](https://github.com/0xShug0/audio.cpp)・[speech.cpp](https://github.com/nyosegawa/speech.cpp) で動かし、速さ・精度・メモリを測りました。同じモデルでも実行系で数倍違います。
- **遅延も測る**：話し終わってから文字が出るまで、最初の声が出るまでなど、音声対話で効く時間も測っています。
- **手元で試せる**：測った構成を、マイクで話して認識する ASR アプリと、文を読み上げる TTS アプリで、その場で試せます。実行系の違いも選べ、話し終わってから文字が出るまでや最初の音が出るまでの時間も表示します（[apps/](apps/README.md)）。

## 最近の更新

- **2026-11-21** 2 人の自由会話での ASR の比較、ストリーミングの遅れ、TTS の自然さ（UTMOS）、LLM の実行系の比較を追加。数字の採点の誤りを直して、全結果を採点し直した。全モデルのライセンスを確かめた
- **2026-11-18** 同じモデルを実行系ごとに比べる [ページ](benchmarks/2026-11-runtimes.md) を追加
- **2026-11-17** CrispASR・audio.cpp・speech.cpp を比べる実行系に追加。ASR の処理時間を GPU と CPU に分けて測った

[すべての更新履歴](CHANGELOG.md)

## 目次

- [目的別のおすすめ](#目的別のおすすめ)
- [ASR（音声認識）](#asr音声認識)
- [TTS（音声合成）](#tts音声合成)
- [LLM](#llm)
- [音声対話の周辺（発話区間検出・ターンテイキング・相槌）](#音声対話の周辺発話区間検出ターンテイキング相槌)
- [実行系（推論のライブラリ）](#実行系推論のライブラリ)
- [動かして試す](#動かして試す)
- [ベンチマーク](#ベンチマーク)
- [追加・修正のしかた](#追加修正のしかた)
- [ライセンス](#ライセンス)

## 目的別のおすすめ

測った結果からの第一候補です。商用で使えるかは、下の表のライセンス欄で確かめてください。

| やりたいこと | 候補 | 根拠 |
|---|---|---|
| GPU で速く正確に認識する | Haas-ja v1.0、Parakeet TDT-CTC 0.6B ja | 1 発話 約 40 ms。学習に入っていない FLEURS で Haas 0.025・Parakeet 0.038（[ASR の精度](benchmarks/asr/2026-11-3sets.md)） |
| CPU だけで認識する | ReazonSpeech k2-v2 | 1 発話 約 90 ms・FLEURS で CER 0.036（[処理時間](benchmarks/asr/2026-11-timing.md)） |
| くだけた会話を認識する | Qwen3-ASR 1.7B（speech.cpp） | 2 人の自由会話で CER 0.091 と最もよい（[会話](benchmarks/asr/2026-11-3sets.md#会話magichub-の自由会話)） |
| 話している途中から文字を出す | Parakeet TDT-CTC・Haas-ja の疑似ストリーミング | 音素を話してから文字が出るまで 約 60〜130 ms。真のストリーミングの Nemotron は 0.5〜0.8 秒（[ストリーミング](benchmarks/2026-11-runtimes.md#ストリーミング)） |
| 速く読み上げる | VOICEVOX（voicevox_core）、Supertonic 3（audio.cpp） | GPU で 1 文 約 50 ms・約 90 ms（[TTS](benchmarks/tts/2026-11-tts.md)） |
| 声を真似て読み上げる | Qwen3-TTS（audio.cpp・speech.cpp） | 1 文 約 0.6 秒、ストリーミングで最初の声まで 32〜92 ms（[実行系の比較](benchmarks/2026-11-runtimes.md)） |
| 声を文章で指定する | Irodori-TTS v4.1（audio.cpp） | 1 文 約 0.4 秒（[実行系の比較](benchmarks/2026-11-runtimes.md)） |
| 返事を速く始める | LFM2.5 1.2B JP、ELYZA-Shortcut 7B | 最初の文字まで 39 ms・45 ms（[LLM](benchmarks/llm/2026-11-lmstudio.md)） |

**表の見方**：CER は読みの CER（カタカナの読みに直して比べた文字誤り率。小さいほどよい）。時間は特に書かないかぎり GPU で、ASR は 1 発話、TTS は 1 文あたり。表の数字は、メモの「値：」に書いた構成で測ったもの。「—」は測っていないもの。

## ASR（音声認識）

FLEURS・ReazonSpeech・CV・会話の列は、それぞれの評価データでの読みの CER です（読み上げ・テレビ放送・マイクの読み上げ・2 人の自由会話。[詳しく](benchmarks/asr/2026-11-3sets.md)）。**†** はモデルカードによるとその評価データ（と同じ出どころのデータ）で学習しているもので、有利に出ます。太字は † の無い値の中でいちばんよい値です。会話の区間は前後に 0.5 秒の無音を足していて、この足し方で値がよくなるモデルと悪くなるモデルがあります。ストリーミングの ○ は話している途中から文字を出せるもの、△ は短く区切って認識し直すもの（疑似ストリーミング）です。

<!-- table:asr -->

### 未評価の候補

<!-- table:asr_todo -->

## TTS（音声合成）

1 文 ms は 20 文の合成時間の平均、読みの CER は合成した音声を 3 つの ASR で書き起こした中央値、UTMOS は自然さの推定（1〜5。英語で学習した推定器なので目安。モデルより選んだ声の差が大きく出る。クローンするモデルは同じ参照音声を使い、その元の録音は 3.52）、最初の声 ms はストリーミングで最初の声が届くまでの中央値です（[詳しく](benchmarks/2026-11-runtimes.md#tts)）。太字は各列でいちばんよい値です。ストリーミングの ○ は、文を作り終える前に音を出し始められるものです。

<!-- table:tts -->

## LLM

日本語の知識や推論の力は [Nejumi LLM リーダーボード](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) や [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) を見てください。ここでは音声対話で効く、最初の文字までの時間と文字の出る速さを測っています。

<!-- table:llm -->

## 音声対話の周辺（発話区間検出・ターンテイキング・相槌）

<!-- table:dialog -->

## 実行系（推論のライブラリ）

ベンチマークで比べた実行系です。同じモデルでも、どれで動かすかで速さやメモリが大きく変わります（[実行系の比較](benchmarks/2026-11-runtimes.md)）。

| 実行系 | 内容 | 対象 |
|---|---|---|
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | ONNX Runtime で動く音声認識・合成の実行系。CPU で速い int8 の配布物が多い | ASR・TTS・VAD |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 で Whisper 系を速く動かす | ASR |
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | ggml で多くの ASR・TTS を動かす C++ の実行系とサーバー | ASR・TTS |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | ggml で ASR・TTS を動かす C++ のサーバー。ストリーミングにも対応 | ASR・TTS |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | ggml で ASR・TTS を動かす C++ の実行系。Windows の GPU は Vulkan | ASR・TTS |
| [llama.cpp](https://github.com/ggml-org/llama.cpp)・[LM Studio](https://lmstudio.ai) | GGUF の LLM を動かす | LLM |

入れ方は [docs/runtimes.md](docs/runtimes.md) にあります。

## 動かして試す

- [examples/asr/](examples/asr/) - WAV を 1 つ渡して書き起こすだけの、モデルごとの最小の例。
- [apps/](apps/README.md) - マイクで話して認識する ASR アプリと、エンジンと声を選んで読み上げる TTS アプリ（デスクトップ版とブラウザ版）。ベンチマークで測った構成（実行系の違いを含む）を全部選べます。入れ方もここにあります。

## ベンチマーク

| 内容 | ページ |
|---|---|
| 同じモデルを実行系ごとに比べた速さ・精度・メモリ、ストリーミングの遅れ、TTS の自然さ、LLM の実行系 | [実行系の比較](benchmarks/2026-11-runtimes.md) |
| ASR の精度（FLEURS・ReazonSpeech・Common Voice・2 人の自由会話） | [ASR の精度](benchmarks/asr/2026-11-3sets.md) |
| ASR の処理時間（同じ 100 発話を GPU と CPU で） | [ASR の処理時間](benchmarks/asr/2026-11-timing.md) |
| TTS の合成時間と読みの正確さ（20 文） | [TTS](benchmarks/tts/2026-11-tts.md) |
| LLM の応答の速さ（LM Studio） | [LLM](benchmarks/llm/2026-11-lmstudio.md) |

測り方・測定環境・前の結果は [benchmarks/README.md](benchmarks/README.md) にあります。

## 追加・修正のしかた

[CONTRIBUTING.md](CONTRIBUTING.md) を見てください。表は `data/models.toml` から作っているので、README は直接編集しません。

## ライセンス

このリポジトリ（一覧・測定結果・コード）は [MIT](LICENSE) です。載せたモデルとデータセットは、それぞれのライセンスに従います。

## 注

- ライセンスは 2026-11-21 に各モデルの公式ページ（モデルカード・リポジトリ・ライセンスの本文）で確かめました。変わることがあるので、使う前に公式ページを見てください。商用の欄は目安で、法的な判断ではありません。「条件つき」は年商などの条件を満たせば使えるもの、「不明」はライセンスの記載が無いものです。
- 関連：[MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) - ここに載せたモデルの多くを組み込んで試している音声対話システム。
