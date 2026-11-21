<!-- このファイルは scripts/build_readme.py が templates/ と data/models.toml から作る。直接編集しない。 -->
# Awesome Japanese Spoken Dialogue Stack

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

日本語の音声対話システムを作るための ASR・TTS・LLM と周辺モデルの一覧です。同じ PC で実行系ごとに測った、速さと精度の比較もあります。

[English](README.en.md)

- **一覧**：日本語への対応・ストリーミング・ライセンス・動かし方を、モデルごとに並べています。
- **実行系の比較**：同じモデルを元の実装・sherpa-onnx・CrispASR・audio.cpp・speech.cpp で動かし、速さ・精度・メモリを測りました。同じモデルでも実行系で数倍違います。
- **遅延も測る**：話し終わってから文字が出るまで、最初の声が出るまでなど、音声対話で効く時間も測っています。

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

| モデル | 開発元 | 日本語 | ストリーミング | FLEURS | ReazonSpeech | CV | 会話 | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|
| [Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) | Alibaba (FunAudioLLM) | 多言語 | — | 0.033 | 0.125 | 0.034 | 0.176 | PyTorch, sherpa-onnx, CrispASR, audio.cpp | 約 0.8B（音声エンコーダ + Qwen3-0.6B デコーダ） | Apache-2.0 | 可 | CrispASR（GPU）で 1 発話 約 130 ms と PyTorch の 5 倍速い。会話の値は相づち 3 区間の繰り返しの暴走を含む（除くと 0.093）。sherpa-onnx 版は話し言葉で崩れる（値：funasr） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [SenseVoiceSmall](https://github.com/QwenAudio/SenseVoice) | Alibaba (FunAudioLLM) | 多言語 | △（区切って再推論） | 0.049 | 0.183 | 0.044 | 0.126 | FunASR, ONNX, sherpa-onnx, CrispASR, audio.cpp | 約 234M | コード MIT・モデル FunASR Model License v1.1（出典の表示が必要） | 可 | CPU 約 90 ms・GPU 約 40 ms と速いが、放送の話し言葉で崩れる。sherpa-onnx は 2024-07-17 版を使う（値：sherpa-onnx） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Parakeet TDT-CTC 0.6B ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | NVIDIA | 専用 | — | 0.038 | 0.092† | 0.023 | 0.122 | NeMo, sherpa-onnx, speech.cpp, CrispASR | 0.6B | CC-BY-4.0 | 可 | GPU 約 40〜55 ms・CPU 約 180 ms。疑似ストリーミングで文字が出るまで 約 70 ms と最も速い（値：NeMo） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Cohere Transcribe (03-2026)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Cohere | 多言語 | — | 0.023 | 0.048 | **0.017** | 0.140 | PyTorch, sherpa-onnx, audio.cpp, CrispASR | Q4_K 約 1.5 GB 〜 Q8_0 約 2.4 GB | Apache-2.0（取得に同意が必要） | 可 | どの評価データでも上位。CPU 約 0.8 秒・GPU（audio.cpp）約 100 ms。14 言語（値：sherpa-onnx） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Cohere ASR ja v0.1](https://huggingface.co/efwkjn/cohere-asr-ja-v0.1) | efwkjn（Cohere Transcribe の日本語版） | 専用 | — | 0.025 | 0.092 | 0.018 | 0.127 | CrispASR (GGUF) | 2B（Q8_0 約 2.4 GB） | 記載なし（元の Cohere Transcribe は Apache-2.0） | 不明 | Cohere Transcribe の日本語の追加学習版。放送の話し言葉では元より悪い。GPU 約 190 ms（値：CrispASR） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Qwen3-ASR (1.7B / 0.6B)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Alibaba (Qwen) | 多言語 | — | 0.029 | 0.182 | 0.036 | **0.091** | PyTorch, sherpa-onnx, speech.cpp, audio.cpp, CrispASR | 1.7B / 0.6B | Apache-2.0 | 可 | 会話で最もよいが、放送の話し言葉で崩れる。speech.cpp（GPU）で 1.7B が 約 165 ms（PyTorch の 5 倍以上速い）。0.6B は会話 0.105。sherpa-onnx では言語の指定が必要（値：speech.cpp・1.7B） （[測定](benchmarks/2026-11-runtimes.md)） |
| [ReazonSpeech NeMo v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | Reazon Holdings | 専用 | — | 0.038 | 0.039† | 0.033 | 0.160 | NVIDIA NeMo, speech.cpp, CrispASR | 619M | Apache-2.0 | 可 | 放送の話し言葉に強い。speech.cpp で 1 発話 約 100 ms（NeMo は 約 350 ms）（値：NeMo） （[測定](benchmarks/2026-11-runtimes.md)） |
| [ReazonSpeech k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | Reazon Holdings | 専用 | — | 0.036 | 0.025† | 0.024 | 0.147 | sherpa-onnx | 159M | Apache-2.0 | 可 | CPU で 約 90 ms と速い。ReazonSpeech で 1 位（同じコーパスで学習）。音声がいきなり始まると最初の語を落とすので、先頭に無音を足す（値：sherpa-onnx） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Nemotron Speech ASR 0.6B (streaming)](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | NVIDIA | 多言語 | ○ | 0.067 | 0.203 | 0.099 | 0.183 | NeMo, sherpa-onnx, audio.cpp, CrispASR | 0.6B | OpenMDW-1.1 | 可 | 真のストリーミング（80 / 560 ms チャンク）。精度は低めで、短い発話を落とすことがある。文字が出るまで 0.5〜0.8 秒（値：NeMo（GPU）） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | Kotoba Technologies | 専用 | — | 0.040 | 0.111† | 0.033 | 0.151 | Transformers, faster-whisper, whisper.cpp | 約 756M | Apache-2.0 | 可 | 日本語特化の蒸留 Whisper。MaAI-SDS の設定（先頭に 0.5 秒の無音）。GPU 約 175 ms（値：faster-whisper・int8・beam 5（ほかは beam 1）・先頭に 0.5 秒の無音） （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI | 多言語 | — | 0.024 | 0.073 | 0.028 | 0.145 | faster-whisper, whisper.cpp, CrispASR, Transformers | 809M | MIT | 可 | 比較の基準。温度を上げて出し直す設定で測った。faster-whisper（GPU）で 約 180 ms（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | Microsoft | 多言語 | — | — | — | — | — | PyTorch | 約 8B | MIT | 可 | 60 分級の長い音声を話者・時刻つきで書き起こす。未測定 |
| [Moonshine tiny-ja](https://huggingface.co/UsefulSensors/moonshine-tiny-ja) | Moonshine AI (Useful Sensors) | 専用 | — | 0.080 | 0.072 | 0.047 | 0.175 | Transformers, ONNX | 27M | Moonshine AI Community License（年商 100 万ドル未満なら商用可） | 条件つき | 27M と極小。CPU で 約 150 ms（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | IBM | 多言語 | — | 0.031 | 0.072† | 0.014† | 0.168 | Transformers (>=5.16), transcribe.cpp (GGUF) | 470M（エンコーダのみ・CTC） | Apache-2.0 | 可 | GPU で 1 発話 16 ms と最も速い。CPU は float32 で 約 320 ms（bfloat16 のままだと 約 6 秒）。読点をほぼ出さない（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | Alibaba (FunAudioLLM) | 多言語 | — | **0.013** | 0.058 | 0.022 | 0.098 | funasr (PyTorch), CrispASR | 約 0.8B | Apache-2.0 | 可 | どの評価データでも上位（FLEURS は学習との重なりの疑い）。CrispASR で 約 160 ms（funasr は 0.4〜1 秒）（値：funasr） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Haas-ja v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | Third Intelligence | 専用 | — | 0.025 | 0.038† | 0.007† | 0.128 | haas-asr (PyTorch), speech.cpp, audio.cpp, CrispASR | 重み 約 1.2 GB | Apache-2.0 | 可 | 3 つの評価データの平均順位で 1 位・GPU 約 40 ms。ReazonSpeech と Common Voice で学習しているので、その 2 つは有利（値：haas-asr） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | IBM | 多言語 | — | 0.036 | 0.095† | 0.017† | 0.201 | Transformers, CrispASR | 1B（重み 約 4.4 GB） | Apache-2.0 | 可 | LLM 型で GPU でも 約 1.2 秒（CrispASR の Q4_K で 約 460 ms）。Common Voice で学習（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | IBM | 多言語 | — | 0.060 | 0.085† | 0.059† | 0.190 | Transformers, CrispASR | 2B | Apache-2.0 | 可 | 1 発話 約 480 ms。同じ句をくり返すことがある（値：CrispASR・Q4_K（ほかは Q8_0 か元の精度）） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Moonshine base-ja](https://huggingface.co/UsefulSensors/moonshine-base-ja) | Moonshine AI (Useful Sensors) | 専用 | — | 0.070 | **0.044** | 0.035 | 0.220 | Transformers, ONNX | 重み 約 240 MB | Moonshine AI Community License（年商 100 万ドル未満なら商用可） | 条件つき | 放送の話し言葉に強い。CPU 約 330 ms。20 秒を超えると最後に繰り返すことがある（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Parakeet CTC 1.1B ja](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) | grider-transwithai（NVIDIA Parakeet の日本語版） | 専用 | — | 0.053 | 0.084 | 0.039 | 0.167 | NVIDIA NeMo | 1.1B | Apache-2.0（元の Parakeet は CC-BY-4.0） | 可 | GPU 約 65 ms。一般向けの parakeet-ja.nemo を使う（-gal はギャルゲーの音声の版）（値：NeMo） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Mistral AI | 多言語 | ○ | 0.066 | 0.187 | 0.074 | 0.232 | Transformers, vLLM, audio.cpp, CrispASR | 4B（重み 約 8.3 GB） | Apache-2.0 | 可 | 何も出さない発話がある。言語を指定できない。audio.cpp・CrispASR で 約 1.3 秒（値：Transformers（会話は CrispASR）） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [anime-whisper](https://huggingface.co/litagin/anime-whisper) | litagin | 専用 | — | 0.043 | 0.191† | 0.036 | 0.145 | Transformers, faster-whisper, CrispASR（whisper.cpp） | 約 756M（Kotoba-Whisper v2.0 ベース） | MIT | 可 | アニメ・ゲームの台詞で追加学習した Kotoba-Whisper。会話では元と同じくらい、放送の話し言葉では悪い。CrispASR の変換版は大きく悪い（値：Transformers） （[測定](benchmarks/asr/2026-11-3sets.md)） |

### 未評価の候補

| モデル | メモ |
|---|---|
| [Parakeet 1.1B RNNT Multilingual](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) | 日本語を含む多言語版。NVIDIA NIM / NGC での配布（Hugging Face の英語版とは別物） |

## TTS（音声合成）

1 文 ms は 20 文の合成時間の平均、読みの CER は合成した音声を 3 つの ASR で書き起こした中央値、UTMOS は自然さの推定（1〜5。英語で学習した推定器なので目安。モデルより選んだ声の差が大きく出る。クローンするモデルは同じ参照音声を使い、その元の録音は 3.52）、最初の声 ms はストリーミングで最初の声が届くまでの中央値です（[詳しく](benchmarks/2026-11-runtimes.md#tts)）。太字は各列でいちばんよい値です。ストリーミングの ○ は、文を作り終える前に音を出し始められるものです。

| モデル | 開発元 | 日本語 | ストリーミング | 1 文 ms | 読みの CER | UTMOS | 最初の声 ms | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---:|---:|---:|---:|---|---|---|---|---|
| [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech-Engine) | Aivis Project | 専用 | — | 408 | 0.004 | 3.85 | — | 独立エンジン + HTTP API（Python 不要） | AIVMX モデルごと | エンジン LGPL-3.0 / 声はモデルごと | 声・モデルごと | 導入が楽で読みが正確。文の長さによる時間の差が小さい（値：エンジン） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [VOICEVOX](https://github.com/VOICEVOX/voicevox_engine) | VOICEVOX | 専用 | — | **53** | **0.000** | 2.74 | — | 独立エンジン + HTTP API, voicevox_core（C API / Python） | — | エンジン LGPL-3.0 / 声はキャラクターごとの規約 | 声・モデルごと | 日本語の定番で、最も速く読みの誤りなし。CPU（8 スレッド）でも 約 500 ms。UTMOS は声による。声ごとに利用規約が違う（値：voicevox_core） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Irodori-TTS v4.1 Small](https://github.com/Aratako/Irodori-TTS) | Aratako | 専用 | — | 1334 | **0.000** | 3.45 | — | PyTorch, audio.cpp, speech.cpp | 約 2.9 GB（実行時 VRAM 3.6 GB） | MIT（コード・モデル） | 可 | 声を文章で指定（キャプション）でき、クローンもできる。公式の高速化の設定（6 ステップ・sway）で 約 0.8 秒。audio.cpp なら v4 Small が 約 0.4 秒、v4.1 Small MF が 約 0.2 秒（値：PyTorch・40 ステップ） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Tsukasa Speech](https://github.com/Respaired/Tsukasa-Speech) | Respair | 専用 | — | — | — | — | — | PyTorch | 重み 約 4.7 GB | CC-BY-NC-4.0 | 不可 | とても速い（RTF 0.015）・アニメ調。非商用。導入が難しい （[測定](benchmarks/tts/2026-08-19.md)） |
| [mio-tts-cpp](https://github.com/mmnga/mio-tts-cpp) | mmnga | 専用 | — | 796 | 0.017 | 2.23 | — | C++ / GGML（Python 不要） | 約 489 MB（0.1B Q8 + コーデック） | MIT（コード。モデルは MioTTS の Apache-2.0） | 可 | C++ だけで動き省メモリ。数字は前処理で読みに直す。文末に余計な言葉が付くことがある（値：C++ サーバー） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [CosyVoice3 0.5B](https://github.com/FunAudioLLM/CosyVoice) | Alibaba (FunAudioLLM) | 多言語 | ○ | 450 | 0.013 | 3.39 | — | PyTorch, ONNX, CrispASR | 0.5B（配布一式 約 9 GB） | Apache-2.0 | 可 | 日本語はカタカナで渡す（漢字のままだと CER 0.44）。公式の PyTorch は 1 文 約 3.3 秒（値：CrispASR・カタカナ） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Supertonic3](https://github.com/supertone-inc/supertonic) | Supertone | 多言語 | — | 91 | 0.022 | **4.18** | **69** | ONNX（C++ / C# / Go / Rust / Node など）, audio.cpp, CrispASR | 99M | コード MIT・モデル OpenRAIL-M | 可 | UTMOS が最も高い。Python の ONNX 版（CPU）はスレッドを 4〜8 にする。開発は 2026-07 に終了（値：audio.cpp） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Qwen3-TTS (0.6B / 1.7B)](https://github.com/QwenLM/Qwen3-TTS) | Alibaba (Qwen) | 多言語 | ○ | 679 | 0.006 | 4.00 | 92 | PyTorch, audio.cpp, speech.cpp, CrispASR | 0.6B / 1.7B | Apache-2.0 | 可 | 声の指示・設計・クローン。公式の PyTorch は 1 文 8〜9 秒。speech.cpp では最初の声まで 32〜81 ms（値：audio.cpp・1.7B Base・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | 多言語 | ○ | — | — | — | — | PyTorch, ONNX | 82M | Apache-2.0 | 可 | 日本語の音素化が崩れて実用にならなかった （[測定](benchmarks/tts/2026-08-19.md)） |
| [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) | Fish Audio | 多言語 | ○ | 2304 | 0.004 | 3.85 | 91 | PyTorch, audio.cpp | 4B（重み 約 11 GB・VRAM 24 GB 以上） | モデル Fish Audio Research License（非商用。商用は別契約）・コード Fish Audio Community License（年商 50 万ドル未満なら商用可） | 不可 | PyTorch 版は 12 GB GPU に載らないが、audio.cpp（Q8_0）なら動く（値：audio.cpp・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | k2-fsa | 多言語 | — | 363 | **0.000** | 3.46 | 306 | PyTorch, audio.cpp, CrispASR | 約 0.6B（Qwen3-0.6B ベース） | コード Apache-2.0・モデル CC-BY-NC | 不可 | 多言語のクローン。PyTorch では 1 文 約 1 秒（値：audio.cpp・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [MioTTS (0.6B / 1.7B)](https://huggingface.co/Aratako/MioTTS-1.7B) | Aratako | 専用 | — | 806 | 0.006 | 3.28 | — | audio.cpp, CrispASR | 0.6B / 1.7B（+ MioCodec） | Apache-2.0（MioCodec は MIT） | 可 | mio-tts-cpp の大きい版。CrispASR の 0.6B は温度を 0.8 にする（既定の 0 では止まらない）（値：audio.cpp・1.7B・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | OpenBMB | 多言語 | ○ | 1368 | 0.011 | 3.66 | — | PyTorch, audio.cpp, CrispASR | — | Apache-2.0 | 可 | 30 言語のクローン。audio.cpp は 約 0.8 秒と速いが CER 0.068（値：CrispASR・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [MOSS-TTS-Local v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | OpenMOSS | 多言語 | — | 1364 | 0.006 | 3.85 | — | PyTorch, CrispASR, audio.cpp | — | Apache-2.0 | 可 | GPU メモリ 9 GB。audio.cpp では 11.6 GB まで増え、1 文 約 10 秒と遅い（値：CrispASR・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Higgs Audio TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | Boson AI | 多言語 | — | 1292 | 0.004 | 4.01 | 205 | PyTorch, audio.cpp | 4B | 研究・非商用のライセンス | 不可 | クローンの読みが正確。非商用（値：audio.cpp・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) | NetEase Youdao | 多言語 | — | 2568 | 0.391 | 2.84 | 2338 | PyTorch, audio.cpp, CrispASR | — | Apache-2.0 | 可 | 日本語は読みが大きく崩れた（CrispASR でも同じ）（値：audio.cpp・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [OuteTTS 1.0 1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | OuteAI | 多言語 | — | 23056 | 0.104 | 3.55 | — | PyTorch, llama.cpp, audio.cpp | 1B | CC-BY-NC-SA-4.0 | 不可 | audio.cpp では 1 文 約 23 秒と遅い（値：audio.cpp・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |
| [ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini) | Zyphra | 多言語 | — | 270 | 0.017 | 3.26 | — | ggml（C++ サーバー）, PyTorch | GGUF 一式 約 1.2 GB | コード MIT・重み Apache-2.0 | 可 | 速いが、数字の前処理が無く「2026年」などを読み違える（値：ggml サーバー） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Kitten TTS 2](https://huggingface.co/KittenML/kitten-tts-2) | KittenML (Stellon Labs) | 多言語 | — | 3065 | 0.013 | 1.60 | — | kittenml（PyTorch, vLLM） | 重み 約 1 GB | Stellon Labs Community License（商用は登録が必要・年商と調達額が 100 万ドル未満・表示義務あり） | 条件つき | 日本語の声が 4 つ。読みは正確だが遅い（値：PyTorch） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) | Liquid AI | 専用 | ○ | 2535 | 0.006 | 3.72 | — | liquid-audio (PyTorch) | 1.5B（重み 約 3.5 GB） | LFM Open License v1.0（年商 1000 万ドル未満は商用可） | 条件つき | 音声対話モデルの TTS 機能で、声は選べない。ASR としては精度が低い（CER 0.28〜0.79）（値：PyTorch） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) | OpenMOSS | 多言語 | ○ | 2499 | 0.137 | 1.50 | — | PyTorch, ONNX, audio.cpp | 0.1B | Apache-2.0 | 可 | CPU で動く 0.1B。漢字の読み間違いが多い（値：PyTorch） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | litagin02 | 専用 | — | 114 | 0.002 | 2.61 | — | PyTorch, ONNX（AivisSpeech） | モデルごと（JP-Extra 約 250 MB + BERT） | コード AGPL-3.0 / 声はモデルごと（例の JVNV の声は CC-BY-SA-4.0） | 声・モデルごと | 速く読みも正確。声と UTMOS は学習したモデルごと。AivisSpeech も同じ系統（値：PyTorch・JVNV の声） （[測定](benchmarks/2026-11-runtimes.md)） |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | RVC-Boss | 多言語 | ○ | 2084 | 0.011 | 3.88 | 382 | PyTorch | v2ProPlus（約 1.4 GB の重み一式） | MIT | 可 | 5 秒ほどの参照音声で、学習なしにクローン（話者の近さ 0.96）（値：PyTorch・クローン） （[測定](benchmarks/2026-11-runtimes.md)） |

## LLM

日本語の知識や推論の力は [Nejumi LLM リーダーボード](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) や [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) を見てください。ここでは音声対話で効く、最初の文字までの時間と文字の出る速さを測っています。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [LFM2.5 1.2B JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) | Liquid AI | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 1.2B | LFM Open License v1.0（年商 1000 万ドル未満は商用可） | 条件つき | 測った中で最も速い（TTFT 39 ms・215 tok/s、Q8_0） （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [ELYZA-Shortcut-1.0-Qwen-7B](https://huggingface.co/elyza/ELYZA-Shortcut-1.0-Qwen-7B) | ELYZA | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 7B | Apache-2.0 | 可 | ELYZA の日本語モデル（Qwen ベース）。7B なのに TTFT 45 ms （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Qwen3.5（0.8B / 4B / 9B）](https://huggingface.co/Qwen/Qwen3.5-9B) | Alibaba (Qwen) | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 0.8B / 4B / 9B | Apache-2.0 | 可 | thinking を切る必要あり（reasoning_effort none）。9B で TTFT 107 ms・56 tok/s （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Gemma 4（E2B / E4B / 12B / 26B-A4B）](https://huggingface.co/google/gemma-4-26B-A4B-it) | Google | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 2B〜26B（26B は MoE・活性 4B） | Apache-2.0 | 可 | MoE の 26B-A4B は 12B より速い（TTFT 103 ms・94 tok/s）。12B QAT は thinking を切る必要あり （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [LLM-jp-4 8B Instruct](https://huggingface.co/llm-jp/llm-jp-4-8b-instruct) | 国立情報学研究所 LLM-jp | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 8B | Apache-2.0 | 可 | 日本語中心の国産モデル。TTFT 101 ms・63 tok/s （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Nemotron Nano 9B v2 Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) | NVIDIA | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 9B | NVIDIA Nemotron Open Model License（商用可） | 可 | API から thinking を切れず、/no_think でも返事まで約 1.5 秒 （[測定](benchmarks/llm/2026-11-lmstudio.md)） |

## 音声対話の周辺（発話区間検出・ターンテイキング・相槌）

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero | 多言語 | ○ | PyTorch, ONNX | 約 2 MB | MIT | 可 | 定番の発話区間検出 |
| [MaAI](https://github.com/MaAI-Kyoto/MaAI) | 京都大学 | 専用 | ○ | PyTorch | — | MIT | 可 | VAP によるターンテイキング・相槌・頷きの予測 |
| [Smart Turn](https://github.com/pipecat-ai/smart-turn) | Pipecat | 多言語 | — | PyTorch, ONNX | — | BSD-2-Clause | 可 | 発話の終わり（ターンの終わり）を音声から判定 |

## 動かして試す

- [examples/asr/](examples/asr/) - WAV を 1 つ渡して書き起こすだけの、モデルごとの最小の例。
- [apps/](apps/README.md) - マイクで話して認識する ASR アプリと、エンジンと声を選んで読み上げる TTS アプリ（デスクトップ版とブラウザ版）。入れ方もここにあります。

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
