<!-- このファイルは scripts/build_readme.py が templates/ と data/models.toml から作る。直接編集しない。 -->
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

| モデル | ストリーミング | FLEURS | ReazonSpeech | CV | 会話 | 動かし方 | 商用 | メモ |
|---|---|---:|---:|---:|---:|---|---|---|
| [Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) | — | 0.033 | 0.125 | 0.034 | 0.176 | PyTorch, sherpa-onnx, CrispASR, audio.cpp | 可 | CrispASR（GPU）で 1 発話 約 130 ms と PyTorch の 5 倍速い。会話の値は相づち 3 区間の繰り返しの暴走を含む（除くと 0.093）。sherpa-onnx 版は話し言葉で崩れる |
| [SenseVoiceSmall](https://github.com/QwenAudio/SenseVoice) | △（区切って再推論） | 0.049 | 0.183 | 0.044 | 0.126 | FunASR, ONNX, sherpa-onnx, CrispASR, audio.cpp | 可 | CPU 約 90 ms・GPU 約 40 ms と速いが、放送の話し言葉で崩れる。sherpa-onnx は 2024-07-17 版を使う |
| [Parakeet TDT-CTC 0.6B ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | — | 0.038 | 0.092† | 0.023 | 0.122 | NeMo, sherpa-onnx, speech.cpp, CrispASR | 可 | GPU 約 40〜55 ms・CPU 約 180 ms。疑似ストリーミングで文字が出るまで 約 70 ms と最も速い |
| [Cohere Transcribe (03-2026)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | — | 0.023 | 0.048 | **0.017** | 0.140 | PyTorch, sherpa-onnx, audio.cpp, CrispASR | 可 | どの評価データでも上位。CPU 約 0.8 秒・GPU（audio.cpp）約 100 ms。14 言語 |
| [Cohere ASR ja v0.1](https://huggingface.co/efwkjn/cohere-asr-ja-v0.1) | — | 0.025 | 0.092 | 0.018 | 0.127 | CrispASR (GGUF) | 不明 | Cohere Transcribe の日本語の追加学習版。放送の話し言葉では元より悪い。GPU 約 190 ms |
| [Qwen3-ASR (1.7B / 0.6B)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | — | 0.029 | 0.182 | 0.036 | **0.091** | PyTorch, sherpa-onnx, speech.cpp, audio.cpp, CrispASR | 可 | 会話で最もよいが、放送の話し言葉で崩れる。speech.cpp（GPU）で 1.7B が 約 165 ms（PyTorch の 5 倍以上速い）。0.6B は会話 0.105。sherpa-onnx では言語の指定が必要 |
| [ReazonSpeech NeMo v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | — | 0.038 | 0.039† | 0.033 | 0.160 | NVIDIA NeMo, speech.cpp, CrispASR | 可 | 放送の話し言葉に強い。speech.cpp で 1 発話 約 100 ms（NeMo は 約 350 ms） |
| [ReazonSpeech k2-v2](https://huggingface.co/reazon-research/reazonspeech-k2-v2) | — | 0.036 | 0.025† | 0.024 | 0.147 | sherpa-onnx | 可 | CPU で 約 90 ms と速い。ReazonSpeech で 1 位（同じコーパスで学習）。音声がいきなり始まると最初の語を落とすので、先頭に無音を足す |
| [Nemotron Speech ASR 0.6B (streaming)](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | ○ | 0.067 | 0.203 | 0.099 | 0.183 | NeMo, sherpa-onnx, audio.cpp, CrispASR | 可 | 真のストリーミング（80 / 560 ms チャンク）。精度は低めで、短い発話を落とすことがある。文字が出るまで 0.5〜0.8 秒 |
| [Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | — | 0.040 | 0.111† | 0.033 | 0.151 | Transformers, faster-whisper, whisper.cpp | 可 | 日本語特化の蒸留 Whisper。MaAI-SDS の設定（先頭に 0.5 秒の無音）。GPU 約 175 ms・CPU 約 3.5 秒 |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | — | 0.024 | 0.073 | 0.028 | 0.145 | faster-whisper, whisper.cpp, CrispASR, Transformers | 可 | 比較の基準。温度を上げて出し直す設定で測った。faster-whisper（GPU）で 約 180 ms |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | — | — | — | — | — | PyTorch | 可 | 60 分級の長い音声を話者・時刻つきで書き起こす。未測定 |
| [Moonshine tiny-ja](https://huggingface.co/UsefulSensors/moonshine-tiny-ja) | — | 0.080 | 0.072 | 0.047 | 0.175 | Transformers, ONNX, CrispASR | 条件つき | 27M と極小。CPU で 約 150 ms。CrispASR（GGUF）は GPU で 約 70 ms と速いが、放送の話し言葉で悪い（0.110） |
| [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | — | 0.031 | 0.072† | 0.014† | 0.168 | Transformers (>=5.16), transcribe.cpp (GGUF) | 可 | GPU で 1 発話 16 ms と最も速い。CPU は float32 で 約 320 ms（bfloat16 のままだと 約 6 秒）。読点をほぼ出さない |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | — | **0.013** | 0.058 | 0.022 | 0.098 | funasr (PyTorch), CrispASR | 可 | どの評価データでも上位（FLEURS は学習との重なりの疑い）。CrispASR で 約 160 ms（funasr は 0.4〜1 秒） |
| [Haas-ja v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | — | 0.025 | 0.038† | 0.007† | 0.128 | haas-asr (PyTorch), speech.cpp, audio.cpp, CrispASR | 可 | 3 つの評価データの平均順位で 1 位・GPU 約 40 ms。ReazonSpeech と Common Voice で学習しているので、その 2 つは有利 |
| [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | — | 0.036 | 0.095† | 0.017† | 0.201 | Transformers, CrispASR | 可 | LLM 型で GPU でも 約 1.2 秒（CrispASR の Q4_K で 約 460 ms）。Common Voice で学習 |
| [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | — | 0.060 | 0.085† | 0.059† | 0.190 | Transformers, CrispASR | 可 | 1 発話 約 480 ms。同じ句をくり返すことがある。Plus 版も同じくらい（会話 0.158）。NAR 版は日本語に非対応 |
| [Moonshine base-ja](https://huggingface.co/UsefulSensors/moonshine-base-ja) | — | 0.070 | **0.044** | 0.035 | 0.220 | Transformers, ONNX, CrispASR | 条件つき | 放送の話し言葉に強い。CPU 約 330 ms。CrispASR（GGUF）は GPU で 約 130 ms と速いが、放送の話し言葉で悪い（0.086）。20 秒を超えると最後に繰り返すことがある |
| [Parakeet CTC 1.1B ja](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) | — | 0.053 | 0.084 | 0.039 | 0.167 | NVIDIA NeMo | 可 | GPU 約 65 ms・CPU 約 590 ms。一般向けの parakeet-ja.nemo を使う（-gal はギャルゲーの音声の版。CrispASR 向けの GGUF はこちらから変換されている） |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | ○ | 0.066 | 0.187 | 0.074 | 0.232 | Transformers, vLLM, audio.cpp, CrispASR | 可 | 何も出さない発話がある。言語を指定できない。audio.cpp・CrispASR で 約 1.3 秒 |
| [anime-whisper](https://huggingface.co/litagin/anime-whisper) | — | 0.043 | 0.191† | 0.036 | 0.145 | Transformers, faster-whisper, CrispASR（whisper.cpp） | 可 | アニメ・ゲームの台詞で追加学習した Kotoba-Whisper。会話では元と同じくらい、放送の話し言葉では悪い。CrispASR の変換版は大きく悪い |

### 未評価の候補

| モデル | メモ |
|---|---|
| [Parakeet 1.1B RNNT Multilingual](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) | 日本語を含む多言語版。NVIDIA NIM / NGC での配布（Hugging Face の英語版とは別物） |

## TTS（音声合成）

1 文 ms は 20 文の平均、読みの CER は合成音声を ASR で書き起こした誤り、UTMOS は自然さの目安（1〜5）、最初の声 ms はストリーミングで最初の音が出るまで（[詳しく](benchmarks/tts/2026-11-tts.md)）。

| モデル | ストリーミング | 1 文 ms | 読みの CER | UTMOS | 最初の声 ms | 動かし方 | 商用 | メモ |
|---|---|---:|---:|---:|---:|---|---|---|
| [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech-Engine) | — | 408 | 0.004 | 3.85 | — | 独立エンジン + HTTP API（Python 不要） | 声・モデルごと | 導入が楽で読みが正確。文の長さによる時間の差が小さい。CPU では 1 文 約 1.8 秒 |
| [VOICEVOX](https://github.com/VOICEVOX/voicevox_engine) | — | **53** | **0.000** | 2.74 | — | 独立エンジン + HTTP API, voicevox_core（C API / Python） | 声・モデルごと | 日本語の定番で、最も速く読みの誤りなし。CPU（8 スレッド）でも 約 500 ms。UTMOS は声による。声ごとに利用規約が違う |
| [Irodori-TTS v4.1 Small](https://github.com/Aratako/Irodori-TTS) | — | 1334 | **0.000** | 3.45 | — | PyTorch, audio.cpp, speech.cpp | 可 | 声を文章で指定（キャプション）でき、クローンもできる。公式の高速化の設定（6 ステップ・sway）で 約 0.8 秒。audio.cpp なら v4 Small が 約 0.4 秒、v4.1 Small MF が 約 0.2 秒 |
| [Tsukasa Speech](https://github.com/Respaired/Tsukasa-Speech) | — | — | — | — | — | PyTorch | 不可 | とても速い（RTF 0.015）・アニメ調。非商用。導入が難しい |
| [mio-tts-cpp](https://github.com/mmnga/mio-tts-cpp) | — | 796 | 0.017 | 2.23 | — | C++ / GGML（Python 不要） | 可 | C++ だけで動き省メモリ。CPU でも 1 文 約 0.7 秒。数字は前処理で読みに直す。文末に余計な言葉が付くことがある |
| [CosyVoice3 0.5B](https://github.com/FunAudioLLM/CosyVoice) | ○ | 450 | 0.013 | 3.39 | — | PyTorch, ONNX, CrispASR | 可 | 日本語はカタカナで渡す（漢字のままだと CER 0.44）。公式の PyTorch は 1 文 約 3.3 秒 |
| [Supertonic3](https://github.com/supertone-inc/supertonic) | — | 91 | 0.022 | **4.18** | **69** | ONNX（C++ / C# / Go / Rust / Node など）, audio.cpp, CrispASR | 可 | UTMOS が最も高い。Python の ONNX 版（CPU）はスレッドを 4〜8 にする。開発は 2026-07 に終了 |
| [Qwen3-TTS (0.6B / 1.7B)](https://github.com/QwenLM/Qwen3-TTS) | ○ | 679 | 0.006 | 4.00 | 92 | PyTorch, audio.cpp, speech.cpp, CrispASR | 可 | 声の指示・設計・クローン。公式の PyTorch は 1 文 8〜9 秒。speech.cpp では最初の声まで 32〜81 ms。VoiceDesign（声を文章で指定）も audio.cpp で 1 文 0.71 秒・最初の声 25 ms |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | ○ | — | — | — | — | PyTorch, ONNX | 可 | 日本語の音素化が崩れて実用にならなかった |
| [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) | ○ | 2304 | 0.004 | 3.85 | 91 | PyTorch, audio.cpp | 不可 | PyTorch 版は 12 GB GPU に載らないが、audio.cpp（Q8_0）なら動く |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | — | 363 | **0.000** | 3.46 | 306 | PyTorch, audio.cpp, CrispASR | 不可 | 多言語のクローン。PyTorch では 1 文 約 1 秒 |
| [MioTTS (0.6B / 1.7B)](https://huggingface.co/Aratako/MioTTS-1.7B) | — | 806 | 0.006 | 3.28 | — | audio.cpp, CrispASR | 可 | mio-tts-cpp の大きい版。CrispASR の 0.6B は温度を 0.8 にする（既定の 0 では止まらない） |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | ○ | 1368 | 0.011 | 3.66 | — | PyTorch, audio.cpp, CrispASR | 可 | 30 言語のクローン。audio.cpp は 約 0.8 秒と速いが CER 0.068 |
| [MOSS-TTS-Local v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | — | 1364 | 0.006 | 3.85 | — | PyTorch, CrispASR, audio.cpp | 可 | GPU メモリ 9 GB。audio.cpp では 11.6 GB まで増え、1 文 約 10 秒と遅い |
| [Higgs Audio TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | — | 1292 | 0.004 | 4.01 | 205 | PyTorch, audio.cpp | 不可 | クローンの読みが正確。非商用 |
| [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) | — | 2568 | 0.391 | 2.84 | 2338 | PyTorch, audio.cpp, CrispASR | 可 | 日本語は読みが大きく崩れた（CrispASR でも同じ） |
| [OuteTTS 1.0 1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | — | 23056 | 0.104 | 3.55 | — | PyTorch, llama.cpp, audio.cpp | 不可 | audio.cpp では 1 文 約 23 秒と遅い |
| [ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini) | — | 270 | 0.017 | 3.26 | — | ggml（C++ サーバー）, audio.cpp, PyTorch | 可 | 速いが、数字の前処理が無く「2026年」などを読み違える。audio.cpp では 1 文 約 225 ms・最初の声 88 ms |
| [Kitten TTS 2](https://huggingface.co/KittenML/kitten-tts-2) | — | 3065 | 0.013 | 1.60 | — | kittenml（PyTorch, vLLM）, audio.cpp | 条件つき | 日本語の声が 4 つ。読みは正確。torch では遅いが、audio.cpp なら 1 文 約 0.9 秒 |
| [LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) | ○ | 2535 | 0.006 | 3.72 | — | liquid-audio (PyTorch), audio.cpp, CrispASR | 条件つき | 音声対話モデルの TTS 機能で、声は選べない。audio.cpp なら 1 文 約 0.33 秒・最初の声 51 ms。ASR としても audio.cpp なら使える（CER 0.035〜0.249）が、CrispASR では同じ句をくり返す |
| [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) | ○ | 2499 | 0.137 | 1.50 | — | PyTorch, ONNX, audio.cpp | 可 | CPU で動く 0.1B。漢字の読み間違いが多い |
| [Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2) | — | 114 | 0.002 | 2.61 | — | PyTorch, ONNX（AivisSpeech） | 声・モデルごと | 速く読みも正確（CPU でも 1 文 約 1.3 秒）。声と UTMOS は学習したモデルごと。AivisSpeech も同じ系統 |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | ○ | 2084 | 0.011 | 3.88 | 382 | PyTorch | 可 | 5 秒ほどの参照音声で、学習なしにクローン（話者の近さ 0.96） |

## LLM

知識や推論は [Nejumi](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA) や [Swallow](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) のリーダーボードを。ここでは最初の文字までの時間と生成の速さを測っています。

| モデル | ストリーミング | 動かし方 | 商用 | メモ |
|---|---|---|---|---|
| [LFM2.5 1.2B JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) | ○ | LM Studio, llama.cpp (GGUF) | 条件つき | 測った中で最も速い（TTFT 39 ms・215 tok/s、Q8_0） |
| [ELYZA-Shortcut-1.0-Qwen-7B](https://huggingface.co/elyza/ELYZA-Shortcut-1.0-Qwen-7B) | ○ | LM Studio, llama.cpp (GGUF) | 可 | ELYZA の日本語モデル（Qwen ベース）。7B なのに TTFT 45 ms |
| [Qwen3.5（0.8B / 4B / 9B）](https://huggingface.co/Qwen/Qwen3.5-9B) | ○ | LM Studio, llama.cpp (GGUF) | 可 | thinking を切る必要あり（reasoning_effort none）。9B で TTFT 107 ms・56 tok/s |
| [Gemma 4（E2B / E4B / 12B / 26B-A4B）](https://huggingface.co/google/gemma-4-26B-A4B-it) | ○ | LM Studio, llama.cpp (GGUF) | 可 | MoE の 26B-A4B は 12B より速い（TTFT 103 ms・94 tok/s）。12B QAT は thinking を切る必要あり |
| [LLM-jp-4 8B Instruct](https://huggingface.co/llm-jp/llm-jp-4-8b-instruct) | ○ | LM Studio, llama.cpp (GGUF) | 可 | 日本語中心の国産モデル。TTFT 101 ms・63 tok/s |
| [Nemotron Nano 9B v2 Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese) | ○ | LM Studio, llama.cpp (GGUF) | 可 | API から thinking を切れず、/no_think でも返事まで約 1.5 秒 |

## 音声対話の周辺

発話区間検出・ターンテイキング・相槌など。

| モデル | ストリーミング | 動かし方 | 商用 | メモ |
|---|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | ○ | PyTorch, ONNX | 可 | 定番の発話区間検出 |
| [MaAI](https://github.com/MaAI-Kyoto/MaAI) | ○ | PyTorch | 可 | VAP によるターンテイキング・相槌・頷きの予測 |
| [Smart Turn](https://github.com/pipecat-ai/smart-turn) | — | PyTorch, ONNX | 可 | 発話の終わり（ターンの終わり）を音声から判定 |

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
