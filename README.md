<!-- このファイルは scripts/build_readme.py が templates/ と data/models.toml から作る。直接編集しない。 -->
# awesome-japanese-speech

日本語の **音声認識（ASR）・音声合成（TTS）・LLM** と、音声対話に使う周辺モデル（発話区間検出・ターンテイキング・相槌）をまとめた一覧です。

[English](README.en.md)

- ローカルで動かせるモデルを中心に、**日本語にどれだけ対応しているか・ストリーミングできるか・ライセンス（商用利用）・動かし方** を並べています。
- 一部のモデルは **同じ PC・同じ条件で実際に測った値**（文字誤り率、処理時間、RTF）と、測ったスクリプトを [benchmarks/](benchmarks/) に載せています。
- 音声対話（話しかけて返事が返ってくるシステム）で使うことを前提に、**精度だけでなく遅延** を重視しています。

> ライセンスは各モデルの公式ページを必ず確認してください。「要確認」はまだ確かめていないものです。
> 「商用」の欄は目安で、法的な判断ではありません。

## 目次

- [ASR（音声認識）](#asr音声認識)
- [TTS（音声合成）](#tts音声合成)
- [LLM](#llm)
- [音声対話の周辺（発話区間検出・ターンテイキング・相槌）](#音声対話の周辺発話区間検出ターンテイキング相槌)
- [ベンチマーク](#ベンチマーク)
- [追加・修正のしかた](#追加修正のしかた)

## ASR（音声認識）

「ストリーミング」の ○ は話している途中から文字が出せるもの、△ は短く区切って毎回認識し直すもの（疑似ストリーミング）です。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) | Alibaba (FunAudioLLM) | 多言語 | — | PyTorch, sherpa-onnx | 約 0.8B（音声エンコーダ + Qwen3-0.6B デコーダ） | 要確認（モデル固有） | 要確認 | 測った中で最も高精度（CER 0.08 台）。LLM デコーダのため遅延がぶれる。生成上限を 128 トークンに絞ると最悪遅延が収まる （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [SenseVoiceSmall](https://github.com/QwenAudio/SenseVoice) | Alibaba (FunAudioLLM) | 多言語 | △（区切って再推論） | FunASR, ONNX, sherpa-onnx, GGUF | 約 234M | 要確認（FunASR モデルライセンス） | 要確認 | 精度と速度の両立（CPU で 1 発話 約 90 ms）。sherpa-onnx の int8-2024-07-17 版を使うこと（2025-09-09 版は広東語向け） （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Parakeet TDT 0.6B Japanese](https://github.com/CrispStrobe/CrispASR/blob/main/hf_readmes/parakeet-tdt-0.6b-ja-GGUF.md) | NVIDIA（日本語版の配布は CrispASR） | 専用 | — | CrispASR (GGUF), sherpa-onnx | 0.6B（Q8 約 660 MB） | CC-BY-4.0 | 可 | GGUF（Q8・CPU）で CER 0.077。sherpa-onnx の CTC 版は CER 0.12 と控えめ （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Cohere ASR Japanese](https://huggingface.co/CKHO/cohere-asr-ja-GGUF) | Cohere | 多言語 | — | CrispASR (GGUF), sherpa-onnx | Q4_K 約 1.5 GB 〜 Q8_0 約 2.4 GB | Apache-2.0 | 可 | 14 言語。日本語の CER 0.086（Q4_K・CPU） （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Qwen3-ASR (1.7B / 0.6B)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Alibaba (Qwen) | 多言語 | — | PyTorch, ONNX, GGUF | 1.7B / 0.6B | Apache-2.0 | 可 | 52 言語と言語識別。0.6B ONNX は CER 0.11。1.7B は手元の ONNX 変換で出力が崩れたため未評価 （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [ReazonSpeech NeMo v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | Reazon Holdings | 専用 | — | NVIDIA NeMo | 619M | Apache-2.0 | 可 | CUDA で CER 0.115・約 300 ms （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [ReazonSpeech k2-v2](https://research.reazon.jp/projects/ReazonSpeech/index.html) | Reazon Holdings | 専用 | — | sherpa-onnx | 159M | Apache-2.0 | 可 | とても軽い日本語専用モデル。CPU 常駐向け （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Nemotron Speech ASR 0.6B (streaming)](https://huggingface.co/nvidia) | NVIDIA | 多言語 | ○ | NeMo, sherpa-onnx（要カスタムビルド） | 0.6B | 要確認（NVIDIA Open Model License） | 要確認 | 測った中で唯一の真のストリーミング（80 / 320 / 560 / 1120 ms チャンク）。精度は CER 0.14〜0.19 （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | Kotoba Technologies | 専用 | — | Transformers, faster-whisper, whisper.cpp | 約 756M | Apache-2.0 | 可 | 日本語特化の蒸留 Whisper。CER 0.133 （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI | 多言語 | — | faster-whisper, whisper.cpp, Transformers | 809M | MIT | 可 | 比較の基準に。CER 0.119。CPU では遅い（1 発話 約 9 秒） （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | Microsoft | 多言語 | — | PyTorch | 約 8B | 要確認 | 要確認 | 60 分級の長い音声を話者・時刻つきで書き起こす。未測定 |
| [Moonshine tiny-ja](https://huggingface.co/UsefulSensors/moonshine-tiny-ja) | Moonshine AI (Useful Sensors) | 専用 | — | Transformers, ONNX | 27M | Moonshine Community License（英語以外のモデルは非商用） | 不可 | 27M と極小なのに句読点ありで CER 0.078（句読点なし 0.052）。CPU で 1 発話 145 ms （[測定](benchmarks/asr/2026-11-jsut20.md)） |
| [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | IBM | 多言語 | — | Transformers (>=5.16), transcribe.cpp (GGUF) | 470M（エンコーダのみ・CTC） | Apache-2.0 | 可 | 7 言語（2026-10 に日本語対応）。GPU で 1 発話 18 ms と非常に速い。読点をほぼ出さない（句読点なし CER 0.065）。-nc 版は日本語非対応だった （[測定](benchmarks/asr/2026-11-jsut20.md)） |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | Alibaba (FunAudioLLM) | 多言語 | — | funasr (PyTorch) | 約 0.8B | Apache-2.0 | 可 | Fun-ASR-Nano の 31 言語版。CER 0.090（句読点なし 0.069）で日本語専用版よりわずかに低い （[測定](benchmarks/asr/2026-11-jsut20.md)） |

### 未評価の候補

| モデル | メモ |
|---|---|
| [Parakeet 1.1B RNNT Multilingual](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) | 日本語を含む多言語版。NVIDIA NIM / NGC での配布（Hugging Face の英語版 1.1B とは別物） |

## TTS（音声合成）

「ストリーミング」の ○ は、文を最後まで作り終える前に音を出し始められるものです。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech-Engine) | Aivis Project | 専用 | — | 独立エンジン + HTTP API（Python 不要） | AIVMX モデルごと | エンジン LGPL-3.0 / 声はモデルごと | 声・モデルごと | 速度・読みの安定性・導入のしやすさで総合 1 位（長文 RTF 0.054） （[測定](benchmarks/tts/2026-08-19.md)） |
| [VOICEVOX](https://github.com/VOICEVOX/voicevox_engine) | VOICEVOX | 専用 | — | 独立エンジン + HTTP API, voicevox_core（C API / Python） | — | エンジン LGPL-3.0 / 声はキャラクターごとの規約 | 声・モデルごと | 日本語の定番。キャラクターごとに利用規約が違う。未測定 |
| [Irodori-TTS v4.1 Small](https://github.com/Aratako/Irodori-TTS) | Aratako | 専用 | — | PyTorch | 約 2.9 GB（実行時 VRAM 3.6 GB） | MIT（コード・モデル） | 可 | 品質・機能で 1 位。声の設計（キャプション）とクローン。長文 RTF 0.092 （[測定](benchmarks/tts/2026-08-19.md)） |
| [Tsukasa Speech](https://github.com/Respaired/Tsukasa-Speech) | Respair | 専用 | — | PyTorch | 重み 約 4.7 GB | CC-BY-NC-4.0 | 不可 | とても速い（RTF 0.015）・アニメ調。非商用。依存関係の修正が多く導入が難しい （[測定](benchmarks/tts/2026-08-19.md)） |
| [mio-tts-cpp](https://github.com/mmnga/mio-tts-cpp) | mmnga | 専用 | — | C++ / GGML（Python 不要） | 約 489 MB（0.1B Q8 + コーデック） | 要確認 | 要確認 | CPU・省メモリ・組み込み向けで 1 位（RTF 0.179）。数字の読みは前処理で正規化を （[測定](benchmarks/tts/2026-08-19.md)） |
| [CosyVoice3 0.5B](https://github.com/FunAudioLLM/CosyVoice) | Alibaba (FunAudioLLM) | 多言語 | ○ | PyTorch, ONNX | 0.5B（配布一式 約 9 GB） | Apache-2.0 | 可 | 音質は良い（RTF 0.557）。日本語はカタカナへの前処理が必要 （[測定](benchmarks/tts/2026-08-19.md)） |
| [Supertonic3](https://github.com/supertone-inc/supertonic) | Supertone | 多言語 | — | ONNX（C++ / C# / Go / Rust / Node など） | 99M | MIT | 可 | CPU だけでほぼ実時間・31 言語。2026-07 にリポジトリがアーカイブされ開発終了 （[測定](benchmarks/tts/2026-08-19.md)） |
| [Qwen3-TTS 0.6B](https://github.com/QwenLM/Qwen3-TTS) | Alibaba (Qwen) | 多言語 | ○ | PyTorch | 0.6B | Apache-2.0 | 可 | 声の指示・設計・クローンが強い。最適化なしの PyTorch では遅い（RTF 2.27） （[測定](benchmarks/tts/2026-08-19.md)） |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | 多言語 | ○ | PyTorch, ONNX | 82M | Apache-2.0 | 可 | 日本語の音素化が崩れて実用にならなかった（短文 CER 3.4） （[測定](benchmarks/tts/2026-08-19.md)） |
| [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) | Fish Audio | 多言語 | ○ | PyTorch | 4B（重み 約 11 GB・VRAM 24 GB 以上） | Research License | 不可 | 12 GB GPU では動かせず未測定 |

## LLM

日本語の知識や推論の力は、既存のリーダーボードを参照してください。

- [Nejumi LLM リーダーボード](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA)
- [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html)

ここでは音声対話で効く **最初の文字までの時間・文字の出る速さ** を手元の GPU で測っています（[benchmarks/llm/2026-11-lmstudio.md](benchmarks/llm/2026-11-lmstudio.md)）。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [LFM2.5 1.2B JP](https://huggingface.co/LiquidAI) | Liquid AI | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 1.2B | 要確認 | 要確認 | 測った中で最も速い（TTFT 39 ms・215 tok/s、Q8_0） （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [ELYZA-Shortcut-1.0-Qwen-7B](https://huggingface.co/elyza) | ELYZA | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 7B | 要確認 | 要確認 | ELYZA の日本語モデル（Qwen ベース）。7B なのに TTFT 45 ms （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Qwen3.5（0.8B / 4B / 9B）](https://huggingface.co/Qwen) | Alibaba (Qwen) | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 0.8B / 4B / 9B | 要確認 | 要確認 | thinking を切る必要あり（reasoning_effort none）。9B で TTFT 107 ms・56 tok/s （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Gemma 4（E2B / E4B / 12B / 26B-A4B）](https://huggingface.co/google) | Google | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 2B〜26B（26B は MoE・活性 4B） | 要確認 | 要確認 | MoE の 26B-A4B は 12B より速い（TTFT 103 ms・94 tok/s）。12B QAT は thinking を切る必要あり （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [LLM-jp-4 8B Instruct](https://huggingface.co/llm-jp) | 国立情報学研究所 LLM-jp | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 8B | 要確認 | 要確認 | 日本語中心の国産モデル。TTFT 101 ms・63 tok/s （[測定](benchmarks/llm/2026-11-lmstudio.md)） |
| [Nemotron Nano 9B v2 Japanese](https://huggingface.co/nvidia) | NVIDIA | 多言語 | ○ | LM Studio, llama.cpp (GGUF) | 9B | 要確認 | 要確認 | API から thinking を切れず、/no_think でも返事まで約 1.5 秒 （[測定](benchmarks/llm/2026-11-lmstudio.md)） |

## 音声対話の周辺（発話区間検出・ターンテイキング・相槌）

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) | Silero | 多言語 | ○ | PyTorch, ONNX | 約 2 MB | MIT | 可 | 定番の発話区間検出 |
| [MaAI](https://github.com/MaAI-Kyoto/MaAI) | 京都大学 | 専用 | ○ | PyTorch | — | 要確認 | 要確認 | VAP によるターンテイキング・相槌・頷きの予測 |
| [Smart Turn](https://github.com/pipecat-ai/smart-turn) | Pipecat | 多言語 | — | PyTorch, ONNX | — | 要確認 | 要確認 | 発話の終わり（ターンの終わり）を音声から判定 |

## ベンチマーク

| 対象 | 内容 | 結果 |
|---|---|---|
| ASR | JSUT 20 発話での文字誤り率（CER）と 1 発話あたりの処理時間 | [2026-08](benchmarks/asr/2026-08-jsut20.md)、[2026-11（新しい候補・句読点なしの CER）](benchmarks/asr/2026-11-jsut20.md) |
| TTS | 長文の RTF（生成時間 ÷ 音声の長さ）、メモリ、ASR で書き起こして読み間違いを検査 | [benchmarks/tts/2026-08-19.md](benchmarks/tts/2026-08-19.md) |
| LLM | 音声対話での最初の文字までの時間（TTFT）と文字の出る速さ（LM Studio・10 モデル） | [2026-11](benchmarks/llm/2026-11-lmstudio.md) |

測り方と測定環境は [benchmarks/README.md](benchmarks/README.md) にまとめています。

## 追加・修正のしかた

[CONTRIBUTING.md](CONTRIBUTING.md) を見てください。表は `data/models.toml` から自動で作っているので、README を直接は編集しません。

## 関連

- [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) — ここに載せたモデルの多くを組み込んで試している音声対話システム
