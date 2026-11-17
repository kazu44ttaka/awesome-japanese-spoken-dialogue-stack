<!-- このファイルは scripts/build_readme.py が templates/ と data/models.toml から作る。直接編集しない。 -->
# awesome-japanese-speech

日本語の **音声認識（ASR）・音声合成（TTS）・LLM** と、音声対話に使う周辺モデル（発話区間検出・ターンテイキング・相槌）をまとめた一覧です。

[English](README.en.md)

- ローカルで動かせるモデルを中心に、**日本語にどれだけ対応しているか・ストリーミングできるか・ライセンス（商用利用）・動かし方** を並べています。
- **同じモデルを実行系ごとに比べられます。** 元の実装（PyTorch など）・sherpa-onnx・faster-whisper と、ggml 系の CrispASR・audio.cpp・speech.cpp で、
  同じモデルの速さ・精度・メモリを同じ PC で測りました（[実行系の比較](benchmarks/2026-11-runtimes.md)）。同じモデルでも、実行系で数倍速くなったり、精度が落ちたりします。
- **カバー範囲が広く**、日本語で使える主な ASR・TTS の多くを、**同じ PC・同じ条件で実際に測った値**（文字誤り率、処理時間、RTF、メモリ）と、測ったスクリプトとともに [benchmarks/](benchmarks/) に載せています。
- 音声対話（話しかけて返事が返ってくるシステム）で使うことを前提に、**精度だけでなく遅延** を重視しています。

> ライセンスは各モデルの公式ページを必ず確認してください。「要確認」はまだ確かめていないものです。
> 「商用」の欄は目安で、法的な判断ではありません。

## 目次

- [ASR（音声認識）](#asr音声認識)
- [TTS（音声合成）](#tts音声合成)
- [LLM](#llm)
- [音声対話の周辺（発話区間検出・ターンテイキング・相槌）](#音声対話の周辺発話区間検出ターンテイキング相槌)
- [動かして試す](#動かして試す)
- [ベンチマーク](#ベンチマーク)
- [追加・修正のしかた](#追加修正のしかた)

## ASR（音声認識）

「ストリーミング」の ○ は話している途中から文字が出せるもの、△ は短く区切って毎回認識し直すもの（疑似ストリーミング）です。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) | Alibaba (FunAudioLLM) | 多言語 | — | PyTorch, sherpa-onnx, CrispASR, audio.cpp | 約 0.8B（音声エンコーダ + Qwen3-0.6B デコーダ） | 要確認（モデル固有） | 要確認 | 読みの CER は FLEURS 0.059・ReazonSpeech 0.132・Common Voice 0.034（funasr・CUDA）。sherpa-onnx 版は放送の話し言葉で崩れやすい（0.217）。CrispASR（CUDA）なら 1 発話 約 130 ms と PyTorch（640 ms）の 5 倍速い （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [SenseVoiceSmall](https://github.com/QwenAudio/SenseVoice) | Alibaba (FunAudioLLM) | 多言語 | △（区切って再推論） | FunASR, ONNX, sherpa-onnx, CrispASR, audio.cpp | 約 234M | 要確認（FunASR モデルライセンス） | 要確認 | CPU で 1 発話 約 90 ms・GPU（fp32）で 約 40 ms と速い。読み上げでは読みの CER 0.05 前後だが、放送の話し言葉（ReazonSpeech）では 0.191 と崩れる。sherpa-onnx の 2024-07-17 版を使うこと（2025-09-09 版は広東語向け） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Parakeet TDT-CTC 0.6B ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | NVIDIA | 専用 | — | NeMo, sherpa-onnx, speech.cpp, CrispASR | 0.6B | CC-BY-4.0 | 可 | 読みの CER 0.030〜0.093。GPU なら NeMo で 1 発話 約 55 ms・speech.cpp で 約 40 ms、CPU でも sherpa-onnx の CTC 版で 約 180 ms と速い。CrispASR 版は発話の前後の小さな声まで書き起こして CER が悪い （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Cohere Transcribe (03-2026)](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Cohere | 多言語 | — | PyTorch, sherpa-onnx, audio.cpp, CrispASR | Q4_K 約 1.5 GB 〜 Q8_0 約 2.4 GB | Apache-2.0（取得に同意が必要） | 可 | 3 つの評価データを通して上位で安定（読みの CER 0.017〜0.049）。CPU で 1 発話 約 0.8 秒（sherpa-onnx・8 スレッド）、GPU なら audio.cpp で 約 100 ms。14 言語 （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Cohere ASR ja v0.1](https://huggingface.co/efwkjn/cohere-asr-ja-v0.1) | efwkjn（Cohere Transcribe の日本語版） | 専用 | — | CrispASR (GGUF) | 2B（Q8_0 約 2.4 GB） | 要確認（元は Apache-2.0） | 要確認 | 読みの CER は FLEURS 0.026・ReazonSpeech 0.093・Common Voice 0.018 で、元の Cohere Transcribe（0.025・0.049・0.017）より放送の話し言葉で悪い。GPU で 1 発話 約 190 ms （[測定](benchmarks/2026-11-runtimes.md)） |
| [Qwen3-ASR (1.7B / 0.6B)](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Alibaba (Qwen) | 多言語 | — | PyTorch, sherpa-onnx, speech.cpp, audio.cpp, CrispASR | 1.7B / 0.6B | Apache-2.0 | 可 | 52 言語と言語識別。0.6B（PyTorch・CUDA）は読み上げで読みの CER 0.05〜0.06 だが、放送の話し言葉では 0.228 と崩れる。sherpa-onnx 版は stream.set_option("language", "Japanese") で言語を指定すること（指定しないと日本語以外で出ることがある）。1.7B は読みの CER 0.033〜0.196。speech.cpp（GPU）なら 1.7B でも 1 発話 約 165 ms と PyTorch（930 ms）の 5 倍以上速く、CER も同じ （[測定](benchmarks/2026-11-runtimes.md)） |
| [ReazonSpeech NeMo v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | Reazon Holdings | 専用 | — | NVIDIA NeMo, speech.cpp, CrispASR | 619M | Apache-2.0 | 可 | 読みの CER は 3 つの評価データとも 0.04 前後で、放送の話し言葉（ReazonSpeech 0.039）に強い。NeMo（CUDA）で 1 発話 約 350 ms、speech.cpp なら 約 100 ms で CER も同じ （[測定](benchmarks/2026-11-runtimes.md)） |
| [ReazonSpeech k2-v2](https://research.reazon.jp/projects/ReazonSpeech/index.html) | Reazon Holdings | 専用 | — | sherpa-onnx | 159M | Apache-2.0 | 可 | CPU で 1 発話 約 90 ms・GPU（fp32）で 約 70 ms と速く、精度も上位（読みの CER 0.024〜0.068）。ReazonSpeech コーパスで学習しているので放送の話し言葉に強い （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Nemotron Speech ASR 0.6B (streaming)](https://huggingface.co/nvidia) | NVIDIA | 多言語 | ○ | NeMo, sherpa-onnx, audio.cpp, CrispASR | 0.6B | 要確認（NVIDIA Open Model License） | 要確認 | 測った中で唯一の真のストリーミング（80 / 560 ms チャンク）。精度は低め（読みの CER 0.10〜0.24）で、短い発話を何も出さずに落とすことがある。PyPI の sherpa-onnx 2.0.1 で動く。audio.cpp（GPU）なら 1 発話 約 55 ms （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Kotoba-Whisper v2.2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | Kotoba Technologies | 専用 | — | Transformers, faster-whisper, whisper.cpp | 約 756M | Apache-2.0 | 可 | 日本語特化の蒸留 Whisper。CER 0.133 （[測定](benchmarks/asr/2026-08-jsut20.md)） |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | OpenAI | 多言語 | — | faster-whisper, whisper.cpp, CrispASR, Transformers | 809M | MIT | 可 | 比較の基準に。読みの CER 0.028〜0.073（transformers・CUDA、温度を上げて出し直す設定を入れたとき）。faster-whisper（CUDA FP16）は 1 発話 約 180 ms。CPU では 3 秒以上かかる （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) | Microsoft | 多言語 | — | PyTorch | 約 8B | 要確認 | 要確認 | 60 分級の長い音声を話者・時刻つきで書き起こす。未測定 |
| [Moonshine tiny-ja](https://huggingface.co/UsefulSensors/moonshine-tiny-ja) | Moonshine AI (Useful Sensors) | 専用 | — | Transformers, ONNX | 27M | Moonshine Community License（英語以外のモデルは非商用） | 不可 | 27M と極小。CPU で 1 発話 約 150 ms、読みの CER 0.050〜0.086 （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite Speech 5.0 470M TurboCTC](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc) | IBM | 多言語 | — | Transformers (>=5.16), transcribe.cpp (GGUF) | 470M（エンコーダのみ・CTC） | Apache-2.0 | 可 | 7 言語（2026-10 に日本語対応）。GPU で 1 発話 16 ms と非常に速く、読みの CER 0.014〜0.073。CPU でも float32 なら 約 320 ms（既定の bfloat16 のままだと約 6 秒）。読点をほぼ出さない。-nc 版は日本語非対応だった （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | Alibaba (FunAudioLLM) | 多言語 | — | funasr (PyTorch), CrispASR | 約 0.8B | Apache-2.0 | 可 | 3 つの評価データで上位（読みの CER 0.014〜0.058）。FLEURS だけ飛び抜けてよいのは学習データとの重なりの可能性あり。funasr（CUDA）で 1 発話 0.4〜1 秒、CrispASR なら 約 160 ms （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Haas-ja v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) | Third Intelligence | 専用 | — | haas-asr (PyTorch), speech.cpp, audio.cpp, CrispASR | 重み 約 1.2 GB | Apache-2.0 | 可 | 3 つの評価データの平均順位で 1 位（読みの CER 0.007〜0.038）・GPU で 1 発話 約 40 ms（speech.cpp・audio.cpp でも同じ速さと CER）。ただし ReazonSpeech と Common Voice を学習に使っているので、その 2 つは有利に出ている（学習に入っていない FLEURS で 0.029） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite 4.0 1B Speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | IBM | 多言語 | — | Transformers, CrispASR | 1B（重み 約 4.4 GB） | Apache-2.0 | 可 | 読みの CER 0.017〜0.098。Common Voice 17 の日本語を学習に使っている。LLM 型で、GPU でも 1 発話 約 1.2 秒（CrispASR の Q4_K なら 約 460 ms） （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | IBM | 多言語 | — | Transformers, CrispASR | 2B | Apache-2.0 | 可 | CrispASR（Q4_K）で読みの CER 0.061〜0.088、1 発話 約 480 ms。同じ句をくり返して止まらない発話がある （[測定](benchmarks/2026-11-runtimes.md)） |
| [Moonshine base-ja](https://huggingface.co/UsefulSensors/moonshine-base-ja) | Moonshine AI (Useful Sensors) | 専用 | — | Transformers, ONNX | 重み 約 240 MB | 要確認（Moonshine Community License とみられる） | 要確認 | 放送の話し言葉（ReazonSpeech）で 2 位（読みの CER 0.045）。CPU でも 1 発話 約 330 ms。20 秒を超える長い発話では最後に繰り返しが出ることがある （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Parakeet CTC 1.1B ja](https://huggingface.co/grider-transwithai/parakeet-ctc-1.1b-ja) | grider-transwithai（NVIDIA Parakeet の日本語版） | 専用 | — | NVIDIA NeMo | 1.1B | Apache-2.0（元の Parakeet は CC-BY-4.0） | 可 | GPU で 1 発話 約 65 ms、読みの CER 0.039〜0.081。一般向けの parakeet-ja.nemo を使うこと（同じリポジトリの parakeet-ja-gal.nemo はギャルゲーの音声で追加学習した版）。CrispASR 向けの GGUF はギャル語版から変換されたとみられる （[測定](benchmarks/asr/2026-11-3sets.md)） |
| [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Mistral AI | 多言語 | ○ | Transformers, vLLM, audio.cpp, CrispASR | 4B（重み 約 8.3 GB） | Apache-2.0 | 可 | 日本語では読みの CER 0.069〜0.189 と中位以下で、何も出さない発話がある（ReazonSpeech で 1000 中 96）。言語を指定できない。transformers では 1 発話 約 5 秒、audio.cpp・CrispASR なら 約 1.3 秒（公式は vLLM を推奨。未測定） （[測定](benchmarks/asr/2026-11-3sets.md)） |

### 未評価の候補

| モデル | メモ |
|---|---|
| [Parakeet 1.1B RNNT Multilingual](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) | 日本語を含む多言語版。NVIDIA NIM / NGC での配布（Hugging Face の英語版 1.1B とは別物） |

## TTS（音声合成）

「ストリーミング」の ○ は、文を最後まで作り終える前に音を出し始められるものです。

| モデル | 開発元 | 日本語 | ストリーミング | 動かし方 | 大きさ | ライセンス | 商用 | メモ |
|---|---|---|---|---|---|---|---|---|
| [AivisSpeech](https://github.com/Aivis-Project/AivisSpeech-Engine) | Aivis Project | 専用 | — | 独立エンジン + HTTP API（Python 不要） | AIVMX モデルごと | エンジン LGPL-3.0 / 声はモデルごと | 声・モデルごと | 導入が楽で読みが正確。短い文から長い文まで 1 文 約 400 ms（GPU）。長文 RTF 0.054 （[測定](benchmarks/tts/2026-11-tts.md)） |
| [VOICEVOX](https://github.com/VOICEVOX/voicevox_engine) | VOICEVOX | 専用 | — | 独立エンジン + HTTP API, voicevox_core（C API / Python） | — | エンジン LGPL-3.0 / 声はキャラクターごとの規約 | 声・モデルごと | 日本語の定番。voicevox_core（GPU）で 1 文 約 50 ms・読みの誤りなし、CPU（8 スレッド）でも 約 500 ms。キャラクターごとに利用規約が違う （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Irodori-TTS v4.1 Small](https://github.com/Aratako/Irodori-TTS) | Aratako | 専用 | — | PyTorch, audio.cpp, speech.cpp | 約 2.9 GB（実行時 VRAM 3.6 GB） | MIT（コード・モデル） | 可 | 品質・機能で 1 位。声の設計（キャプション）とクローン。1 文 約 1.3 秒（40 ステップ）、公式の高速化の設定（6 ステップ・sway）で 約 0.8 秒。audio.cpp（GGUF）なら 約 0.4 秒、MF 版は 約 0.2 秒で読みも崩れない （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Tsukasa Speech](https://github.com/Respaired/Tsukasa-Speech) | Respair | 専用 | — | PyTorch | 重み 約 4.7 GB | CC-BY-NC-4.0 | 不可 | とても速い（RTF 0.015）・アニメ調。非商用。依存関係の修正が多く導入が難しい （[測定](benchmarks/tts/2026-08-19.md)） |
| [mio-tts-cpp](https://github.com/mmnga/mio-tts-cpp) | mmnga | 専用 | — | C++ / GGML（Python 不要） | 約 489 MB（0.1B Q8 + コーデック） | 要確認 | 要確認 | C++ だけで動き省メモリ。GPU で 1 文 約 0.8 秒（RTF 0.186）。数字の読みは前処理で正規化を。文の最後に余計な言葉が付くことがある （[測定](benchmarks/tts/2026-11-tts.md)） |
| [CosyVoice3 0.5B](https://github.com/FunAudioLLM/CosyVoice) | Alibaba (FunAudioLLM) | 多言語 | ○ | PyTorch, ONNX, CrispASR | 0.5B（配布一式 約 9 GB） | Apache-2.0 | 可 | 日本語はカタカナで渡す（参照音声の書き起こしも）。漢字のままだと読みが崩れる（CER 0.44）。カタカナなら読みの CER 0.008（公式・1 文 約 3.3 秒）、CrispASR では 0.016・約 0.45 秒 （[測定](benchmarks/2026-11-runtimes.md)） |
| [Supertonic3](https://github.com/supertone-inc/supertonic) | Supertone | 多言語 | — | ONNX（C++ / C# / Go / Rust / Node など）, audio.cpp, CrispASR | 99M | MIT | 可 | 31 言語。audio.cpp のサーバー（GGUF・CUDA）なら 1 文 約 90 ms・RTF 0.024 と速い。Python の ONNX 版（CPU）はスレッドを 4〜8 にして 1 文 約 0.9 秒（既定の全コアだと 約 2 秒）。2026-07 にリポジトリがアーカイブされ開発終了 （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Qwen3-TTS (0.6B / 1.7B)](https://github.com/QwenLM/Qwen3-TTS) | Alibaba (Qwen) | 多言語 | ○ | PyTorch, audio.cpp, speech.cpp, CrispASR | 0.6B / 1.7B | Apache-2.0 | 可 | 声の指示・設計・クローンが強い。公式の PyTorch（flash-attn なし）は 1 文 8〜9 秒（RTF 1.9）と遅いが、audio.cpp・speech.cpp なら 約 0.6 秒（RTF 0.12〜0.16）で読みも同じくらい正確（クローンで読みの CER 0.002〜0.010） （[測定](benchmarks/2026-11-runtimes.md)） |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | 多言語 | ○ | PyTorch, ONNX | 82M | Apache-2.0 | 可 | 日本語の音素化が崩れて実用にならなかった（短文 CER 3.4） （[測定](benchmarks/tts/2026-08-19.md)） |
| [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) | Fish Audio | 多言語 | ○ | PyTorch, audio.cpp | 4B（重み 約 11 GB・VRAM 24 GB 以上） | Research License | 不可 | PyTorch 版は 12 GB GPU に載らないが、audio.cpp（Q8_0）なら動き、クローンで読みの CER 0.006・1 文 約 2.3 秒 （[測定](benchmarks/2026-11-runtimes.md)） |
| [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | k2-fsa | 多言語 | — | PyTorch, audio.cpp, CrispASR | 約 0.6B（Qwen3-0.6B ベース） | コード Apache-2.0・モデル CC-BY-NC | 不可 | 多言語のクローン。読みが正確（読みの CER 0.000〜0.010）で、audio.cpp なら 1 文 約 0.36 秒（PyTorch は 約 1 秒） （[測定](benchmarks/2026-11-runtimes.md)） |
| [MioTTS (0.6B / 1.7B)](https://huggingface.co/Aratako/MioTTS-1.7B) | Aratako | 専用 | — | audio.cpp, CrispASR | 0.6B / 1.7B（+ MioCodec） | Apache-2.0（MioCodec は MIT） | 可 | mio-tts-cpp の大きい版。1.7B は audio.cpp でクローンして読みの CER 0.008・1 文 約 0.8 秒。CrispASR の 0.6B は温度の既定が 0 で止まらなくなるので、0.8 にすること （[測定](benchmarks/2026-11-runtimes.md)） |
| [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | OpenBMB | 多言語 | ○ | PyTorch, audio.cpp, CrispASR | — | Apache-2.0 | 可 | 30 言語のクローン。CrispASR で読みの CER 0.014・1 文 約 1.4 秒、audio.cpp は 約 0.8 秒だが CER 0.075 （[測定](benchmarks/2026-11-runtimes.md)） |
| [MOSS-TTS-Local v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | OpenMOSS | 多言語 | — | PyTorch, CrispASR, audio.cpp | — | Apache-2.0 | 可 | クローンで読みの CER 0.006（CrispASR・1 文 約 1.4 秒・GPU メモリ 9 GB）。audio.cpp では GPU メモリが 11.6 GB まで増え、1 文 約 10 秒と遅い （[測定](benchmarks/2026-11-runtimes.md)） |
| [Higgs Audio TTS 3 4B](https://huggingface.co/bosonai/higgs-tts-3-4b) | Boson AI | 多言語 | — | PyTorch, audio.cpp | 4B | 研究・非商用のライセンス | 不可 | audio.cpp（Q8_0）でクローンして読みの CER 0.006・1 文 約 1.3 秒 （[測定](benchmarks/2026-11-runtimes.md)） |
| [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) | NetEase Youdao | 多言語 | — | PyTorch, audio.cpp, CrispASR | — | Apache-2.0 | 可 | 日本語は読みが大きく崩れた（読みの CER 0.42〜0.43、audio.cpp・CrispASR とも） （[測定](benchmarks/2026-11-runtimes.md)） |
| [OuteTTS 1.0 1B](https://huggingface.co/OuteAI/Llama-OuteTTS-1.0-1B) | OuteAI | 多言語 | — | PyTorch, llama.cpp, audio.cpp | 1B | CC-BY-NC-SA-4.0 | 不可 | audio.cpp で 1 文 約 23 秒（RTF 3.3）と遅く、読みの CER 0.110 （[測定](benchmarks/2026-11-runtimes.md)） |
| [ZONOS3-Mini](https://github.com/Zyphra/ZONOS3-Mini) | Zyphra | 多言語 | — | ggml（C++ サーバー）, PyTorch | GGUF 一式 約 1.2 GB | コード MIT・重み Apache-2.0 | 可 | ggml の CUDA サーバーで 1 文 約 270 ms（RTF 0.064）と速い。数字の前処理が無く「2026年」などを読み違える （[測定](benchmarks/tts/2026-11-tts.md)） |
| [Kitten TTS 2](https://huggingface.co/KittenML/kitten-tts-2) | KittenML (Stellon Labs) | 多言語 | — | kittenml（PyTorch, vLLM） | 重み 約 1 GB | Stellon Labs Community License（商用は登録が必要・年商 100 万ドル未満・表示義務あり） | 要確認 | 日本語の声が 4 つ。読みは正確（読みの CER 0.014）だが、torch では 1 文 約 3 秒（RTF 0.72） （[測定](benchmarks/tts/2026-11-tts.md)） |
| [LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) | Liquid AI | 専用 | ○ | liquid-audio (PyTorch) | 1.5B（重み 約 3.5 GB） | LFM Open License v1.0（年商 1000 万ドル未満は商用可） | 要確認 | 音声対話モデルの TTS 機能。読みは正確（読みの CER 0.008）だが 1 文 約 2.5 秒（RTF 0.71）。声は選べない。ASR としては（CrispASR）読みの CER 0.29〜0.78 と低く、同じ句をくり返すことがある （[測定](benchmarks/tts/2026-11-tts.md)） |
| [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) | OpenMOSS | 多言語 | ○ | PyTorch, ONNX, audio.cpp | 0.1B | Apache-2.0 | 可 | CPU で動く 0.1B。日本語は漢字の読み間違いが多く（読みの CER 0.14〜0.17）、まれに止まらず上限まで生成する （[測定](benchmarks/tts/2026-11-tts.md)） |

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

## 動かして試す

- **最小の推論例**：[examples/asr/](examples/asr/) — WAV を 1 つ渡すと書き起こすだけの、モデルごとのスクリプト（単独で動く）
- **マイクで試すアプリ**：[apps/asr_demo.py](apps/asr_demo.py)（PySide6）・[apps/asr_web.py](apps/asr_web.py)（ブラウザ） — モデルを選び、話した声をその場で認識する。発話区間検出（Silero VAD）で区切って確定し、話している途中の経過も出せる（疑似ストリーミング）。認識にかかった時間と、話し終わりから表示までの時間を表示する。Nemotron 3.5 streaming は真のストリーミング（音声を少しずつ入れて内部状態を引き継ぐ）で動く
- **TTS を試すアプリ**：[apps/tts_demo.py](apps/tts_demo.py)（PySide6）・[apps/tts_web.py](apps/tts_web.py)（ブラウザ） — エンジン（AivisSpeech・VOICEVOX・VOICEVOX Core・Irodori-TTS・Supertonic 3・mio-tts-cpp・Qwen3-TTS・Tsukasa・Kokoro・ZONOS3-Mini・Kitten TTS 2・LFM2.5-Audio・MOSS-TTS-Nano）と声を選んで読み上げる。「文ごとに合成して、できた文から再生」で、最初の音が出るまでの時間（音声対話で効く）を確かめられる。AivisSpeech / VOICEVOX は入っていればアプリから起動できる。入れ方は [apps/README.md](apps/README.md)

```
uv venv --python 3.11 .venv
uv pip install sherpa-onnx sounddevice soundfile numpy
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-deps "silero-vad==6.2.1"               # 発話区間検出（MaAI-SDS と同じ）
uv pip install "transformers>=5.16" funasr                 # transformers / funasr のモデルも使うなら
uv pip install "pyside6-essentials==6.12.0" fastapi "uvicorn[standard]"   # アプリの画面（PySide6）と Web 版
python apps/asr_demo.py      # Web 版は python apps/asr_web.py（http://127.0.0.1:8765）
.venv-tts/Scripts/python apps/tts_demo.py   # TTS は別の環境（apps/README.md）。Web 版は apps/tts_web.py
```

## ベンチマーク

| 対象 | 内容 | 結果 |
|---|---|---|
| ASR・TTS | 同じモデルを実行系（元の実装・sherpa-onnx・CrispASR・audio.cpp・speech.cpp など）ごとに比べた速さ・精度・メモリ（ASR 18 モデル・TTS 22 モデル） | [2026-11（実行系の比較）](benchmarks/2026-11-runtimes.md) |
| ASR | 3 つの公開評価データ（FLEURS・ReazonSpeech・Common Voice）での読みの CER（23 構成） | [2026-11（3 つの評価データ）](benchmarks/asr/2026-11-3sets.md) |
| ASR | 同じ 100 発話での処理時間を GPU・CPU（8 スレッド）に分けて測定 | [2026-11（処理時間）](benchmarks/asr/2026-11-timing.md) |
| ASR | JSUT 20 発話での文字誤り率（CER）と 1 発話あたりの処理時間 | [2026-08](benchmarks/asr/2026-08-jsut20.md)、[2026-11（新しい候補・句読点なしの CER）](benchmarks/asr/2026-11-jsut20.md) |
| TTS | 短い文から長い文まで 20 文の合成時間・RTF と、3 つの ASR で書き起こした読みの CER（GPU・CPU に分けて 9 エンジン・18 構成） | [2026-11](benchmarks/tts/2026-11-tts.md) |
| TTS | 長文の RTF（生成時間 ÷ 音声の長さ）、メモリ、ASR で書き起こして読み間違いを検査 | [2026-08](benchmarks/tts/2026-08-19.md) |
| LLM | 音声対話での最初の文字までの時間（TTFT）と文字の出る速さ（LM Studio・10 モデル） | [2026-11](benchmarks/llm/2026-11-lmstudio.md) |

測り方と測定環境は [benchmarks/README.md](benchmarks/README.md) にまとめています。

## 追加・修正のしかた

[CONTRIBUTING.md](CONTRIBUTING.md) を見てください。表は `data/models.toml` から自動で作っているので、README を直接は編集しません。

## 関連

- [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) — ここに載せたモデルの多くを組み込んで試している音声対話システム
