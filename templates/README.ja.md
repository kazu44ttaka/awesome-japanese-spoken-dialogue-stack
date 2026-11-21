# awesome-japanese-spoken-dialogue-component

日本語の音声対話システムを作るための部品、**音声認識（ASR）・音声合成（TTS）・LLM** と周辺モデル（発話区間検出・ターンテイキング・相槌）をまとめた一覧です。

[English](README.en.md)

- ローカルで動かせるモデルを中心に、**日本語にどれだけ対応しているか・ストリーミングできるか・ライセンス（商用利用）・動かし方** を並べています。
- **同じモデルを実行系ごとに比べられます。** 元の実装（PyTorch など）・sherpa-onnx・faster-whisper と、ggml 系の CrispASR・audio.cpp・speech.cpp で、
  同じモデルの速さ・精度・メモリを同じ PC で測りました（[実行系の比較](benchmarks/2026-11-runtimes.md)）。同じモデルでも、実行系で数倍速くなったり、精度が落ちたりします。
- **カバー範囲が広く**、日本語で使える主な ASR・TTS の多くを、**同じ PC・同じ条件で実際に測った値**（文字誤り率、処理時間、RTF、メモリ）と、測ったスクリプトとともに [benchmarks/](benchmarks/) に載せています。
- 音声対話（話しかけて返事が返ってくるシステム）で使うことを前提に、**精度だけでなく遅延** を重視しています。

> ライセンスは 2026-11-21 に各モデルの公式ページ（モデルカード・リポジトリ・ライセンスの本文）で確かめました。変わることがあるので、使う前に必ず公式ページを確認してください。商用の「条件つき」は年商などの条件を満たせば使えるもの、「不明」はライセンスの記載が無いものです。
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

<!-- table:asr -->

### 未評価の候補

<!-- table:asr_todo -->

## TTS（音声合成）

「ストリーミング」の ○ は、文を最後まで作り終える前に音を出し始められるものです。

<!-- table:tts -->

## LLM

日本語の知識や推論の力は、既存のリーダーボードを参照してください。

- [Nejumi LLM リーダーボード](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA)
- [Swallow LLM Leaderboard](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html)

ここでは音声対話で効く **最初の文字までの時間・文字の出る速さ** を手元の GPU で測っています（[benchmarks/llm/2026-11-lmstudio.md](benchmarks/llm/2026-11-lmstudio.md)）。

<!-- table:llm -->

## 音声対話の周辺（発話区間検出・ターンテイキング・相槌）

<!-- table:dialog -->

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

## ライセンス

このリポジトリ（一覧・測定結果・コード）は MIT です（[LICENSE](LICENSE)）。載せたモデル・データセットのライセンスは、それぞれの公式ページに従います。

## 関連

- [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) — ここに載せたモデルの多くを組み込んで試している音声対話システム
