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
- **マイクで試すアプリ**：[apps/asr_demo.py](apps/asr_demo.py) — モデルを選び、話した声をその場で認識する。発話区間検出（Silero VAD）で区切って確定し、話している途中の経過も出せる（疑似ストリーミング）。認識にかかった時間と、話し終わりから表示までの時間を表示する。Nemotron 3.5 streaming は真のストリーミング（音声を少しずつ入れて内部状態を引き継ぐ）で動く
- **TTS を試すアプリ**：[apps/tts_demo.py](apps/tts_demo.py) — エンジン（AivisSpeech・VOICEVOX・VOICEVOX Core・Irodori-TTS・Supertonic 3・mio-tts-cpp・Qwen3-TTS・Tsukasa・Kokoro）と声を選んで読み上げる。「文ごとに合成して、できた文から再生」で、最初の音が出るまでの時間（音声対話で効く）を確かめられる。AivisSpeech / VOICEVOX は入っていればアプリから起動できる。入れ方は [apps/README.md](apps/README.md)

```
uv venv --python 3.11 .venv
uv pip install sherpa-onnx sounddevice soundfile numpy      # sherpa-onnx のモデルだけなら PyTorch 不要
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # transformers / funasr のモデルも使うなら
uv pip install "transformers>=5.16" funasr
python apps/asr_demo.py
.venv-tts/Scripts/python apps/tts_demo.py   # TTS は別の環境（apps/README.md）
```

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
