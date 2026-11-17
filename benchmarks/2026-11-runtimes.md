# 同じモデルを実行系ごとに比べる（2026-11）

同じモデルでも、どの実行系（推論のライブラリ）で動かすかで、速さ・精度・メモリが大きく変わる。このページでは、
日本語の ASR 18 モデル・TTS 22 モデルを、元の実装（PyTorch など）・sherpa-onnx・faster-whisper と、ggml 系の
3 つの実行系 [CrispASR](https://github.com/CrispStrobe/CrispASR)・[audio.cpp](https://github.com/0xShug0/audio.cpp)・
[speech.cpp](https://github.com/nyosegawa/speech.cpp) で、同じ PC・同じ入力で測って並べた。

- 実行系の入れ方とこの PC で動かすときの注意：[docs/runtimes.md](../docs/runtimes.md)。モデルの名前と立ち上げ方は
  [runtimes/catalog.py](../runtimes/catalog.py)
- PC：[benchmarks/README.md](README.md) のもの（RTX 5070 Ti Laptop・Core Ultra 9 275HX）。AC 電源・電源プラン「高パフォーマンス」
- 測定日：2026-11-17〜18。1 回だけの測定なので、数 % の差は気にしない
- 表はこのファイルの [make_runtime_page.py](make_runtime_page.py) で結果の JSON から作っている。「—」は測っていない（その実行系に無い、
  または動かなかった）ところ

## 測り方

**ASR**（[asr/scripts/eval_asr.py](asr/scripts/eval_asr.py)）
- 速さ：`timing100`（ReazonSpeech 40・Common Voice 30・FLEURS 30 発話、平均 8.5 秒。[ASR の速さのページ](asr/2026-11-timing.md) と同じ）を 1 発話ずつ
  認識し、認識し終えるまでの平均。GPU と CPU（8 スレッド）。speech.cpp は新しい長さの音声で GPU のシェーダーを作り直すので、
  100 発話をすべて 1 回通してから測った（ほかは 3 発話でウォームアップ）
- 精度：読み（カナ）で比べた CER。`timing100` の値は全構成、3 つの評価データ（FLEURS 650・ReazonSpeech 1000・Common Voice 1000 発話）は、
  初めて測るモデルと、`timing100` で元の実装より CER が 0.01 以上ずれた構成だけ
- メモリ：`timing100` の最長 10 発話（18〜24 秒）を認識したときの最大。GPU は読み込む前からの増え方（nvidia-smi の GPU 全体の使用量）、
  RAM は測定の処理とサーバーなどの子プロセスの合計。読み込み時間はモデルを取ってきたあとの 2 回目の読み込み

**TTS**（[tts/scripts/bench_tts.py](tts/scripts/bench_tts.py)・[tts/scripts/score_tts.py](tts/scripts/score_tts.py)）
- 速さ：[TTS のページ](tts/2026-11-tts.md) と同じ 20 文を 1 文ずつ合成し、作り終えるまでの平均（ストリーミングは使わない）
- 精度：合成した音声を 3 つの ASR で書き起こした読みの CER の中央値（同じページの方法）。声の自然さの評価ではない
- 声：声を選べるモデルは同じ声（Qwen3-TTS CustomVoice は Ono_Anna、Supertonic 3 は F1）。参照音声で声をクローンするモデルは、
  すべて同じ参照音声（Common Voice の 1 発話・5.5 秒、[data/voices](../data/voices/README.md)）
- メモリと読み込み時間：3 文（最短・中間・最長）を合成したときの最大
- CPU：小さいモデルだけ、同じ 20 文を CPU（8 スレッド）で合成した平均

## わかったこと

- **速さは、モデルごとに一番速い実行系が違う。** ASR は Qwen3-ASR なら speech.cpp、Fun-ASR なら CrispASR、Cohere・Nemotron なら
  audio.cpp。TTS は多くのモデルで audio.cpp が速く（Irodori・Qwen3-TTS・OmniVoice・VoxCPM2）、MOSS-TTS-Local だけは CrispASR が 7 倍速い
- **LLM 型のモデルは、ggml の実行系にすると元の実装より数倍〜十数倍速い。**
  - ASR：Qwen3-ASR 1.7B は PyTorch 932 ms → speech.cpp 165 ms、0.6B は 716 ms → 95 ms、Voxtral は 5.2 秒 → 1.3 秒、
    Fun-ASR-Nano は 640 ms → 134 ms（CrispASR）
  - TTS：Qwen3-TTS は公式の PyTorch（flash-attn なし）で 1 文 8〜9 秒 → audio.cpp・speech.cpp で 約 0.6 秒、
    CosyVoice3 は 3.3 秒 → 0.45 秒（CrispASR）、OmniVoice は 1.0 秒 → 0.36 秒（audio.cpp）
  - LLM 型でない NeMo のモデルでも、ReazonSpeech NeMo v2 は 348 ms → 98 ms（speech.cpp）。Haas・Parakeet のように元から速いものは同じくらい
- **精度は、ほとんどの実行系で元の実装とそろう。** 量子化（Q8_0）での差は CER で 0.01 以内が多い。
  speech.cpp の ReazonSpeech NeMo v2・Parakeet TDT-CTC は、3 つの評価データでも元の NeMo と CER が 0.001 しか違わない
- **同じモデルでも、実行系によって結果が大きく変わるものがある。**
  - ※1 CrispASR の NeMo 系（Haas・ReazonSpeech・Parakeet）は、発話の前後に小さく入った別の人の声まで書き起こし、CER が 0.06〜0.13 悪い。
    F16 の重みでも、VAD の区切りを止めても同じなので、実装の違いとみられる
  - ※2 CrispASR 向けに配布されている Parakeet CTC 1.1B ja の GGUF は、元のリポジトリのギャル語版から変換されたとみられる
    （出力がギャル語版の PyTorch とよく似ていて、CER も近い）
  - ※3 Granite 4.1 2B・LFM2.5-Audio JP（CrispASR）は、同じ句をくり返して止まらない発話がある
  - TTS の読みの CER は、実行系で 0.01〜0.07 ずれるものがある。Irodori は audio.cpp（0.006・0.008）より speech.cpp（0.032・0.038）が悪く、
    VoxCPM2・MOSS-TTS-Local は逆に CrispASR（0.014・0.006）より audio.cpp（0.075・0.030）が悪い
  - 速さも大きく違うことがある。MOSS-TTS-Local は CrispASR 1.4 秒・audio.cpp 9.8 秒、Confucius4-TTS は audio.cpp 2.6 秒・CrispASR 28 秒、
    Irodori 500M / 600M は audio.cpp 0.35 秒・CrispASR 1.8 秒
- **ggml の実行系は、読み込みが速く、メモリも少ない。**
  - 読み込みは多くが 0.5〜5 秒で、PyTorch（NeMo 15〜50 秒、funasr 39 秒、qwen-tts・CosyVoice 15〜16 秒）よりずっと速い
  - speech.cpp は特に小さい。Haas・ReazonSpeech NeMo v2 は GPU メモリ 0.8 GB（NeMo は 3.7 GB・4.9 GB）、Qwen3-ASR 1.7B は 2.6 GB
    （PyTorch 4.6 GB）で、RAM もほとんど使わない（0.2 GB）
  - 例外は audio.cpp の MOSS-TTS-Local で、GPU メモリが 11.6 GB とこの PC の 12 GB にほぼ届き、読み込みに 5 分、1 文に 約 10 秒かかった
    （CrispASR は 9.0 GB・1.4 秒）
- **CPU だけなら、sherpa-onnx（int8）が速いことが多い。** ggml の実行系の強みは GPU で出る
  - SenseVoice は sherpa-onnx 89 ms・audio.cpp 178 ms・CrispASR 232 ms、Parakeet TDT-CTC は sherpa-onnx 180 ms・speech.cpp 310 ms、
    Cohere Transcribe は sherpa-onnx 780 ms・CrispASR 808 ms・audio.cpp 1472 ms
  - Qwen3-ASR 0.6B（speech.cpp 826 ms・sherpa-onnx 916 ms・PyTorch 1600 ms）と Fun-ASR-Nano（CrispASR 798 ms・sherpa-onnx 851 ms）は同じくらい
  - Whisper large-v3-turbo の CrispASR（whisper.cpp）は CPU で 1 発話 10.5 秒と、faster-whisper（3.5 秒）より遅い
  - TTS では、Supertonic 3 は audio.cpp 662 ms・ONNX 864 ms・CrispASR 877 ms、MOSS-TTS-Nano は audio.cpp 836 ms・PyTorch 2472 ms。
    Irodori MF（4.4〜4.5 秒）と Qwen3-TTS 0.6B（speech.cpp 3.7 秒）は CPU では会話に使うには遅い
- **使い方を間違えると、モデルが悪く見える。**
  - CosyVoice3 は、公式の例にあるとおり日本語をカタカナで渡す必要がある（参照音声の書き起こしも）。漢字のまま渡すと読みが崩れる
    （CrispASR で CER 0.44、カタカナで 0.016。公式の PyTorch でもカタカナで 0.008）
  - CrispASR の LLM 型 TTS は温度の既定が 0（貪欲）で、MioTTS 0.6B は 20 文中 3 文が止まらずに 30 秒まで伸びた。モデルの既定の 0.8 にして測った
- 測らなかったもの：IndexTTS2 / 2.5（日本語に非対応。同じ音のくり返しになる）、ZONOS v0.1 と Kokoro（CrispASR で日本語を読めない・
  声のファイルが別に要る）、VibeVoice-ASR（長い録音向けで、短い発話では英語の文などを出す。audio.cpp では時間切れ）、
  audio.cpp の Granite Speech 5.0 TurboCTC（日本語に対応する前の英語だけの重み）

## ASR

<!-- asr-tables:start -->
### Haas-ja v1.0

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| haas-asr（PyTorch・NeMo） | 42 | — | 0.019 | 0.029 | 0.038 | 0.007 | 15.2 | 3.7 | 3.4 |  |
| speech.cpp（Vulkan） | 43 | 314 | 0.019 | — | — | — | 0.5 | 0.8 | 0.2 |  |
| audio.cpp | 42 | 552 | 0.019 | — | — | — | 2.7 | 2.0 | 2.2 |  |
| CrispASR | 176 | 659 | 0.089 | 0.080 | 0.138 | 0.018 | 1.0 | 1.0 | 0.7 | ※1 |

### ReazonSpeech NeMo v2

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 348 | — | 0.027 | 0.040 | 0.039 | 0.040 | 15.8 | 4.9 | 3.3 |  |
| speech.cpp（Vulkan） | 98 | 506 | 0.027 | 0.041 | 0.038 | 0.041 | 0.5 | 0.8 | 0.2 |  |
| CrispASR | 142 | 842 | 0.105 | — | — | — | 1.0 | 1.0 | 0.8 | ※1 |

### Parakeet TDT-CTC 0.6B ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 54 | — | 0.040 | 0.041 | 0.093 | 0.030 | 19.1 | 5.1 | 3.4 |  |
| sherpa-onnx（int8） | — | 180 | 0.040 | — | 0.088 | 0.036 | 2.4 | 0.7 | 1.4 |  |
| speech.cpp（Vulkan） | 42 | 310 | 0.041 | 0.041 | 0.092 | 0.031 | 0.5 | 0.7 | 0.2 |  |
| CrispASR | 77 | 649 | 0.109 | — | — | — | 1.1 | 1.0 | 0.8 | ※1 |

### Parakeet CTC 1.1B ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 68 | — | 0.056 | 0.053 | 0.081 | 0.039 | 26.5 | 8.3 | 4.9 |  |
| CrispASR | 67 | — | 0.183 | — | — | — | 1.0 | 1.7 | 1.3 | ※2 |

### Qwen3-ASR 0.6B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-asr（PyTorch） | 716 | 1600 | 0.103 | 0.054 | 0.228 | 0.060 | 12.8 | 2.0 | 2.5 |  |
| sherpa-onnx（int8） | — | 916 | 0.120 | 0.078 | 0.245 | 0.072 | 3.7 | 3.3 | 2.4 |  |
| speech.cpp（Vulkan） | 95 | 826 | 0.106 | — | — | — | 1.0 | 1.4 | 0.3 |  |
| audio.cpp | 210 | 1228 | 0.109 | — | — | — | 1.2 | 2.3 | 0.9 |  |
| CrispASR | 248 | 1080 | 0.105 | — | — | — | 1.0 | 2.5 | 1.1 |  |

### Qwen3-ASR 1.7B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-asr（PyTorch） | 932 | — | 0.089 | 0.033 | 0.196 | 0.038 | 14.0 | 4.6 | 4.1 |  |
| speech.cpp（Vulkan） | 165 | — | 0.085 | 0.033 | 0.196 | 0.038 | 1.5 | 2.6 | 0.2 |  |
| audio.cpp | 280 | — | 0.084 | — | — | — | 2.7 | 3.5 | 1.9 |  |
| CrispASR | 274 | — | 0.088 | — | — | — | 2.5 | 4.2 | 2.5 |  |

### Qwen3-ASR 1.7B ja-anime

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 274 | — | 0.096 | 0.041 | 0.207 | 0.043 | 2.5 | 4.2 | 2.5 |  |

### Fun-ASR-Nano-2512

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| funasr（PyTorch） | 640 | 1727 | 0.072 | — | 0.132 | 0.034 | 38.9 | 3.4 | 4.3 |  |
| sherpa-onnx | 902 | 851 | 0.082 | 0.068 | 0.217 | 0.045 | 7.1 | 6.4 | 1.6 |  |
| audio.cpp | 267 | 1286 | 0.091 | 0.054 | 0.273 | 0.037 | 3.7 | 2.1 | 2.1 |  |
| CrispASR | 134 | 798 | 0.069 | — | — | — | 1.0 | 1.7 | 1.1 |  |

### Fun-ASR-MLT-Nano-2512

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| funasr（PyTorch） | 639 | 1755 | 0.024 | — | 0.058 | 0.023 | 39.0 | 3.5 | 4.3 |  |
| CrispASR | 159 | — | 0.038 | — | — | — | 2.5 | — | — | 3 データとメモリの測定では最初の発話でサーバーが落ちた |

### SenseVoiceSmall

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sherpa-onnx | 41 | 89 | 0.090 | — | 0.191 | 0.051 | 1.7 | 2.2 | 1.3 |  |
| audio.cpp | 99 | 178 | 0.120 | 0.050 | 0.218 | 0.050 | 0.7 | 0.5 | 0.7 |  |
| CrispASR | 59 | 232 | 0.090 | — | — | — | 0.5 | 0.5 | 0.6 |  |

### Cohere Transcribe

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sherpa-onnx（int8） | — | 780 | 0.021 | 0.025 | 0.049 | 0.017 | 6.7 | 2.7 | 3.0 |  |
| audio.cpp | 102 | 1472 | 0.022 | — | — | — | 2.7 | 3.6 | 3.1 |  |
| CrispASR | 189 | 808 | 0.022 | — | — | — | 1.0 | 1.8 | 1.3 | Q4_K |

### Cohere ASR ja v0.1

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 188 | — | 0.052 | 0.026 | 0.093 | 0.018 | 1.5 | 2.8 | 2.5 |  |

### Granite 4.0 1B Speech

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 1166 | 5156 | 0.048 | 0.047 | 0.098 | 0.017 | 9.6 | 4.6 | 4.8 |  |
| CrispASR | 460 | — | 0.058 | 0.062 | 0.083 | 0.075 | 2.5 | 3.3 | 2.6 | Q4_K |

### Granite 4.1 2B Speech

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 480 | — | 0.177 | 0.070 | 0.088 | 0.061 | 2.5 | 3.3 | 2.7 | Q4_K・※3 |

### Voxtral Mini 4B Realtime

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 5181 | — | 0.136 | 0.069 | 0.189 | 0.080 | 12.2 | 9.0 | 8.7 |  |
| audio.cpp | 1331 | — | 0.121 | — | — | — | 3.2 | 5.3 | 4.2 |  |
| CrispASR | 1349 | — | 0.121 | 0.067 | 0.191 | 0.080 | 3.0 | 5.6 | 4.2 |  |

### Nemotron 3.5 ASR streaming 0.6B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 475 | 1986 | 0.129 | 0.101 | 0.231 | 0.099 | 49.8 | 2.7 | 5.9 |  |
| sherpa-onnx（int8・560 ms） | — | 760 | 0.149 | 0.117 | 0.237 | 0.098 | 2.0 | 0.3 | 1.4 |  |
| audio.cpp | 55 | — | 0.161 | 0.104 | 0.244 | 0.105 | 1.7 | 1.3 | 1.1 |  |
| CrispASR | 342 | — | 0.154 | 0.103 | 0.245 | 0.105 | 1.0 | 1.8 | 1.3 | F16 |

### Whisper large-v3-turbo

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 334 | 6513 | 0.038 | 0.028 | 0.073 | 0.035 | 6.1 | 2.2 | 2.3 |  |
| faster-whisper（CTranslate2） | 180 | 3470 | 0.057 | 0.028 | 0.104 | 0.036 | 1.8 | 2.3 | 1.0 |  |
| CrispASR | 195 | 10514 | 0.051 | 0.029 | 0.077 | 0.037 | 1.5 | 2.0 | 0.6 | whisper.cpp の ggml |

### LFM2.5-Audio-1.5B-JP

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 196 | — | 0.336 | 0.352 | 0.780 | 0.285 | 1.5 | 2.3 | 1.4 | Q5_K・※3 |

<!-- asr-tables:end -->

## TTS

<!-- tts-tables:start -->
### Supertonic 3

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| supertonic（ONNX・CPU 4 スレッド） | — | — | 0.021 | 864 | 2.9 | 0.0 | 0.5 |  |
| audio.cpp | 91 | 0.023 | 0.015 | 662 | 1.4 | 0.6 | 0.7 |  |
| CrispASR | 191 | 0.047 | 0.016 | 877 | 0.8 | 0.5 | 0.7 |  |

### Irodori-TTS v4 Small

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 571 | 0.127 | 0.032 | — | 1.8 | 2.2 | 0.2 |  |
| audio.cpp | 394 | 0.092 | 0.006 | — | 3.1 | 1.5 | 1.9 |  |

### Irodori-TTS v4.1 Small

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Irodori-TTS（PyTorch） | 1334 | 0.289 | 0.000 | — | 12.0 | 4.2 | 6.9 | キャプションで声を決める |

### Irodori-TTS v4.1 Small MF

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 165 | 0.037 | 0.038 | 4381 | 1.6 | 2.1 | 0.2 |  |
| audio.cpp | 207 | 0.048 | 0.008 | 4547 | 2.9 | 2.7 | 1.7 |  |

### Irodori-TTS v4.1 Anime

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 390 | 0.092 | 0.002 | — | 3.0 | 1.5 | 1.6 |  |

### Irodori-TTS Large

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 1304 | 0.295 | 0.024 | — | 3.3 | 4.0 | 0.2 |  |

### Irodori-TTS 500M v3

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 351 | 0.084 | 0.006 | — | 3.0 | 2.4 | 1.5 |  |
| CrispASR | 1800 | 0.434 | 0.018 | — | 2.1 | 2.3 | 0.9 |  |

### Irodori-TTS 600M v3 VoiceDesign

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 366 | 0.085 | 0.004 | — | 3.0 | 2.5 | 1.6 |  |
| CrispASR | 1887 | 0.447 | 0.012 | — | 2.2 | 2.5 | 1.0 | Q4_K |

### Qwen3-TTS 0.6B Base（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 8753 | 1.886 | 0.010 | — | 14.7 | 2.9 | 2.5 | flash-attn なし |
| audio.cpp | 612 | 0.132 | 0.002 | — | 3.7 | 2.6 | 3.7 |  |
| CrispASR | 1036 | 0.237 | 0.010 | — | 2.3 | 2.4 | 1.1 |  |

### Qwen3-TTS 0.6B CustomVoice（Ono_Anna）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 9070 | 1.869 | 0.010 | — | 14.6 | 2.7 | 2.5 | flash-attn なし |
| speech.cpp（Vulkan） | 604 | 0.115 | 0.019 | 3683 | 1.8 | 1.3 | 0.2 |  |
| CrispASR | 868 | 0.172 | 0.032 | — | 2.2 | 2.1 | 1.1 |  |

### Qwen3-TTS 1.7B Base（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 7813 | 1.878 | 0.002 | — | 16.0 | 4.7 | 3.4 | flash-attn なし |
| audio.cpp | 679 | 0.160 | 0.008 | — | 3.7 | 3.3 | 4.5 |  |
| CrispASR | 756 | 0.185 | 0.008 | — | 2.2 | 3.2 | 1.1 |  |

### Qwen3-TTS 1.7B CustomVoice（Ono_Anna）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 8589 | 1.870 | 0.054 | — | 16.2 | 4.3 | 3.8 | flash-attn なし |
| speech.cpp（Vulkan） | 620 | 0.141 | 0.045 | — | 2.7 | 2.3 | 0.2 |  |
| audio.cpp | 634 | 0.147 | 0.019 | — | 3.4 | 2.8 | 4.0 |  |
| CrispASR | 820 | 0.182 | 0.091 | — | 2.8 | 3.1 | 1.1 |  |

### OmniVoice

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| omnivoice（PyTorch） | 998 | 0.264 | 0.010 | — | 8.2 | 2.4 | 2.1 |  |
| audio.cpp | 363 | 0.098 | 0.000 | — | 3.0 | 1.6 | 1.5 |  |
| CrispASR | 508 | 0.135 | 0.006 | — | 1.5 | 1.6 | 0.7 |  |

### CosyVoice3 0.5B（クローン・カタカナで入力）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CosyVoice（公式・PyTorch） | 3337 | 0.827 | 0.008 | — | 16.1 | 3.9 | 5.8 |  |
| CrispASR | 450 | 0.113 | 0.016 | — | 4.5 | 1.8 | 0.8 | LLM は Q4_K |

### MOSS-TTS-Nano（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MOSS-TTS-Nano（PyTorch） | 2499 | 0.306 | 0.142 | 2472 | — | — | — |  |
| audio.cpp | 229 | 0.049 | 0.200 | 836 | 1.0 | 0.8 | 0.8 |  |

### MOSS-TTS-Local v1.5（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 9766 | 2.432 | 0.030 | — | 312.6 | 11.6 | 10.7 | GPU メモリが 12 GB にほぼ届く |
| CrispASR | 1364 | 0.362 | 0.006 | — | 4.6 | 9.0 | 5.6 |  |

### VoxCPM2（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 780 | 0.235 | 0.075 | — | 3.3 | 3.4 | 3.4 |  |
| CrispASR | 1368 | 0.406 | 0.014 | — | 2.7 | 4.1 | 3.6 |  |

### MioTTS

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mio-tts-cpp（0.1B） | 796 | 0.186 | 0.016 | — | 4.9 | 1.3 | 1.2 |  |
| CrispASR | 1637 | 0.325 | 0.068 | — | 1.6 | 1.5 | 0.9 | 0.6B・温度 0.8 |
| audio.cpp | 806 | 0.194 | 0.008 | — | 3.6 | 3.9 | 1.8 | 1.7B・クローン |

### Confucius4-TTS（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 2568 | 0.692 | 0.418 | — | 9.4 | 7.1 | 4.3 |  |
| CrispASR | 28309 | 7.460 | 0.432 | — | 16.6 | 3.1 | 1.7 |  |

### Fish Audio S2 Pro（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 2304 | 0.666 | 0.006 | — | 8.9 | 6.9 | 7.4 |  |

### Higgs Audio TTS 3 4B（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 1292 | 0.329 | 0.006 | — | 6.3 | 5.0 | 5.4 |  |

### OuteTTS 1.0 1B（クローン）

| 実行系 | 1 文の平均 ms | RTF | 読みの CER | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 23056 | 3.255 | 0.110 | — | 7.7 | 6.6 | 2.6 |  |

<!-- tts-tables:end -->
