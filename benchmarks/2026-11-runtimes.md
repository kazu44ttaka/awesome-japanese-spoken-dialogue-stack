# 同じモデルを実行系ごとに比べる（2026-11）

同じモデルでも、どの実行系（推論のライブラリ）で動かすかで、速さ・精度・メモリが大きく変わる。このページでは、
日本語の ASR 18 モデル・TTS 22 モデルを、元の実装（PyTorch など）・[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)・[faster-whisper](https://github.com/SYSTRAN/faster-whisper) と、ggml 系の
3 つの実行系 [CrispASR](https://github.com/CrispStrobe/CrispASR)・[audio.cpp](https://github.com/0xShug0/audio.cpp)・
[speech.cpp](https://github.com/nyosegawa/speech.cpp) で、同じ PC・同じ入力で測って並べた。

- 実行系の入れ方とこの PC で動かすときの注意：[docs/runtimes.md](../docs/runtimes.md)。モデルの名前と立ち上げ方は
  [runtimes/catalog.py](../runtimes/catalog.py)
- PC：[benchmarks/README.md](README.md) のもの（RTX 5070 Ti Laptop・Core Ultra 9 275HX）。AC 電源・電源プラン「高パフォーマンス」
- 測定日：2026-11-17〜22。1 回だけの測定なので、数 % の差は気にしない
- 表はこのファイルの [make_runtime_page.py](make_runtime_page.py) で結果の JSON から作っている。「—」は測っていない（その実行系に無い、
  または動かなかった）ところ

## 要点

- **一番速い実行系はモデルごとに違う。** LLM 型のモデル（Qwen3-ASR・Qwen3-TTS・CosyVoice3 など）は、GPU なら ggml の実行系で元の実装の数倍〜十数倍速い
- **精度はほとんどの実行系でそろう**（CER の差は 0.01 以内が多い）。例外は CrispASR の NeMo 系で、CER が 0.07〜0.13 悪い
- **CPU だけなら sherpa-onnx（int8）が速いことが多い**
- **音声対話の待ち時間は、速いモデルの疑似ストリーミングがいちばん短い。** 音素を話してから文字が出るまで 約 60〜130 ms で、真のストリーミング（0.5〜0.8 秒）より速い
- **TTS の最初の声は、逐次に音を作るモデルなら 100 ms 以下**（Qwen3-TTS は 32〜92 ms）
- **LLM は実行系による差が小さい。** ただし公式の llama.cpp は、最初の文字まで数秒かかる回があった

このページの中身：[測り方](#測り方)・[わかったこと](#わかったこと)・[ASR](#asr)・[TTS](#tts)・[ストリーミング](#ストリーミング)・[LLM](#llm)

## 測り方

<details><summary>開く</summary>

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
- 最初の声：同じ 20 文をストリーミングで要求し、振幅が 1% を超える音が届くまでの中央値（先頭の無音を除く。[下](#tts-1)）
- UTMOS：自然さの推定（UTMOS22 strong、1〜5）の 20 文の平均（[tts/scripts/quality_tts.py](tts/scripts/quality_tts.py)）。
  英語の読み上げで学習した推定器なので、日本語では値そのものより比べ合いに使う
- 話者の近さ：参照音声との話者ベクトル（microsoft/wavlm-base-plus-sv）のコサイン類似度。参照音声でクローンするモデルだけ

</details>

## わかったこと

<details><summary>開く</summary>

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
  - ※1 CrispASR の NeMo 系（Haas・ReazonSpeech・Parakeet）は、発話の前後に小さく入った別の人の声まで書き起こし、CER が 0.07〜0.13 悪い。
    F16 の重みでも、VAD の区切りを止めても同じなので、実装の違いとみられる
  - ※2 CrispASR 向けに配布されている Parakeet CTC 1.1B ja の GGUF は、元のリポジトリのギャル語版から変換されたとみられる
    （出力がギャル語版の PyTorch とよく似ていて、CER も近い）
  - ※3 Granite 4.1 2B・LFM2.5-Audio JP（CrispASR）は、同じ句をくり返して止まらない発話がある。LFM2.5-Audio JP は audio.cpp では繰り返さず、
    読みの CER が 0.037 / 0.249 / 0.035・会話 0.188 と、CrispASR（0.347 / 0.794 / 0.281・会話 0.650）よりずっとよい
  - TTS の読みの CER は、実行系で 0.01〜0.07 ずれるものがある。Irodori は audio.cpp（0.006・0.008）より speech.cpp（0.028・0.035）が悪く、
    VoxCPM2・MOSS-TTS-Local は逆に CrispASR（0.011・0.006）より audio.cpp（0.068・0.029）が悪い
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
- **音声対話の待ち時間は、速いモデルの疑似ストリーミングがいちばん短かった。**（[下の表](#asr-1)）
  - JSUT で音素を話し終えてから文字が出るまでの中央値は、デモアプリの疑似ストリーミング（音声 0.25 秒ごとに区間の頭から
    認識し直す）で Parakeet TDT-CTC 62〜79 ms、ReazonSpeech k2（先頭に無音を足す）87 ms、Haas 94〜129 ms
  - 真のストリーミングの Nemotron は 477〜839 ms、Voxtral（audio.cpp）は 763 ms。チャンクとモデルの先読みの分だけ遅い。
    読みの CER も、疑似ストリーミングの Parakeet・Haas（0.010〜0.017）のほうが Nemotron（0.048〜0.070）よりよい
  - ただし疑似ストリーミングは、1 回の認識が 250 ms 以内の速いモデルでしか成り立たない。この PC の GPU を 1 人の音声で使った値
  - speech.cpp の Realtime（0.2 秒ごとに頭から認識し直す）は、同じモデルでもアプリの疑似ストリーミングより遅い
    （Haas 490 ms と 94 ms）。2 回続けて同じだった頭の部分だけを途中経過として出すため
- **話し終わってから最終的な文字が出るまでは、話し終わりがすぐわかれば 15〜50 ms。**
  `timing100` で音声の終わりにすぐ「終わり」を送ると、audio.cpp の Voxtral 15 ms・Nemotron 18 ms、speech.cpp の Parakeet 33 ms・
  Haas 35 ms。実際には話し終わりの判定が要るので、JSUT（VAD が無音 0.1 秒を聞いて判定する疑似ストリーミング、または最後の音素の
  0.3 秒あとに「終わり」を送るストリーミング）ではどれも 0.3 秒前後になる
- **途中経過を出さない「ストリーミング」がある。** CrispASR の Realtime は、途中経過を出すのが Nemotron だけで、Haas・Qwen3-ASR・
  SenseVoice は「終わり」を送ってから 1 回認識するだけ。Qwen3-ASR は 34 発話中 16 発話（JSUT では 60 発話中 29 発話）で確定の文字が返らなかった。サーバーが処理の例外を記録せずに捨てているとみられる（[docs/runtimes.md](../docs/runtimes.md)）。audio.cpp の
  Qwen3-ASR・SenseVoice も、既定の窓（Qwen3-ASR は 30 秒）では短い発話の途中経過が出ない。audio.cpp の Haas は 2 秒の窓なので、最初の文字は 2 秒あとになる
- **TTS の最初の声は、逐次に音を作るモデルなら 100 ms 以下。**
  - Qwen3-TTS は speech.cpp で 32〜81 ms、audio.cpp で 71〜92 ms、CrispASR で 114〜123 ms。audio.cpp の MOSS-TTS-Nano 18 ms・
    VoxCPM2 58 ms・Fish S2 Pro 91 ms
  - Irodori は 1 文を作ってから区切って送るので、speech.cpp の MF 版の 57 ms を除き 0.13〜1.7 秒。CrispASR は Qwen3-TTS 以外を
    1 文ずつ作ってから送る
  - 元の実装では、CosyVoice3（`stream=True`）が 2.7 秒、GPT-SoVITS（`streaming_mode`）が 0.38 秒
- **音質の推定（UTMOS）は、同じモデルなら実行系でほとんど変わらない。** Qwen3-TTS 1.7B Base は PyTorch 3.98・audio.cpp 4.00・
  CrispASR 3.91。高いのは Supertonic 3（4.1〜4.2）、次が Qwen3-TTS Base・Higgs・GPT-SoVITS・Fish・AivisSpeech（3.8〜4.0）。
  参照音声そのもの（Common Voice の録音）は 3.52。英語の読み上げで学習した推定器なので、日本語では目安
  - 話者の近さは、クローンしたモデルで 0.90〜0.97。ただし、クローンしない AivisSpeech の既定の声も 0.93 になったので、
    この指標の差は小さい。CrispASR の MOSS-TTS-Local だけ 0.70 と低い（audio.cpp は 0.95）
- **Style-Bert-VITS2 は速く読みも正確**（GPU で 1 文 114 ms・読みの CER 0.002）だが、例の声（JVNV）は UTMOS 2.61 と低め。
  GPT-SoVITS（v2ProPlus）は学習なしのクローンで UTMOS 3.88・話者の近さ 0.96 だが、1 文 2.1 秒と遅い
- **LLM は、実行系による差が小さい。**（[下の表](#llm)）TTFT の中央値は 30〜90 ms、tok/s の差は多くが 1 割以内。
  ただし公式の llama.cpp は、TTFT が 0.3〜5.8 秒になる回があった（LM Studio は最悪でも 0.11 秒）
- **使い方を間違えると、モデルが悪く見える。**
  - CosyVoice3 は、公式の例にあるとおり日本語をカタカナで渡す必要がある（参照音声の書き起こしも）。漢字のまま渡すと読みが崩れる
    （CrispASR で CER 0.44、カタカナで 0.013。公式の PyTorch でもカタカナで 0.008）
  - CrispASR の LLM 型 TTS は温度の既定が 0（貪欲）で、MioTTS 0.6B は 20 文中 3 文が止まらずに 30 秒まで伸びた。モデルの既定の 0.8 にして測った
- **2026-11-21〜22 に足した構成**（実行系は対応しているのに、構成に入れていなかったもの）
  - TTS は audio.cpp で大きく速くなるものが多い。LFM2.5-Audio JP は 2.5 秒 → 0.33 秒（最初の声 51 ms）、Kitten TTS 2 は 3.1 秒 → 0.91 秒、
    ZONOS3-Mini は 270 ms → 225 ms（最初の声 88 ms）。読みの CER はどれも元とほぼ同じ
  - Qwen3-TTS 1.7B VoiceDesign（声を文章で指定）は、audio.cpp で 1 文 0.71 秒・最初の声 25 ms・読みの CER 0.004。CrispASR は CER 0.024
  - Moonshine の CrispASR 版（作者の日本語の GGUF）は、GPU で transformers の 3 倍ほど速い（tiny 70 ms・base 128 ms）が、
    ReazonSpeech で悪い（tiny 0.110・base 0.086。transformers は 0.072・0.044）。CrispASR は Moonshine を英語専用として ja を断るので、言語は auto で渡す
  - Granite 4.1 の Plus 版は 2B と同じくらい（会話は 0.158 と少しよい）。NAR 版は日本語に対応しておらず、ローマ字の断片を出す
- **CPU で測り足したもの**：NeMo の Parakeet TDT-CTC 360 ms・Parakeet CTC 1.1B 593 ms・ReazonSpeech NeMo v2 666 ms、
  Qwen3-ASR 1.7B は speech.cpp 2.3 秒・PyTorch 4.9 秒。TTS は mio-tts-cpp 736 ms（0.1B と小さく GPU と同じくらい）・Style-Bert-VITS2 1.3 秒・
  AivisSpeech 1.8 秒（エンジンはスレッド数を指定できないので既定のまま）。読みの CER はどれも GPU とほぼ同じ
- 測らなかったもの：IndexTTS2 / 2.5（日本語に非対応。同じ音のくり返しになる）、ZONOS v0.1 と Kokoro（CrispASR で日本語を読めない・
  声のファイルが別に要る。audio.cpp の Kokoro も、配布の GGUF に日本語の辞書が入っておらず読めない）、speech.cpp の Qwen3-TTS のクローン
  （speech.cpp の 0.6B は CustomVoice 版で、録音から声を作れない。Base 版は配布されていない）、VibeVoice-ASR（長い録音向けで、短い発話では英語の文などを出す。audio.cpp では時間切れ）、
  audio.cpp の Granite Speech 5.0 TurboCTC（日本語に対応する前の英語だけの重み）

</details>

## ASR

<!-- asr-tables:start -->
### Haas-ja v1.0

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| haas-asr（PyTorch・NeMo） | 42 | — | 0.019 | 0.025 | 0.038 | 0.007 | 15.2 | 3.7 | 3.4 |  |
| speech.cpp（Vulkan） | 43 | 314 | 0.019 | — | — | — | 0.5 | 0.8 | 0.2 |  |
| audio.cpp | 42 | 552 | 0.018 | — | — | — | 2.7 | 2.0 | 2.2 |  |
| CrispASR | 176 | 659 | 0.093 | 0.078 | 0.138 | 0.018 | 1.0 | 1.0 | 0.7 | ※1 |

### ReazonSpeech NeMo v2

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 348 | 666 | 0.024 | 0.038 | 0.039 | 0.033 | 15.8 | 4.9 | 3.3 |  |
| speech.cpp（Vulkan） | 98 | 506 | 0.025 | 0.039 | 0.038 | 0.033 | 0.5 | 0.8 | 0.2 |  |
| CrispASR | 142 | 842 | 0.104 | 0.090 | 0.154 | 0.075 | 1.0 | 1.0 | 0.8 | ※1 |

### Parakeet TDT-CTC 0.6B ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 54 | 360 | 0.036 | 0.038 | 0.092 | 0.023 | 19.1 | 5.1 | 3.4 |  |
| sherpa-onnx（int8） | — | 180 | 0.035 | 0.042 | 0.087 | 0.029 | 2.4 | 0.7 | 1.4 |  |
| speech.cpp（Vulkan） | 42 | 310 | 0.037 | 0.038 | 0.092 | 0.023 | 0.5 | 0.7 | 0.2 |  |
| CrispASR | 77 | 649 | 0.107 | 0.086 | 0.178 | 0.034 | 1.1 | 1.0 | 0.8 | ※1 |

### Parakeet CTC 1.1B ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 68 | 593 | 0.058 | 0.053 | 0.084 | 0.039 | 26.5 | 8.3 | 4.9 |  |
| CrispASR | 67 | 67 | 0.186 | 0.171 | 0.327 | 0.112 | 1.0 | 1.7 | 1.3 | ※2 |

### Qwen3-ASR 0.6B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-asr（PyTorch） | 716 | 1600 | 0.101 | 0.050 | 0.208 | 0.059 | 12.8 | 2.0 | 2.5 |  |
| sherpa-onnx（int8） | — | 916 | 0.120 | 0.071 | 0.229 | 0.070 | 3.7 | 3.3 | 2.4 |  |
| speech.cpp（Vulkan） | 95 | 826 | 0.104 | — | — | — | 1.0 | 1.4 | 0.3 |  |
| audio.cpp | 210 | 1228 | 0.106 | — | — | — | 1.2 | 2.3 | 0.9 |  |
| CrispASR | 248 | 1080 | 0.103 | — | — | — | 1.0 | 2.5 | 1.1 |  |

### Qwen3-ASR 1.7B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-asr（PyTorch） | 932 | 4869 | 0.089 | 0.029 | 0.182 | 0.036 | 14.0 | 4.6 | 4.1 |  |
| speech.cpp（Vulkan） | 165 | 2320 | 0.086 | 0.029 | 0.182 | 0.036 | 1.5 | 2.6 | 0.2 |  |
| audio.cpp | 280 | 2876 | 0.085 | — | — | — | 2.7 | 3.5 | 1.9 |  |
| CrispASR | 274 | 2759 | 0.088 | — | — | — | 2.5 | 4.2 | 2.5 |  |

### Qwen3-ASR 1.7B ja-anime

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 274 | 2741 | 0.093 | 0.036 | 0.202 | 0.037 | 2.5 | 4.2 | 2.5 |  |

### Fun-ASR-Nano-2512

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| funasr（PyTorch） | 640 | 1727 | 0.061 | 0.033 | 0.125 | 0.034 | 38.9 | 3.4 | 4.3 |  |
| sherpa-onnx | 902 | 851 | 0.073 | 0.044 | 0.190 | 0.044 | 7.1 | 6.4 | 1.6 |  |
| audio.cpp | 267 | 1286 | 0.080 | 0.036 | 0.252 | 0.035 | 3.7 | 2.1 | 2.1 |  |
| CrispASR | 134 | 798 | 0.056 | — | — | — | 1.0 | 1.7 | 1.1 |  |

### Fun-ASR-MLT-Nano-2512

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| funasr（PyTorch） | 639 | 1755 | 0.028 | 0.013 | 0.058 | 0.022 | 39.0 | 3.5 | 4.3 |  |
| CrispASR | 159 | 1277 | 0.042 | 0.012 | 0.062 | 0.026 | 2.5 | — | — | GPU では 2 回目の要求で落ちる（2026-11-20）。3 データは CPU で測った |

### SenseVoiceSmall

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sherpa-onnx | 41 | 89 | 0.090 | 0.049 | 0.183 | 0.044 | 1.7 | 2.2 | 1.3 |  |
| audio.cpp | 99 | 178 | 0.108 | 0.048 | 0.200 | 0.043 | 0.7 | 0.5 | 0.7 |  |
| CrispASR | 59 | 232 | 0.090 | — | — | — | 0.5 | 0.5 | 0.6 |  |

### Cohere Transcribe

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sherpa-onnx（int8） | — | 780 | 0.020 | 0.023 | 0.048 | 0.017 | 6.7 | 2.7 | 3.0 |  |
| audio.cpp | 102 | 1472 | 0.021 | — | — | — | 2.7 | 3.6 | 3.1 |  |
| CrispASR | 189 | 808 | 0.020 | — | — | — | 1.0 | 1.8 | 1.3 | Q4_K |

### Cohere ASR ja v0.1

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 188 | 1766 | 0.051 | 0.025 | 0.092 | 0.018 | 1.5 | 2.8 | 2.5 |  |

### Granite 4.0 1B Speech

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 1166 | 5156 | 0.043 | 0.036 | 0.095 | 0.017 | 9.6 | 4.6 | 4.8 |  |
| CrispASR | 460 | 2657 | 0.053 | 0.057 | 0.081 | 0.072 | 2.5 | 3.3 | 2.6 | Q4_K |

### Granite 4.1 2B Speech

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 480 | 2785 | 0.172 | 0.060 | 0.085 | 0.059 | 2.5 | 3.3 | 2.7 | Q4_K・※3 |
| CrispASR | 450 | 2529 | 0.068 | 0.058 | 0.084 | 0.062 | 2.5 | 3.3 | 2.3 | Plus 版 |
| CrispASR | 332 | 2272 | 0.999 | 0.998 | 0.995 | 0.957 | 2.5 | 3.5 | 2.7 | NAR 版。日本語に非対応（ローマ字の断片を出す） |

### Voxtral Mini 4B Realtime

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 5181 | — | 0.139 | 0.066 | 0.187 | 0.074 | 12.2 | 9.0 | 8.7 |  |
| audio.cpp | 1331 | — | 0.124 | 0.068 | 0.268 | 0.085 | 3.2 | 5.3 | 4.2 |  |
| CrispASR | 1349 | — | 0.124 | 0.064 | 0.189 | 0.074 | 3.0 | 5.6 | 4.2 |  |

### Nemotron 3.5 ASR streaming 0.6B

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NeMo | 475 | 1986 | 0.105 | 0.067 | 0.203 | 0.099 | 49.8 | 2.7 | 5.9 |  |
| sherpa-onnx（int8・560 ms） | — | 760 | 0.124 | 0.084 | 0.210 | 0.098 | 2.0 | 0.3 | 1.4 |  |
| sherpa-onnx（int8・80 ms） | — | 4065 | 0.127 | 0.093 | 0.230 | 0.115 | — | — | — |  |
| audio.cpp | 55 | 975 | 0.137 | 0.070 | 0.216 | 0.105 | 1.7 | 1.3 | 1.1 |  |
| CrispASR | 342 | 2360 | 0.130 | 0.070 | 0.217 | 0.105 | 1.0 | 1.8 | 1.3 | F16 |

### Whisper large-v3-turbo

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 334 | 6513 | 0.039 | 0.024 | 0.073 | 0.028 | 6.1 | 2.2 | 2.3 |  |
| faster-whisper（CTranslate2） | 180 | 3470 | 0.058 | 0.025 | 0.103 | 0.029 | 1.8 | 2.3 | 1.0 |  |
| CrispASR | 195 | 10514 | 0.052 | 0.026 | 0.077 | 0.030 | 1.5 | 2.0 | 0.6 | whisper.cpp の ggml |

### LFM2.5-Audio-1.5B-JP

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CrispASR | 196 | — | 0.332 | 0.347 | 0.794 | 0.281 | 1.5 | 2.3 | 1.4 | Q5_K・※3 |
| audio.cpp | 143 | 1273 | 0.080 | 0.037 | 0.249 | 0.035 | 1.2 | 1.7 | 0.6 | Q8_0。CrispASR のような繰り返しは出ない |

### Moonshine tiny-ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 235 | 187 | 0.060 | — | — | — | 6.6 | 0.3 | 1.6 |  |
| CrispASR | 70 | 100 | 0.069 | 0.082 | 0.110 | 0.057 | 0.5 | 0.2 | 0.6 | F16・言語は auto で渡す |

### Moonshine base-ja

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 321 | 330 | 0.065 | 0.070 | 0.044 | 0.035 | 6.6 | 0.3 | 1.6 |  |
| CrispASR | 128 | 195 | 0.052 | 0.069 | 0.086 | 0.042 | 0.5 | 0.3 | 0.7 | F16・言語は auto で渡す |

### anime-whisper

| 実行系 | GPU 平均 ms | CPU 平均 ms | 読みの CER（timing100） | FLEURS | ReazonSpeech | Common Voice | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| transformers | 225 | 4718 | 0.080 | 0.043 | 0.191 | 0.036 | 8.3 | 2.0 | 3.3 |  |
| faster-whisper（有志の CTranslate2 変換） | 159 | 3394 | 0.087 | 0.054 | 0.191 | 0.039 | 2.5 | 2.2 | 1.1 |  |
| CrispASR | 155 | 4628 | 0.146 | 0.087 | 0.278 | 0.114 | 1.0 | 1.2 | 0.6 | 有志の whisper.cpp 変換 |

<!-- asr-tables:end -->

## TTS

<!-- tts-tables:start -->
### Supertonic 3

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| supertonic（ONNX・CPU 4 スレッド） | — | — | — | 0.017 | 4.18 | — | 864 | 2.9 | 0.0 | 0.5 |  |
| audio.cpp | 91 | 69 | 0.023 | 0.022 | 4.18 | — | 662 | 1.4 | 0.6 | 0.7 |  |
| CrispASR | 191 | 157 | 0.047 | 0.017 | 4.09 | — | 877 | 0.8 | 0.5 | 0.7 |  |

### Irodori-TTS v4 Small

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 571 | 439 | 0.127 | 0.028 | 3.36 | — | — | 1.8 | 2.2 | 0.2 |  |
| audio.cpp | 394 | 302 | 0.092 | 0.006 | 3.52 | — | 14887 | 3.1 | 1.5 | 1.9 |  |

### Irodori-TTS v4.1 Small

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Irodori-TTS（PyTorch） | 1334 | — | 0.289 | 0.000 | 3.45 | — | — | 12.0 | 4.2 | 6.9 | キャプションで声を決める |

### Irodori-TTS v4.1 Small MF

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 165 | 57 | 0.037 | 0.035 | 3.26 | — | 4381 | 1.6 | 2.1 | 0.2 |  |
| audio.cpp | 207 | 125 | 0.048 | 0.007 | 3.13 | — | 4547 | 2.9 | 2.7 | 1.7 |  |

### Irodori-TTS v4.1 Anime

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 390 | 305 | 0.092 | 0.002 | 2.90 | — | — | 3.0 | 1.5 | 1.6 |  |

### Irodori-TTS Large

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| speech.cpp（Vulkan） | 1304 | 1054 | 0.295 | 0.022 | 3.53 | — | — | 3.3 | 4.0 | 0.2 |  |

### Irodori-TTS 500M v3

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 351 | 263 | 0.084 | 0.004 | 3.52 | — | — | 3.0 | 2.4 | 1.5 |  |
| CrispASR | 1800 | 1718 | 0.434 | 0.015 | 3.32 | — | — | 2.1 | 2.3 | 0.9 |  |

### Irodori-TTS 600M v3 VoiceDesign

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 366 | 280 | 0.085 | 0.004 | 3.51 | — | — | 3.0 | 2.5 | 1.6 |  |
| CrispASR | 1887 | 1769 | 0.447 | 0.009 | 3.35 | — | — | 2.2 | 2.5 | 1.0 | Q4_K |

### Qwen3-TTS 0.6B Base（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 8753 | — | 1.886 | 0.008 | 3.90 | 0.95 | — | 14.7 | 2.9 | 2.5 | flash-attn なし |
| audio.cpp | 612 | 82 | 0.132 | 0.002 | 4.02 | 0.95 | 6571 | 3.7 | 2.6 | 3.7 |  |
| CrispASR | 1036 | 114 | 0.237 | 0.010 | 3.80 | 0.94 | 3592 | 2.3 | 2.4 | 1.1 |  |

### Qwen3-TTS 0.6B CustomVoice（Ono_Anna）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 9070 | — | 1.869 | 0.009 | 3.67 | — | — | 14.6 | 2.7 | 2.5 | flash-attn なし |
| speech.cpp（Vulkan） | 604 | 81 | 0.115 | 0.019 | 3.54 | — | 3683 | 1.8 | 1.3 | 0.2 |  |
| CrispASR | 868 | 114 | 0.172 | 0.027 | 3.56 | — | 3892 | 2.2 | 2.1 | 1.1 |  |

### Qwen3-TTS 1.7B Base（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 7813 | — | 1.878 | 0.002 | 3.98 | 0.96 | — | 16.0 | 4.7 | 3.4 | flash-attn なし |
| audio.cpp | 679 | 92 | 0.160 | 0.006 | 4.00 | 0.97 | — | 3.7 | 3.3 | 4.5 |  |
| CrispASR | 756 | 123 | 0.185 | 0.006 | 3.91 | 0.95 | — | 2.2 | 3.2 | 1.1 |  |

### Qwen3-TTS 1.7B CustomVoice（Ono_Anna）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen-tts（PyTorch） | 8589 | — | 1.870 | 0.050 | 3.25 | — | — | 16.2 | 4.3 | 3.8 | flash-attn なし |
| speech.cpp（Vulkan） | 620 | 32 | 0.141 | 0.043 | 3.27 | — | — | 2.7 | 2.3 | 0.2 |  |
| audio.cpp | 634 | 71 | 0.147 | 0.019 | 3.27 | — | — | 3.4 | 2.8 | 4.0 |  |
| CrispASR | 820 | 118 | 0.182 | 0.087 | 3.31 | — | — | 2.8 | 3.1 | 1.1 |  |

### OmniVoice

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| omnivoice（PyTorch） | 998 | — | 0.264 | 0.009 | 3.45 | — | — | 8.2 | 2.4 | 2.1 |  |
| audio.cpp | 363 | 306 | 0.098 | 0.000 | 3.46 | — | — | 3.0 | 1.6 | 1.5 |  |
| CrispASR | 508 | 445 | 0.135 | 0.006 | 3.56 | — | — | 1.5 | 1.6 | 0.7 |  |

### CosyVoice3 0.5B（クローン・カタカナで入力）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CosyVoice（公式・PyTorch） | 3337 | 2675 | 0.827 | 0.007 | 3.60 | 0.96 | — | 16.1 | 3.9 | 5.8 |  |
| CrispASR | 450 | 399 | 0.113 | 0.013 | 3.39 | 0.90 | — | 4.5 | 1.8 | 0.8 | LLM は Q4_K |

### MOSS-TTS-Nano（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MOSS-TTS-Nano（PyTorch） | 2499 | — | 0.306 | 0.137 | 1.50 | 0.43 | 2472 | — | — | — |  |
| audio.cpp | 229 | 18 | 0.049 | 0.191 | 3.49 | 0.94 | 836 | 1.0 | 0.8 | 0.8 |  |

### MOSS-TTS-Local v1.5（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 9766 | — | 2.432 | 0.029 | 3.80 | 0.95 | — | 312.6 | 11.6 | 10.7 | GPU メモリが 12 GB にほぼ届く |
| CrispASR | 1364 | 1286 | 0.362 | 0.006 | 3.85 | 0.70 | — | 4.6 | 9.0 | 5.6 |  |

### VoxCPM2（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 780 | 58 | 0.235 | 0.068 | 3.44 | 0.94 | — | 3.3 | 3.4 | 3.4 |  |
| CrispASR | 1368 | 1327 | 0.406 | 0.011 | 3.66 | 0.96 | — | 2.7 | 4.1 | 3.6 |  |

### MioTTS

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mio-tts-cpp（0.1B） | 796 | — | 0.186 | 0.017 | 2.23 | — | 736 | 4.9 | 1.3 | 1.2 |  |
| CrispASR | 1637 | 1342 | 0.325 | 0.065 | 1.24 | — | — | 1.6 | 1.5 | 0.9 | 0.6B・温度 0.8 |
| audio.cpp | 806 | — | 0.194 | 0.006 | 3.28 | — | — | 3.6 | 3.9 | 1.8 | 1.7B・クローン |

### Confucius4-TTS（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 2568 | 2338 | 0.692 | 0.391 | 2.84 | 0.96 | — | 9.4 | 7.1 | 4.3 |  |
| CrispASR | 28309 | 26715 | 7.460 | 0.414 | 2.87 | 0.95 | — | 16.6 | 3.1 | 1.7 |  |

### Fish Audio S2 Pro（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 2304 | 91 | 0.666 | 0.004 | 3.85 | 0.96 | — | 8.9 | 6.9 | 7.4 |  |

### Higgs Audio TTS 3 4B（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 1292 | 205 | 0.329 | 0.004 | 4.01 | 0.95 | — | 6.3 | 5.0 | 5.4 |  |

### OuteTTS 1.0 1B（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 23056 | — | 3.255 | 0.104 | 3.55 | 0.96 | — | 7.7 | 6.6 | 2.6 |  |

### Style-Bert-VITS2 JP-Extra

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Style-Bert-VITS2（PyTorch） | 114 | — | 0.032 | 0.002 | 2.61 | — | 1263 | 8.4 | 1.9 | 2.0 | 声は jvnv-F1-jp |
| AivisSpeech（ONNX） | 408 | — | 0.106 | 0.004 | 3.85 | — | 1839 | 19.4 | 1.4 | 1.1 | 同じ系統。声は別のモデル（既定の声） |

### GPT-SoVITS v2ProPlus（クローン）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GPT-SoVITS（PyTorch） | 2084 | 382 | 0.458 | 0.011 | 3.88 | 0.96 | 4036 | 12.2 | 2.1 | 3.8 |  |

### Qwen3-TTS 1.7B VoiceDesign（声を文章で指定）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| audio.cpp | 710 | 25 | 0.146 | 0.004 | 3.54 | — | — | 4.0 | 2.9 | 4.0 | 説明：落ち着いた若い女性の声 |
| CrispASR | 811 | — | 0.182 | 0.024 | 3.40 | — | — | 3.0 | 3.8 | 2.2 | 同じ説明 |

### Kitten TTS 2

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kittenml（PyTorch） | 3065 | — | 0.722 | 0.013 | 1.60 | — | 12185 | 12.2 | 4.8 | 5.2 | JapaneseAnimeHeroine |
| audio.cpp | 910 | — | 0.212 | 0.011 | 1.60 | — | — | 5.7 | 3.3 | 1.3 | 同じ声。ストリーミングは非対応 |

### LFM2.5-Audio-1.5B-JP（TTS）

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| liquid-audio（PyTorch） | 2535 | — | 0.707 | 0.006 | 3.72 | — | — | 8.3 | 6.1 | 3.4 |  |
| audio.cpp | 328 | 51 | 0.092 | 0.007 | 3.63 | — | — | 1.5 | 1.7 | 0.6 |  |

### ZONOS3-Mini

| 実行系 | 1 文の平均 ms | 最初の声 ms | RTF | 読みの CER | UTMOS | 話者の近さ | CPU 1 文の平均 ms | 読み込み 秒 | GPU メモリ GB | RAM GB | 注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ZONOS3 の ggml サーバー | 270 | — | 0.064 | 0.017 | 3.26 | — | — | 5.2 | 2.0 | 0.8 |  |
| audio.cpp | 225 | 88 | 0.053 | 0.021 | 3.38 | — | — | 2.5 | 1.4 | 0.7 |  |

<!-- tts-tables:end -->

## ストリーミング

音声対話では、ASR は **話し終えてから文字が確定するまで**、TTS は **要求してから最初の声が届くまで** が待ち時間になる。
ASR は `timing100` から 3 発話おきの 34 発話を 80 ms ずつ実時間で送り（[asr/scripts/stream_asr.py](asr/scripts/stream_asr.py)）、
TTS は上と同じ 20 文をストリーミングで要求した（[tts/scripts/stream_tts.py](tts/scripts/stream_tts.py)）。実行系ごとの呼び方は
[runtimes/streaming.py](../runtimes/streaming.py)。

### ASR

**話し終わってから文字が出るまで**（`timing100` の 34 発話）。利用者から見た待ち時間は、どちらの使い方でも「話し終わってから
最終的な文字が出るまで」:
- 録音をまとめて渡す（ストリーミングなし）：話し終わってから、録音した発話全体を認識する。時間は上の表の GPU 平均 ms
- 話しながら送る（ストリーミング）：話している間に 80 ms ずつ送り、その間に途中の文字も出る。話し終わった時点ではほとんど
  認識が済んでいるので、残りの分だけで最終的な文字が出る。音声の終わりで「終わり」を送る（話し終わりがすぐわかる場合）
- 「話している途中に文字が出た割合」は、話し終わる前に途中の文字が 1 回でも届いた発話の割合（0% ならストリーミングになっていない）

<!-- stream-asr-tables:start -->
| モデル | 実行系 | 話し終わってから文字が出るまで ms：話しながら送る（中央値 / p95） | 話している途中に文字が出た割合 | 読みの CER | 比べる：録音をまとめて渡す ms | 注 |
|---|---|---:|---:|---:|---:|---|
| Nemotron 3.5 ASR streaming 0.6B | sherpa-onnx（int8・80 ms・CPU） | 165 / 189 | 91% | 0.104 | — | 真のストリーミング |
| Nemotron 3.5 ASR streaming 0.6B | sherpa-onnx（int8・560 ms・CPU） | 59 / 121 | 91% | 0.097 | — | 真のストリーミング |
| Nemotron 3.5 ASR streaming 0.6B | NeMo（560 ms・GPU） | 34 / 44 | 91% | 0.073 | 475 | 真のストリーミング |
| Nemotron 3.5 ASR streaming 0.6B | audio.cpp | 18 / 23 | 91% | 0.107 | 55 | 真のストリーミング |
| Nemotron 3.5 ASR streaming 0.6B | CrispASR | 66 / 94 | 91% | 0.168 | 342 | 真のストリーミング |
| Voxtral Mini 4B Realtime | audio.cpp | 15 / 30 | 88% | 0.184 | 1331 | 80 ms ずつ |
| Haas-ja v1.0 | audio.cpp | 48 / 62 | 71% | 0.015 | 42 | 2 秒の窓で認識し直す（右に 2 秒待つ） |
| Haas-ja v1.0 | speech.cpp（Vulkan） | 35 / 57 | 97% | 0.014 | 43 | 0.2 秒ごとに頭から認識し直す |
| Haas-ja v1.0 | CrispASR | 110 / 252 | 0% | 0.040 | 176 | 途中経過なし（終わりを送ってから 1 回認識） |
| ReazonSpeech NeMo v2 | speech.cpp（Vulkan） | 76 / 170 | 94% | 0.023 | 98 | 0.2 秒ごとに頭から |
| Parakeet TDT-CTC 0.6B ja | speech.cpp（Vulkan） | 33 / 52 | 97% | 0.033 | 42 | 0.2 秒ごとに頭から |
| Qwen3-ASR 0.6B | audio.cpp | 195 / 311 | 0% | 0.067 | 210 | 既定の 30 秒の窓 |
| Qwen3-ASR 0.6B | speech.cpp（Vulkan） | 50 / 140 | 94% | 0.073 | 95 | 0.2 秒ごとに頭から |
| Qwen3-ASR 0.6B | CrispASR | 128 / 343 | 0% | 0.442 | 248 | 途中経過なし（終わりを送ってから 1 回認識）。34 発話中 16 発話で確定した文字が返らなかった（CER は参考） |
| SenseVoiceSmall | audio.cpp | 46 / 70 | 0% | 0.056 | 99 | 窓ごと |
| SenseVoiceSmall | CrispASR | 33 / 46 | 0% | 0.058 | 59 | 途中経過なし（終わりを送ってから 1 回認識） |

<!-- stream-asr-tables:end -->

**音素を話し終えてから文字が出るまで**（JSUT basic5000。ストリーミングは 60 発話、疑似ストリーミングは 30 発話）
- 音素ごとの時刻は [jsut-lab](https://github.com/r9y9/jsut-lab)（Julius による強制アラインメント）。途中の文字を pyopenjtalk で
  音素に直して正解の音素の頭と対応づけ、その音素を話し終えてから対応する文字が初めて出るまで
  （[asr/scripts/emission_latency.py](asr/scripts/emission_latency.py)）。「書き換わらなくなるまで」は、出たあと確定まで
  消えなくなった時点まで。認識を誤った音素は数えない（「対応がついた音素」の割合）
- 「話し終わってから文字が出るまで」は、最後の音素の終わりから最終的な文字が届くまで（音声は最後の音素の 0.3 秒あとで切る）
- **アプリの疑似ストリーミング**：デモアプリ（[apps/asr_core.py](../apps/asr_core.py)。MaAI-SDS の ASR と同じしくみと既定値）を
  そのまま動かした。Silero VAD で区切り（始まりは確率 0.7 超えが 0.1 秒、終わりは 0.3 未満が 0.1 秒）、プリロール 0.5 秒（※）、
  音声 0.25 秒ごとに区間の頭から認識し直して途中経過にし、間に合わない途中経過の依頼は捨てる。実行系の構成は、認識するモデル
  だけを差し替えた。発話の前に 0.5 秒・後ろに 1.5 秒の無音を実時間で送る。疑似ストリーミングで測ったのは、`timing100` の
  1 発話（平均 8.5 秒）を 1 回で 250 ms 以内に認識できる構成だけ（途中経過の間隔の 0.25 秒に追いつけるもの）。もともと
  ストリーミングのモデル（Nemotron・Voxtral Realtime）は、ストリーミングとして測った
- ※ アプリと MaAI-SDS の既定のプリロールは 0.2 秒だが、VAD が始まりと判定するのは話し始めの 0.15 秒ほどあとなので、
  区間の頭に残る無音は 50 ms ほどしかなく、ReazonSpeech k2 は最初の語を落とした（MaAI-SDS は ReazonSpeech k2 にだけ
  先頭 0.5 秒の無音を足して防いでいる。`sherpa-reazonspeech-k2-sds`）。モデルによる差が出ないように、すべて 0.5 秒にそろえた

<!-- jsut-asr-tables:start -->
| 構成 | 実行系 | 方式 | 音素から文字が出るまで ms（中央値 / p90） | 書き換わらなくなるまで ms（中央値 / p90） | 話し終わってから文字が出るまで ms（中央値 / p90） | 読みの CER | 対応がついた音素 |
|---|---|---|---:|---:|---:|---:|---:|
| speechcpp-parakeet-tdt-ctc-0.6b-ja | speech.cpp（Vulkan） | アプリの疑似ストリーミング | 62 / 225 | 70 / 233 | 287 / 324 | 0.012 | 100% |
| parakeet-tdt-ctc-0.6b-ja | parakeet-tdt-ctc-0.6b-ja | アプリの疑似ストリーミング | 73 / 233 | 77 / 242 | 292 / 332 | 0.012 | 100% |
| crispasr-parakeet-tdt-ctc-0.6b-ja | CrispASR | アプリの疑似ストリーミング | 79 / 239 | 85 / 249 | 298 / 332 | 0.012 | 100% |
| sherpa-reazonspeech-k2-sds | sherpa-reazonspeech-k2-sds | アプリの疑似ストリーミング | 87 / 250 | 91 / 256 | 312 / 343 | 0.010 | 100% |
| speechcpp-haas-ja | speech.cpp（Vulkan） | アプリの疑似ストリーミング | 94 / 261 | 96 / 264 | 284 / 319 | 0.011 | 100% |
| crispasr-anime-whisper | CrispASR | アプリの疑似ストリーミング | 98 / 311 | 159 / 373 | 373 / 454 | 0.134 | 98% |
| haas-ja | haas-ja | アプリの疑似ストリーミング | 102 / 262 | 102 / 264 | 291 / 330 | 0.011 | 100% |
| granite | granite | アプリの疑似ストリーミング | 104 / 304 | 105 / 324 | 262 / 304 | 0.018 | 100% |
| audiocpp-haas-ja | audio.cpp | アプリの疑似ストリーミング | 107 / 274 | 108 / 277 | 296 / 340 | 0.011 | 100% |
| sherpa-parakeet-ctc | sherpa-parakeet-ctc | アプリの疑似ストリーミング | 107 / 272 | 110 / 288 | 356 / 438 | 0.017 | 100% |
| crispasr-haas-ja | CrispASR | アプリの疑似ストリーミング | 129 / 298 | 132 / 301 | 364 / 400 | 0.015 | 100% |
| audiocpp-cohere-transcribe | audio.cpp | アプリの疑似ストリーミング | 141 / 345 | 151 / 458 | 322 / 367 | 0.033 | 99% |
| speechcpp-reazonspeech-nemo-v2 | speech.cpp（Vulkan） | アプリの疑似ストリーミング | 142 / 354 | 151 / 442 | 322 / 359 | 0.052 | 99% |
| sherpa-reazonspeech-k2 | sherpa-reazonspeech-k2 | アプリの疑似ストリーミング | 147 / 428 | 163 / 531 | 306 / 339 | 0.022 | 100% |
| speechcpp-qwen3-asr-0.6b | speech.cpp（Vulkan） | アプリの疑似ストリーミング | 151 / 314 | 155 / 323 | 320 / 369 | 0.047 | 99% |
| crispasr-reazonspeech-nemo-v2 | CrispASR | アプリの疑似ストリーミング | 152 / 370 | 159 / 462 | 316 / 349 | 0.051 | 99% |
| parakeet-ctc-1.1b-ja | parakeet-ctc-1.1b-ja | アプリの疑似ストリーミング | 163 / 327 | 171 / 372 | 309 / 356 | 0.044 | 99% |
| sherpa-sensevoice | sherpa-sensevoice | アプリの疑似ストリーミング | 166 / 310 | 168 / 324 | 304 / 341 | 0.036 | 99% |
| speechcpp-qwen3-asr-1.7b | speech.cpp（Vulkan） | アプリの疑似ストリーミング | 167 / 341 | 170 / 348 | 378 / 454 | 0.038 | 99% |
| crispasr-cohere-asr-ja-v0.1 | CrispASR | アプリの疑似ストリーミング | 172 / 351 | 177 / 365 | 427 / 508 | 0.017 | 100% |
| crispasr-cohere-transcribe | CrispASR | アプリの疑似ストリーミング | 176 / 386 | 190 / 492 | 438 / 521 | 0.036 | 99% |
| crispasr-sensevoice | CrispASR | アプリの疑似ストリーミング | 180 / 321 | 181 / 326 | 308 / 348 | 0.047 | 99% |
| audiocpp-sensevoice | audio.cpp | アプリの疑似ストリーミング | 188 / 327 | 192 / 336 | 328 / 356 | 0.047 | 99% |
| faster-whisper | faster-whisper | アプリの疑似ストリーミング | 194 / 397 | 219 / 421 | 457 / 532 | 0.026 | 100% |
| crispasr-parakeet-ctc-1.1b-ja | CrispASR | アプリの疑似ストリーミング | 202 / 381 | 207 / 437 | 304 / 354 | 0.066 | 99% |
| crispasr-whisper-large-v3-turbo | CrispASR | アプリの疑似ストリーミング | 206 / 412 | 220 / 429 | 488 / 558 | 0.026 | 99% |
| crispasr-qwen3-asr-0.6b | CrispASR | アプリの疑似ストリーミング | 207 / 396 | 229 / 433 | 491 / 576 | 0.098 | 98% |
| anime-whisper | anime-whisper | アプリの疑似ストリーミング | 208 / 371 | 213 / 381 | 501 / 604 | 0.026 | 100% |
| kotoba-whisper-v2.0 | kotoba-whisper-v2.0 | アプリの疑似ストリーミング | 213 / 388 | 216 / 403 | 503 / 593 | 0.026 | 99% |
| audiocpp-qwen3-asr-0.6b | audio.cpp | アプリの疑似ストリーミング | 214 / 393 | 221 / 411 | 444 / 534 | 0.048 | 99% |
| faster-whisper-sds | faster-whisper-sds | アプリの疑似ストリーミング | 225 / 379 | 228 / 387 | 449 / 532 | 0.028 | 100% |
| anime-whisper-faster | anime-whisper-faster | アプリの疑似ストリーミング | 226 / 366 | 231 / 373 | 433 / 506 | 0.029 | 100% |
| crispasr-lfm2.5-audio-jp | CrispASR | アプリの疑似ストリーミング | 258 / 1449 | 310 / 2250 | 425 / 1224 | 1.801 | 96% |
| moonshine-tiny-ja | moonshine-tiny-ja | アプリの疑似ストリーミング | 325 / 892 | 390 / 1174 | 525 / 811 | 0.065 | 96% |
| nemotron-80ms | nemotron-80ms | ストリーミング：真のストリーミング | 477 / 619 | 477 / 617 | 476 / 497 | 0.067 | 96% |
| speechcpp-haas-ja | speech.cpp（Vulkan） | ストリーミング：0.2 秒ごとに頭から認識し直す | 490 / 1870 | 491 / 1874 | 324 / 354 | 0.013 | 99% |
| audiocpp-nemotron-streaming | audio.cpp | ストリーミング：真のストリーミング | 505 / 676 | 505 / 684 | 316 / 346 | 0.053 | 97% |
| speechcpp-parakeet-tdt-ctc-0.6b-ja | speech.cpp（Vulkan） | ストリーミング：0.2 秒ごとに頭から | 520 / 2766 | 519 / 2774 | 323 / 353 | 0.018 | 99% |
| speechcpp-qwen3-asr-0.6b | speech.cpp（Vulkan） | ストリーミング：0.2 秒ごとに頭から | 539 / 1825 | 542 / 1850 | 340 / 375 | 0.050 | 98% |
| nemotron-560ms | nemotron-560ms | ストリーミング：真のストリーミング | 630 / 900 | 630 / 901 | 380 / 434 | 0.070 | 96% |
| nemotron-nemo | nemotron-nemo | ストリーミング：真のストリーミング | 639 / 880 | 639 / 880 | 335 / 370 | 0.048 | 97% |
| speechcpp-reazonspeech-nemo-v2 | speech.cpp（Vulkan） | ストリーミング：0.2 秒ごとに頭から | 705 / 2988 | 705 / 2988 | 351 / 397 | 0.029 | 98% |
| audiocpp-voxtral-realtime | audio.cpp | ストリーミング：80 ms ずつ | 763 / 894 | 763 / 894 | 314 / 353 | 0.128 | 89% |
| crispasr-nemotron-streaming | CrispASR | ストリーミング：真のストリーミング | 839 / 1024 | 838 / 1024 | 365 / 396 | 0.057 | 97% |
| audiocpp-haas-ja | audio.cpp | ストリーミング：2 秒の窓で認識し直す（右に 2 秒待つ） | 2117 / 3330 | 2117 / 3330 | 345 / 375 | 0.011 | 100% |
| crispasr-qwen3-asr-0.6b | CrispASR | ストリーミング：途中経過なし（終わりを送ってから 1 回認識） | 2168 / 4388 | 2166 / 4388 | 456 / 510 | 0.368 | 97% |
| crispasr-sensevoice | CrispASR | ストリーミング：途中経過なし（終わりを送ってから 1 回認識） | 2300 / 5280 | 2300 / 5280 | 328 / 361 | 0.032 | 98% |
| audiocpp-sensevoice | audio.cpp | ストリーミング：窓ごと | 2314 / 5280 | 2314 / 5280 | 338 / 369 | 0.034 | 98% |
| crispasr-haas-ja | CrispASR | ストリーミング：途中経過なし（終わりを送ってから 1 回認識） | 2340 / 5308 | 2340 / 5308 | 381 / 420 | 0.020 | 98% |
| audiocpp-qwen3-asr-0.6b | audio.cpp | ストリーミング：既定の 30 秒の窓 | 2407 / 5412 | 2407 / 5412 | 436 / 502 | 0.043 | 97% |

<!-- jsut-asr-tables:end -->


### TTS

<!-- stream-tts-tables:start -->
| モデル | 実行系 | 最初の音 ms（中央値） | 最初の声 ms（中央値 / 平均） | 全部届くまで ms（平均） | 比べる：ストリーミングなし ms | 注 |
|---|---|---:|---:|---:|---:|---|
| Supertonic 3 | audio.cpp | 69 | 69 / 67 | 67 | 91 | ほぼまとめて届く |
| Supertonic 3 | CrispASR | 157 | 157 / 163 | 164 | 191 | ほぼまとめて届く |
| Irodori-TTS v4 Small | speech.cpp（Vulkan） | 439 | 439 / 431 | 535 | 571 | ほぼまとめて届く |
| Irodori-TTS v4 Small | audio.cpp | 302 | 302 / 306 | 604 | 394 | ほぼまとめて届く |
| Irodori-TTS v4.1 Small MF | speech.cpp（Vulkan） | 57 | 57 / 57 | 146 | 165 | 逐次に届く |
| Irodori-TTS v4.1 Small MF | audio.cpp | 125 | 125 / 125 | 420 | 207 | 逐次に届く |
| Irodori-TTS v4.1 Anime | audio.cpp | 305 | 305 / 304 | 597 | 390 | ほぼまとめて届く |
| Irodori-TTS Large | speech.cpp（Vulkan） | 1054 | 1054 / 1105 | 1234 | 1304 | ほぼまとめて届く |
| Irodori-TTS 500M v3 | audio.cpp | 263 | 263 / 270 | 551 | 351 | 逐次に届く |
| Irodori-TTS 500M v3 | CrispASR | 1718 | 1718 / 1783 | 1784 | 1800 | ほぼまとめて届く |
| Irodori-TTS 600M v3 VoiceDesign | audio.cpp | 280 | 280 / 280 | 576 | 366 | 逐次に届く |
| Irodori-TTS 600M v3 VoiceDesign | CrispASR | 1769 | 1769 / 1848 | 1850 | 1887 | ほぼまとめて届く・Q4_K |
| Qwen3-TTS 0.6B Base（クローン） | audio.cpp | 40 | 82 / 91 | 691 | 612 | 逐次に届く |
| Qwen3-TTS 0.6B Base（クローン） | CrispASR | 114 | 114 / 114 | 936 | 1036 | 逐次に届く |
| Qwen3-TTS 0.6B CustomVoice（Ono_Anna） | speech.cpp（Vulkan） | 14 | 81 / 89 | 546 | 604 | 逐次に届く |
| Qwen3-TTS 0.6B CustomVoice（Ono_Anna） | CrispASR | 112 | 114 / 153 | 1132 | 868 | 逐次に届く |
| Qwen3-TTS 1.7B Base（クローン） | audio.cpp | 44 | 92 / 106 | 760 | 679 | 逐次に届く |
| Qwen3-TTS 1.7B Base（クローン） | CrispASR | 123 | 123 / 125 | 925 | 756 | 逐次に届く |
| Qwen3-TTS 1.7B CustomVoice（Ono_Anna） | speech.cpp（Vulkan） | 19 | 32 / 32 | 577 | 620 | 逐次に届く |
| Qwen3-TTS 1.7B CustomVoice（Ono_Anna） | audio.cpp | 23 | 71 / 70 | 719 | 634 | 逐次に届く |
| Qwen3-TTS 1.7B CustomVoice（Ono_Anna） | CrispASR | 118 | 118 / 120 | 1030 | 820 | 逐次に届く |
| OmniVoice | audio.cpp | 306 | 306 / 319 | 321 | 363 | ほぼまとめて届く |
| OmniVoice | CrispASR | 445 | 445 / 459 | 459 | 508 | ほぼまとめて届く |
| CosyVoice3 0.5B（クローン・カタカナで入力） | CosyVoice（公式・PyTorch） | 2675 | 2675 / 2683 | 2761 | 3337 | ほぼまとめて届く |
| CosyVoice3 0.5B（クローン・カタカナで入力） | CrispASR | 399 | 399 / 417 | 418 | 450 | ほぼまとめて届く・LLM は Q4_K |
| MOSS-TTS-Nano（クローン） | audio.cpp | 17 | 18 / 20 | 338 | 229 | 逐次に届く |
| MOSS-TTS-Local v1.5（クローン） | CrispASR | 1286 | 1286 / 1372 | 1373 | 1364 | ほぼまとめて届く |
| VoxCPM2（クローン） | audio.cpp | 58 | 58 / 58 | 763 | 780 | 逐次に届く |
| VoxCPM2（クローン） | CrispASR | 1327 | 1327 / 1309 | 1310 | 1368 | ほぼまとめて届く |
| MioTTS | CrispASR | 1342 | 1342 / 1578 | 1579 | 1637 | ほぼまとめて届く・0.6B・温度 0.8 |
| Confucius4-TTS（クローン） | audio.cpp | 2338 | 2338 / 2508 | 2509 | 2568 | ほぼまとめて届く |
| Confucius4-TTS（クローン） | CrispASR | 26715 | 26715 / 28207 | 28208 | 28309 | ほぼまとめて届く |
| Fish Audio S2 Pro（クローン） | audio.cpp | 91 | 91 / 115 | 2433 | 2304 | 逐次に届く |
| Higgs Audio TTS 3 4B（クローン） | audio.cpp | 183 | 205 / 214 | 1075 | 1292 | 逐次に届く |
| GPT-SoVITS v2ProPlus（クローン） | GPT-SoVITS（PyTorch） | 382 | 382 / 570 | 1244 | 2084 | 逐次に届く |
| Qwen3-TTS 1.7B VoiceDesign（声を文章で指定） | audio.cpp | 24 | 25 / 40 | 753 | 710 | 逐次に届く・説明：落ち着いた若い女性の声 |
| LFM2.5-Audio-1.5B-JP（TTS） | audio.cpp | 22 | 51 / 54 | 331 | 328 | 逐次に届く |
| ZONOS3-Mini | audio.cpp | 61 | 88 / 97 | 240 | 225 | 逐次に届く |

<!-- stream-tts-tables:end -->

## LLM

同じ GGUF を LM Studio（中の llama.cpp）と公式の llama.cpp（`llama-server`、CUDA 版と Vulkan 版）で動かした
（[llm/scripts/bench_runtimes.py](llm/scripts/bench_runtimes.py)。測り方は [LLM のページ](llm/2026-11-lmstudio.md) と同じ）。

<!-- llm-tables:start -->
| モデル | 実行系 | TTFT 中央値 / 最悪 ms | tok/s 短い返事 | tok/s 長めの返事 |
|---|---|---:|---:|---:|
| elyza-shortcut-1.0-qwen-7b | LM Studio | 29 / 54 | 104 | 101 |
| elyza-shortcut-1.0-qwen-7b | llama.cpp（CUDA） | 36 / 2348 | 102 | 96 |
| elyza-shortcut-1.0-qwen-7b | llama.cpp（Vulkan） | 44 / 783 | 110 | 116 |
| gemma-4-26b-a4b-it-japanese-64e | LM Studio | 67 / 106 | 126 | 133 |
| gemma-4-26b-a4b-it-japanese-64e | llama.cpp（CUDA） | 80 / 5844 | 124 | 133 |
| gemma-4-26b-a4b-it-japanese-64e | llama.cpp（Vulkan） | 78 / 607 | 120 | 122 |
| lfm2.5-1.2b-jp-202606 | LM Studio | 38 / 39 | 367 | 318 |
| lfm2.5-1.2b-jp-202606 | llama.cpp（CUDA） | 30 / 36 | 371 | 326 |
| lfm2.5-1.2b-jp-202606 | llama.cpp（Vulkan） | 30 / 36 | 378 | 381 |
| llm-jp-4-8b-instruct | LM Studio | 66 / 78 | 128 | 93 |
| llm-jp-4-8b-instruct | llama.cpp（CUDA） | 66 / 73 | 119 | 94 |
| llm-jp-4-8b-instruct | llama.cpp（Vulkan） | 74 / 294 | 124 | 102 |
| qwen3.5-9b | LM Studio | 82 / 98 | 84 | 80 |
| qwen3.5-9b | llama.cpp（CUDA） | 78 / 86 | 82 | 82 |
| qwen3.5-9b | llama.cpp（Vulkan） | 90 / 102 | 79 | 79 |

<!-- llm-tables:end -->
