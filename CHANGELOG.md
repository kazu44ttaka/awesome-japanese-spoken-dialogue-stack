# 更新履歴

新しい順。測った値が変わった更新には、どの値が変わったかを書いています。

## 2026-11-21

- README に「実行系（推論のライブラリ）」の節を足し、sherpa-onnx・faster-whisper・CrispASR・audio.cpp・speech.cpp・llama.cpp・LM Studio へのリンクを載せた
- デモアプリで、ベンチマークで測った構成を全部選べるようにした（ASR 62・TTS 59。実行系ごとにまとめて表示）。ggml 系の実行系の構成はサーバーを自動で立ち上げ、audio.cpp と CrispASR の Nemotron・Voxtral は真のストリーミングで動く
- 公平性を見直した。README の ASR の表に、評価データで学習しているモデルの印（†）を付け、太字のいちばんよい値はそれ以外から選ぶようにした（モデルカードで確かめると、ReazonSpeech で学習しているモデルは 9 つ、Common Voice は 4 つ）。Nemotron の値を元の NeMo 版にし（会話も測った）、Kotoba-Whisper v2.2 も会話で測り、beam や量子化がほかと違う行に注意を書いた。UTMOS が声で大きく変わることを書いた
- README の ASR の表に 4 つの評価データの読みの CER、TTS の表に合成時間・読みの CER・UTMOS・最初の声までの時間の列を足した。README を awesome の一覧の形に整理し、目的別のおすすめを足した
- 名前を awesome-japanese-spoken-dialogue-stack に変えた（前は awesome-japanese-speech）。LLM やターンテイキングのモデルも入る、音声対話の部品の一覧なので
- [ASR の精度](benchmarks/asr/2026-11-3sets.md) に、2 人の自由会話（MagicHub）での 31 構成の比較を追加
- [実行系の比較](benchmarks/2026-11-runtimes.md) に、ストリーミングの遅れ（話し終わってから文字が出るまで・音素ごとの遅れ・TTS の最初の声）、TTS の自然さ（UTMOS）と話者の近さ、LLM の実行系（LM Studio・llama.cpp）の比較を追加
- 一覧に anime-whisper・Style-Bert-VITS2・GPT-SoVITS を追加
- **採点の直し**：読みの CER が算用数字を読みに直していなかった（「72 時間」と「七十二時間」が別の読みになっていた）。直して全結果を採点し直した。漢数字で書くモデルが大きく変わり、ReazonSpeech k2 は ReazonSpeech で 0.055 → 0.025、Nemotron は約 0.03 よくなった。ほかの多くは 0.01 以下の変化（[詳しく](benchmarks/asr/2026-11-3sets.md#採点の直し2026-11-21)）
- 全モデルのライセンスを公式ページで確かめた。Moonshine の日本語版（年商 100 万ドル未満なら商用可）、Supertonic 3（モデルは OpenRAIL-M）などを直し、商用の欄に「条件つき」「不明」を足した
- このリポジトリのライセンスを MIT にした

## 2026-11-18

- 同じモデルを実行系ごとに比べる [ページ](benchmarks/2026-11-runtimes.md) を追加。メモリ・読み込み時間・CPU での速さも測った
- 元の PyTorch の実装でも測り、CosyVoice3 にカタカナで渡す方法を追加

## 2026-11-17

- CrispASR・audio.cpp・speech.cpp を比べる実行系に追加
- ASR の処理時間を、同じ 100 発話で GPU と CPU に分けて測った（[ASR の処理時間](benchmarks/asr/2026-11-timing.md)）。TTS の新しい結果を追加
- 設定の誤りで本来の値が出ていなかった構成を直して測り直した（Parakeet CTC 1.1B の重みの取り違え、Whisper の繰り返し止め、sherpa-onnx の Qwen3-ASR の言語指定）
- デモアプリを PySide6 にして、ブラウザ版を追加

## 2026-11-16

- ASR を 3 つの公開評価データ（FLEURS・ReazonSpeech・Common Voice）で比較
- 新しい ASR の候補（Moonshine・Granite・Fun-ASR-MLT など）と、LLM の応答の速さを測定
- 最小の推論例、マイクで試す ASR アプリ、TTS アプリを追加

## 2026-11-15

- 日本語の ASR・TTS・LLM の一覧と、測定結果のページを作った
