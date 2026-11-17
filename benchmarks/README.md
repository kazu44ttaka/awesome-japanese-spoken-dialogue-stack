# ベンチマーク

同じ PC・同じ条件で測った値だけを載せます。値は測定日・環境・版で変わるので、各ページに必ず書き残します。

## 測定環境（2026-08）

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core Ultra 9 275HX |
| メモリ | 32 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop（12 GB） |

## 指標

| 指標 | 意味 |
|---|---|
| CER | 文字誤り率。正解文との編集距離 ÷ 正解の文字数。小さいほどよい。句読点を含めるかどうかで値が変わるので各ページに書く |
| 読みの CER | 正解文と認識結果をカタカナの読みに変換してから比べた CER。句読点や表記（漢字・かな）の違いを数えない（[asr/scripts/scoring.py](asr/scripts/scoring.py)） |
| 平均 ms / 最悪 ms | 1 発話あたりの処理時間。音声対話では **最悪値** が体感を決める。ウォームアップを含む 1 回目は除く |
| RTF | 生成時間 ÷ 音声の長さ。1 より小さければ実時間より速い |
| ASR 検査（TTS） | 合成した音声を同じ ASR で書き起こし、正解文との CER を見る。欠落・繰り返し・読み間違いの検出用で、声の自然さの評価ではない |

## ページ

- [2026-11-runtimes.md](2026-11-runtimes.md) — 同じモデルを実行系ごとに比較（元の実装・sherpa-onnx・faster-whisper・CrispASR・audio.cpp・speech.cpp）。ASR 18 モデル・TTS 22 モデルの速さ（GPU・CPU）・読みの CER・メモリ・読み込み時間
- [asr/2026-11-3sets.md](asr/2026-11-3sets.md) — ASR 23 構成（MaAI-SDS の設定 2 つを含む）を 3 つの公開評価データ（FLEURS・ReazonSpeech・Common Voice）で比較。読みの CER
- [asr/2026-11-timing.md](asr/2026-11-timing.md) — 同じ 100 発話での ASR の処理時間（GPU と CPU 8 スレッドを分けて）
- [asr/2026-08-jsut20.md](asr/2026-08-jsut20.md) — ASR 18 構成の CER と処理時間
- [asr/2026-11-jsut20.md](asr/2026-11-jsut20.md) — 新しい候補（Moonshine・Granite・Fun-ASR-MLT）と、句読点を除いた CER
- [tts/2026-11-tts.md](tts/2026-11-tts.md) — TTS 9 エンジン（18 構成。MaAI-SDS の設定 2 つを含む）の 20 文での合成時間と読みの CER（GPU・CPU を分けて）
- [tts/2026-08-19.md](tts/2026-08-19.md) — TTS 9 候補の速度・メモリ・ASR 検査
- [llm/2026-11-lmstudio.md](llm/2026-11-lmstudio.md) — LLM 10 モデルの最初の文字までの時間と速さ（LM Studio）

## 評価データ

評価用の音声（JSUT・FLEURS・ReazonSpeech・Common Voice）はライセンス上、このリポジトリには入れません。取得のしかたを各ページに書きます。ReazonSpeech は著作権法 30 条の 4 の範囲でだけ使え、ここでは集計した値だけを載せています。

## 測定スクリプト

- [asr/scripts/eval_asr.py](asr/scripts/eval_asr.py) — ASR の CER（句読点あり・なし・読み）と処理時間。`--dataset fleurs-ja` / `reazonspeech-test` / `cv8-ja` で評価データを取得する
- [asr/scripts/make_table.py](asr/scripts/make_table.py) — 結果を採点し直して Markdown の表にする（`--summary` で評価データを横に並べる）
- [asr/scripts/make_timing_set.py](asr/scripts/make_timing_set.py) — 処理時間を測るための 100 発話の評価セットを作る
- [tts/scripts/bench_tts.py](tts/scripts/bench_tts.py) — TTS で 20 文を 1 文ずつ合成して時間を測る（`--set KEY=VALUE` で設定を変える）
- [tts/scripts/score_tts.py](tts/scripts/score_tts.py) — 合成した音声を 3 つの ASR で書き起こし、読みの CER を付ける
- [llm/scripts/bench_llm.py](llm/scripts/bench_llm.py) — LLM の最初の文字までの時間と速さ（OpenAI 互換 API）

2026-08 の測定に使ったスクリプトは [MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) の `benchmarks/` にあり、手元のパスを取り除いてからここへ移す予定です。

環境は `uv venv` で作り、torch（CUDA 12.8）・transformers 5.16 以降・funasr を入れる:

```
uv venv --python 3.11 .venv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=5.16" funasr soundfile numpy accelerate sentencepiece
```
