# ベンチマーク

同じ PC・同じ条件で測った値だけを載せています。値は測定日・環境・版で変わるので、各ページに書き残しています。

## ページ

| 内容 | ページ |
|---|---|
| 同じモデルを実行系ごとに比べた速さ・精度・メモリ、ストリーミングの遅れ、TTS の自然さ、LLM の実行系 | [2026-11-runtimes.md](2026-11-runtimes.md) |
| ASR の精度（FLEURS・ReazonSpeech・Common Voice・2 人の自由会話） | [asr/2026-11-3sets.md](asr/2026-11-3sets.md) |
| ASR の処理時間（同じ 100 発話を GPU と CPU で） | [asr/2026-11-timing.md](asr/2026-11-timing.md) |
| TTS の合成時間と読みの正確さ（20 文） | [tts/2026-11-tts.md](tts/2026-11-tts.md) |
| LLM の応答の速さ（LM Studio） | [llm/2026-11-lmstudio.md](llm/2026-11-lmstudio.md) |

前の結果（JSUT 20 発話での比較など。今の採点とは違う）：[asr/2026-08-jsut20.md](asr/2026-08-jsut20.md)・[asr/2026-11-jsut20.md](asr/2026-11-jsut20.md)・[tts/2026-08-19.md](tts/2026-08-19.md)

## 測定環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| CPU | Intel Core Ultra 9 275HX（24 コア。うち 16 が E コア） |
| メモリ | 32 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop（12 GB） |

## 指標

| 指標 | 意味 |
|---|---|
| 読みの CER | 正解文と結果をカタカナの読みに直してから比べた文字誤り率。句読点・表記・算用数字と漢数字の違いを数えない。小さいほどよい（[asr/scripts/scoring.py](asr/scripts/scoring.py)） |
| 処理時間 | 1 発話（ASR）・1 文（TTS）あたりの時間。最初の 1 回はウォームアップとして除く |
| RTF | 処理時間 ÷ 音声の長さ。1 より小さければ実時間より速い |
| ストリーミングの遅れ | 話し終わってから最終的な文字が出るまで、音素を話してから対応する文字が出るまで、TTS で最初の声が出るまで |
| TTS の読みの CER | 合成した音声を 3 つの ASR で書き起こした読みの CER の中央値。読み間違い・欠落の検出用 |
| UTMOS・話者の近さ | 自然さの推定値（UTMOS22）と、参照音声との声の近さ（WavLM の話者ベクトルのコサイン類似度）。どちらも目安 |

## 評価データ

音声と書き起こしはライセンス上このリポジトリに入れていません。各ページに取得のしかたを書いています。

| データ | 使いかた |
|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) 日本語 test | CC-BY-4.0。650 発話全部 |
| [ReazonSpeech](https://huggingface.co/datasets/japanese-asr/ja_asr.reazonspeech_test) test | 著作権法 30 条の 4 の範囲でだけ使う。集計した値だけを載せる |
| [Common Voice 8.0](https://huggingface.co/datasets/japanese-asr/ja_asr.common_voice_8_0) 日本語 test | CC0 |
| [MagicHub](https://huggingface.co/datasets/MagicHub/Japanese_Spontaneous_Conversation_Training_Dataset) の自由会話 | 再配布できないので評価にだけ使う |
| [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) と [jsut-lab](https://github.com/r9y9/jsut-lab) | 音素ごとの時刻を使って、ストリーミングの遅れを測る |

## 測定スクリプト

| スクリプト | すること |
|---|---|
| [asr/scripts/eval_asr.py](asr/scripts/eval_asr.py) | ASR の CER と処理時間。`--dataset` で評価データを取得する |
| [asr/scripts/stream_asr.py](asr/scripts/stream_asr.py) | ストリーミング・疑似ストリーミングで、話し終わってから文字が出るまでを測る |
| [asr/scripts/emission_latency.py](asr/scripts/emission_latency.py) | JSUT の音素の時刻から、音素ごとの遅れを出す |
| [asr/scripts/make_table.py](asr/scripts/make_table.py) | 結果を採点し直して表にする（`--summary` で評価データを横に並べる） |
| [asr/scripts/rescore.py](asr/scripts/rescore.py) | 採点の決まりを変えたあと、結果のファイルを採点し直す |
| [asr/scripts/make_timing_set.py](asr/scripts/make_timing_set.py)・[make_extra_sets.py](asr/scripts/make_extra_sets.py)・[make_jsut_align.py](asr/scripts/make_jsut_align.py) | 処理時間用の 100 発話、会話の評価データ、JSUT の音素の時刻つきのデータを作る |
| [tts/scripts/bench_tts.py](tts/scripts/bench_tts.py)・[score_tts.py](tts/scripts/score_tts.py) | TTS で 20 文を合成して時間を測り、3 つの ASR で読みの CER を付ける |
| [tts/scripts/stream_tts.py](tts/scripts/stream_tts.py)・[quality_tts.py](tts/scripts/quality_tts.py) | TTS の最初の声までの時間、UTMOS と話者の近さ |
| [llm/scripts/bench_llm.py](llm/scripts/bench_llm.py)・[bench_runtimes.py](llm/scripts/bench_runtimes.py) | LLM の最初の文字までの時間と速さ（LM Studio・llama.cpp） |
| [make_runtime_page.py](make_runtime_page.py) | 実行系の比較のページの表を作る |

実行系（CrispASR・audio.cpp・speech.cpp）の入れ方は [docs/runtimes.md](../docs/runtimes.md) にあります。
