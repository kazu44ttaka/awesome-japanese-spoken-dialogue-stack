# LLM：音声対話での応答の速さ（2026-11、LM Studio）

音声対話では、利用者が話し終えてから **返事の最初の文字が出るまでの時間（TTFT）** と、読み上げに追いつく **文字の出る速さ（tok/s）** が体感を決める。日本語の賢さは測っていない（[Nejumi](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/reports/Nejumi-LLM-Leaderboaed-4--VmlldzoxNDQyMzkxMA)・[Swallow](https://swallow-llm.github.io/swallow-leaderboard-v2.en.html) を参照）。

- 環境：[benchmarks/README.md](../README.md) の PC（RTX 5070 Ti Laptop 12 GB）、LM Studio（llama.cpp）、GPU に全層を載せ、コンテキスト 4096
- スクリプト：[scripts/bench_llm.py](scripts/bench_llm.py)（測定日 2026-11-16）
- 音声対話用のシステムプロンプト（1〜2 文の話し言葉で返す）と、毎回ちがう利用者の発話で、ウォームアップ 1 回のあと 5 回測った
- **考える（thinking）モードは切った**（`reasoning_effort: "none"`）。切らないと、128 トークンを考えることに使い切って返事が始まらないモデルがあった
- tok/s（短い返事）は 10〜30 トークンの返事での値で誤差が大きい。長めの返事（約 200 文字を指示）2 回の値も載せる
- モデル名は LM Studio での名前（量子化した GGUF。配布元はモデルごとに違う）

| モデル | 量子化 | GPU 上の大きさ | TTFT 中央値 / 最悪（ms） | tok/s 短い返事 | tok/s 長めの返事 |
|---|---|---:|---:|---:|---:|
| lfm2.5-1.2b-jp-202606 | Q8_0 | 1.25 GB | **39 / 46** | 250 | **215** |
| elyza-shortcut-1.0-qwen-7b | Q4_K_S | 4.46 GB | **45 / 53** | 66 | 64 |
| qwen3.5-0.8b-jp | Q8_0 | 0.81 GB | 63 / 66 | 262 | **251** |
| gemma-4-e2b-it-japanese | Q6_K | 2.12 GB | 66 / 77 | 161 | 146 |
| unsloth/qwen3.5-4b | Q4_K_S | 3.92 GB | 87 / 94 | 97 | 85 |
| gemma-4-e4b-it-japanese | Q4_K_M | 3.00 GB | 95 / 99 | 94 | 86 |
| llm-jp-4-8b-instruct | Q4_K_M | 5.30 GB | 101 / 112 | 78 | 63 |
| gemma-4-26b-a4b-it-japanese-64e（MoE） | Q4_K_S | 8.85 GB | 103 / 110 | 88 | 94 |
| qwen3.5-9b | Q4_K_M | 6.55 GB | 107 / 121 | 59 | 56 |
| gemma-4-12b-it-qat-ud-japanese-imatrix | Q4_K_S | 6.97 GB | 147 / 153 | 38 | 39 |

測れなかったもの：

- **nvidia-nemotron-nano-9b-v2-japanese**（Q4_K_S、6.21 GB）：`reasoning_effort: "none"` でも考えるのをやめず、5 回中 4 回は 128 トークン以内に返事が始まらなかった。システムプロンプトに `/no_think` を入れると返事はするが、それでも少し考えてから答えるため約 1.5 秒かかった。
- **tanuki-8b-dpo-v1.0-4k**：LM Studio の llama.cpp ランタイムが読み込みの途中で終了し、動かせなかった。

## 分かったこと

- **TTFT はどれも 0.15 秒以内** で、ASR（0.1〜0.6 秒）や TTS より小さい。音声対話全体の遅延では、LLM の最初の文字よりも、ASR と TTS の最初の音のほうが効く。
- **読み上げに追いつくか**：日本語の読み上げは目安として 1 秒に 7〜8 文字ほど。最も遅い gemma-4-12b（約 38 tok/s）でも十分に速い。
- **MoE は大きさの割に速い**：gemma-4-26b-a4b（活性 4B）は 12B の密なモデルより速く、TTFT も短い。12 GB の GPU に収まる中では有力。
- **thinking モデルは必ず切る**：Qwen3.5 の各サイズと gemma-4-12b-it-qat は、既定のままだと 128 トークンを考える部分で使い切り、返事が始まらなかった。OpenAI 互換 API なら LM Studio は `reasoning_effort: "none"`、vLLM / llama.cpp は `chat_template_kwargs: {"enable_thinking": false}` で切れる。
