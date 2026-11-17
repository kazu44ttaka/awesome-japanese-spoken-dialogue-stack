# 声のクローンに使う参照音声

- `ref_ja.wav`（16 kHz・モノラル・5.5 秒）と書き起こし `ref_ja.txt`
- `ref_ja_24k.wav`：同じ音声を 24 kHz にしたもの（16 kHz を受け付けない Qwen3-TTS の Base 用）

出典は Common Voice 8.0 日本語の 1 発話（このリポジトリの評価で使う `cv8-ja --subset 1000` の `00530`）。
Common Voice の音声は CC0（パブリックドメイン）。
