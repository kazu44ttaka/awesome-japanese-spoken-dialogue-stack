# ASR の最小の推論例

WAV を 1 つ渡すと書き起こした文を出すだけの、モデルごとのスクリプトです。それぞれ単独で動きます（コピーして使ってください）。マイクで試すなら [apps/asr_demo.py](../../apps/asr_demo.py) を使います。

| ファイル | モデル | 必要なもの | メモ |
|---|---|---|---|
| [sherpa_onnx_ja.py](sherpa_onnx_ja.py) | SenseVoiceSmall / Parakeet TDT-CTC 0.6B 日本語 / ReazonSpeech k2-v2 / Cohere ASR / Fun-ASR-Nano（ONNX） | `sherpa-onnx` | PyTorch 不要・CPU。モデルは自動でダウンロード |
| [moonshine_tiny_ja.py](moonshine_tiny_ja.py) | Moonshine tiny-ja | `torch` `transformers` | 27M。CPU でも速い。非商用 |
| [granite_speech_turboctc.py](granite_speech_turboctc.py) | Granite Speech 5.0 470M TurboCTC | `torch` `transformers>=5.16` | GPU でとても速い。読点をほぼ出さない |
| [whisper_ja.py](whisper_ja.py) | Whisper large-v3-turbo / Kotoba-Whisper v2.0 | `torch` `transformers` | GPU 推奨 |
| [fun_asr_nano.py](fun_asr_nano.py) | Fun-ASR-Nano-2512 / Fun-ASR-MLT-Nano-2512 | `torch` `funasr` | GPU 推奨 |

どれも `soundfile` で WAV を読みます。精度と速さの比較は [benchmarks/asr/](../../benchmarks/asr/) を見てください。
