# 実行系の準備（CrispASR・audio.cpp・speech.cpp）

Python を使わずに動く ggml 系の 3 つの実行系を、このリポジトリの中だけに用意する手順です。システムの `PATH` やユーザーのフォルダーは書き換えません。

| 実行系 | 版 | 入れ方 | GPU | 置き場所 |
|---|---|---|---|---|
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | v0.8.54 | 配布のビルド済み（CUDA 12） | CUDA | `models/src/crispasr-bin/` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `cfc30b04`（2026-11-16） | ソースからビルド（CUDA 12.8） | CUDA | `models/src/audio.cpp/` |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | v0.11.0 | 配布のビルド済み（Vulkan） | Vulkan（Windows に CUDA 版は無い） | `models/src/speechcpp-bin/` |

確かめた環境は [benchmarks/README.md](../benchmarks/README.md) の PC（Windows 11・RTX 5070 Ti Laptop・CUDA 12.8・Visual Studio 2022）です。

## 入れ方

### CrispASR

```bash
mkdir -p models/src/crispasr-bin && cd models/src/crispasr-bin
curl -L -o crispasr-cuda.zip https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.54/crispasr-windows-x86_64-cuda.zip
sha256sum crispasr-cuda.zip   # 9506a83dabedd9e702922f01aefc18c63c4c6ac4aa8002a73e89d23939446ef4
unzip crispasr-cuda.zip       # → crispasr-windows-x86_64-cuda/crispasr.exe
crispasr-windows-x86_64-cuda/crispasr.exe --diagnostics   # cuda archs に 120-real（RTX 50 系）があること
```

- サーバーは `crispasr.exe --server --backend <バックエンド> -m auto --cache-dir models/crispasr --model-quant q8_0 -l ja`。`-m auto` の既定は多くが Q4_K なので、量子化をそろえるときは `--model-quant` か `-m <パス>` を使う（`runtimes/catalog.py`）
- `parakeet-ja`・`parakeet-ctc-1.1b-ja`・`qwen3-ja-anime` は登録表の別名で、`--backend` には使えない。`--backend parakeet` / `fastconformer-ctc` / `qwen3-1.7b` にして `-m` でファイルを渡す
- ライセンスへの同意が要るモデルがある：Fun-ASR・SenseVoice は `--accept-license funasr-v1.1`、OmniVoice は `omnivoice-terms`、Supertonic 3 は `openrail-m`（測定では同意して使っている）
- CPU で動かすときは `-ng`
- LLM 系の TTS の温度（`-tp`）の既定は 0。MioTTS 0.6B はこのままだと止まらない文が出るので、0.8 を渡す

### audio.cpp

ソースから CUDA 版のサーバーをビルドします。Visual Studio 2022（C++）と CUDA 12 以降が要ります。

```powershell
git clone https://github.com/0xShug0/audio.cpp models/src/audio.cpp
cd models/src/audio.cpp
# CUDA 11.8 も入っている PC では既定でそちらを拾って止まるので、12.8 を指す
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_server -Jobs 12
# → build\windows-cuda-release\bin\audiocpp_server.exe
```

- 一度 11.8 で失敗したら、`build\windows-cuda-release` を消してからやり直す
- モデルは `tools/model_manager_v2.py install <パッケージ> --models-root models/audiocpp` でも入るが、遅いことがある。`runtimes/servers.py` はパッケージの情報から URL を組み立てて curl で取る
- サーバーの設定の `models` に、モデルごとに `id`・`family`・`path`・`task`・`mode` を書く。MioTTS 1.7B は `session_options` の `miotts.codec_model_path` にコーデック（`miocodec_q8_0`）の場所が要る
- 声のクローンは、リクエストに `voice_ref`（WAV のパス）と、要るモデルでは `reference_text` を付ける
- Nemotron Streaming の言語は `ja` ではなく `ja-JP`

### speech.cpp

公式のインストーラーはユーザーの `PATH` を書き換えるので使わず、リリースの zip を取ります。

```bash
mkdir -p models/src/speechcpp-bin && cd models/src/speechcpp-bin
B=https://github.com/nyosegawa/speech.cpp/releases/download/v0.11.0
curl -L -o speech.zip $B/speech-0.11.0-windows-x64-vulkan.zip
sha256sum speech.zip          # b64f498179fc46d08149317eaea9fe6f24e6fa0cd194557b374c97e76ccc9c67
unzip speech.zip              # → speech.exe・speech.dll
export SPEECH_MODEL_DIR="$PWD/../../speechcpp"   # モデルの置き場所（既定は %LOCALAPPDATA%\speech.cpp\models）
```

- 既定の `--device auto` は最初の GPU なので、内蔵 GPU がある PC ではそちらで動く。`speech devices` で一覧を見て `--device Vulkan1` のように指定する
- サーバーの `model` に名前を渡すと 404 になることがあるので、1 モデルだけ読み込んで `model` を省く
- Irodori は声の指定が必須。参照音声なしなら `voice` に `none` を渡す

## このリポジトリから使う

モデルの名前と立ち上げ方は `runtimes/catalog.py` にあります。構成の名前は `<実行系>-<名前>`（例 `speechcpp-haas-ja`・`audiocpp-supertonic-3`）で、ASR は `asrkit`、TTS は `ttskit` からそのまま使えます。サーバーは空いているポートで自動で立ち上げ、終わると止めます。

```bash
# ASR（timing100 の 100 発話。speech.cpp はシェーダーを作るので全発話でウォームアップ）
.venv/Scripts/python benchmarks/asr/scripts/eval_asr.py --manifest data/eval/timing100/manifest.json \
    --backend speechcpp-haas-ja --device cuda --warmup 100 --out data/results/timing_gpu/speechcpp-haas-ja.json
# TTS（20 文）と読みの採点
.venv-tts/Scripts/python benchmarks/tts/scripts/bench_tts.py --engine audiocpp-supertonic-3
.venv/Scripts/python benchmarks/tts/scripts/score_tts.py audiocpp-supertonic-3
# ストリーミング（実時間で送る）
.venv/Scripts/python benchmarks/asr/scripts/stream_asr.py --backend audiocpp-nemotron-streaming
.venv-tts/Scripts/python benchmarks/tts/scripts/stream_tts.py --engine speechcpp-qwen3-tts-0.6b-customvoice
# 動くかだけ確かめる（2 発話・2 文）
.venv/Scripts/python benchmarks/asr/scripts/smoke_runtimes.py crispasr-haas-ja
.venv-tts/Scripts/python benchmarks/tts/scripts/smoke_runtimes.py crispasr-supertonic-3
```

- 声をクローンする TTS の参照音声は `data/voices/ref_ja.wav`（Common Voice 8.0 の 1 発話・CC0）と書き起こし `ref_ja.txt`。Qwen3-TTS の Base には 24 kHz の `ref_ja_24k.wav` を使う
- 結果には読み込み時間（`load_s`）・GPU のメモリの増え方の最大（`gpu_mem_mb`）・RAM の最大（`ram_mb`、子プロセスを含む）も入る（`runtimes/memwatch.py`）

### ストリーミングの呼び方

`runtimes/streaming.py` が 3 つの実行系をストリーミングで呼びます（測り方は [実行系の比較](../benchmarks/2026-11-runtimes.md#ストリーミング)）。

| 実行系 | TTS | ASR |
|---|---|---|
| audio.cpp | モデルを `"mode": "streaming"` で登録し、`"stream": true, "stream_format": "audio", "response_format": "pcm"` で int16 の PCM がチャンクで届く。MioTTS・MOSS-TTS-Local・OuteTTS は非対応、VoxCPM2 は `retry_badcase: false` が要る | `POST /v1/audio/transcriptions/live` にチャンク転送で PCM を送り、SSE で文字が返る。最後に長さ 0 のチャンクを送る |
| speech.cpp | `"response_format": "pcm"` でチャンクで届く（`X-Sample-Rate` ヘッダー付き。`"stream"` を付けると 400） | WebSocket `/v1/realtime`（OpenAI Realtime の形）。音声は 24 kHz の PCM を base64 で送る |
| CrispASR | `"stream": true`。逐次に音を出すのは Qwen3-TTS だけで、ほかは 1 文ずつ作ってから送る | `--ws-port` を付けると開く vLLM Realtime（`/v1/realtime`、16 kHz）。途中経過を出すのは Nemotron だけ |

## 比べるための元の実装（PyTorch）

移植で値が変わっていないかを見るため、同じモデルを公式の実装でも測ります。依存がぶつかるので、モデルごとに環境を分けます（PyTorch は RTX 50 系が動く CUDA 12.8 版）。

```bash
T="torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
# Qwen3-TTS（qwen-tts は transformers 4.57.3 に固定）
uv venv .venv-qwen-tts --python 3.11 && uv pip install --python .venv-qwen-tts/Scripts/python.exe $T
uv pip install --python .venv-qwen-tts/Scripts/python.exe qwen-tts==0.1.1
# OmniVoice（transformers 5.3 以降）
uv venv .venv-omnivoice --python 3.11 && uv pip install --python .venv-omnivoice/Scripts/python.exe $T
uv pip install --python .venv-omnivoice/Scripts/python.exe omnivoice==0.2.1
# CosyVoice3（requirements.txt は古い PyTorch に固定なので、推論に要るものだけ入れる）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/src/CosyVoice
uv venv .venv-cosyvoice --python 3.11 && uv pip install --python .venv-cosyvoice/Scripts/python.exe $T
uv pip install --python .venv-cosyvoice/Scripts/python.exe conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 inflect==7.3.1 librosa lightning omegaconf==2.3.0 onnxruntime openai-whisper \
    x-transformers==2.11.24 wetext==0.0.4 transformers==4.51.3 numpy==1.26.4 pyarrow==18.1.0 pyworld wget \
    soundfile matplotlib huggingface_hub rich gdown
# Style-Bert-VITS2（pip 版。pyopenjtalk-dict が numpy 1 向け、BERT の型がずれるので transformers<4.50）
uv venv .venv-sbv2 --python 3.11
uv pip install --python .venv-sbv2/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sbv2/Scripts/python.exe style-bert-vits2 soundfile "numpy<2" "transformers<4.50" "setuptools<81"
# GPT-SoVITS（公式リポジトリ）
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS models/src/GPT-SoVITS
uv venv .venv-gptsovits --python 3.12
uv pip install --python .venv-gptsovits/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

- 構成の名前は `qwen3-tts-0.6b-base` などで、`apps/tts_engines.toml` の節の `python` にその環境を書く（`apps/tts_engines.example.toml`）。モデルは `models/hf` に取ってくる
- CosyVoice3 は日本語をカタカナで渡す（設定の `kana = true`）。漢字のまま渡すと読みが崩れる（CrispASR 版で読みの CER 0.44、カタカナで 0.013）
- GPT-SoVITS は、[公式の Windows 用 wheel](https://huggingface.co/lj1995/GPT-SoVITS-windows-package)（`wheels/cu128/` の jieba_fast・pyopenjtalk・langsegment）を入れてから `requirements.txt`（先頭の `--no-binary=opencc` を除く）を入れる。重みは `XXXXRT/GPT-SoVITS-Pretrained` から v2ProPlus に要るもの（chinese-hubert-base・chinese-roberta-wwm-ext-large・fast_langdetect・s1v3.ckpt・sv・v2Pro/s2Gv2ProPlus.pth）を `GPT_SoVITS/pretrained_models` へ置き、Open JTalk の辞書を pyopenjtalk のフォルダーへ展開する
- Style-Bert-VITS2 の声は公式の例の JVNV コーパス（CC BY-SA 4.0）のモデル `jvnv-F1-jp`
- ReazonSpeech NeMo v2 と Parakeet TDT-CTC 0.6B ja は、Haas-ja と同じ `.venv-haas`（NeMo）で動く
- anime-whisper は transformers・faster-whisper（有志の CTranslate2 変換）・CrispASR（有志の whisper.cpp 変換）の 3 通りで動かす
- Windows の torchaudio 2.9 以降の `load` は torchcodec（FFmpeg の DLL）を使うので、soundfile で読むように差し替えている

## 既知の問題

- **CrispASR の NeMo 系**（Haas・ReazonSpeech・Parakeet）は、発話の前後に小さく入った別の人の声まで書き起こすので、ほかの実行系より読みの CER が悪い。重みを F16 にしても、VAD の区切りを止めても変わらない
- **CrispASR の `parakeet-ctc-1.1b-ja` の GGUF** は、元のリポジトリのギャル語版（`parakeet-ja-gal.nemo`）から変換されたとみられる
- **CrispASR の CLI** に NeMo 系で 20 秒ほどより長い音声を渡すと、自動で VAD を入れ、そのモデルを `~/.cache/crispasr/` に取ってくる（サーバーでは起きない）
- **CrispASR の Realtime の Qwen3-ASR** は、途中まで文字を送ったあと確定の文字（`completed`）を返さず、接続を開いたまま止まることがある（timing100 の 34 発話中 16 発話。同じ発話で再現する）。サーバーの処理（`examples/server/realtime_server.cpp`）が例外を `catch (...) {}` で記録せずに捨てるので、そこで止まっているとみられる（中身は確かめていない）
- **CrispASR の Realtime の WebSocket** は、0.8.54 では HTTP と同じ番号で `0.0.0.0` に開く。HTTP は `127.0.0.1` に開くので、`127.0.0.1` へつなぐと 404 になる。`127.0.0.2` へつなぐと WebSocket に届く
- **CrispASR の Fun-ASR** は、GPU では 2 回目の要求で落ちる（2026-11-20 から）。CPU では動く
- **speech.cpp の Vulkan** は、初回に GPU のシェーダーを作るので遅い（Haas-ja で 1 回目 13 秒、2 回目から 98 ms）。新しい長さの音声でも作り直すので、測る前に測る発話をすべて 1 回通しておく
