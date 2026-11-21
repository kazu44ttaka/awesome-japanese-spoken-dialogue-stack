# 実行系の準備（CrispASR・audio.cpp・speech.cpp）

このリポジトリでは、同じモデルを複数の実行系で動かして速さと精度を比べる。ここでは、Python を使わずに動く
ggml 系の 3 つの実行系を、**このリポジトリの中だけに**用意する手順を書く（システムの PATH や
ユーザーのフォルダーは書き換えない）。

| 実行系 | 版 | 入れ方 | GPU | 置き場所 |
|---|---|---|---|---|
| [CrispASR](https://github.com/CrispStrobe/CrispASR) | v0.8.54 | 配布のビルド済み（CUDA 12） | CUDA | `models/src/crispasr-bin/` |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | `cfc30b04`（2026-11-16） | ソースからビルド（CUDA 12.8） | CUDA | `models/src/audio.cpp/` |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | v0.11.0 | 配布のビルド済み（Vulkan） | Vulkan（Windows に CUDA 版は無い） | `models/src/speechcpp-bin/` |

動作を確かめた環境は [benchmarks/README.md](../benchmarks/README.md) の PC（Windows 11・RTX 5070 Ti Laptop・
CUDA 12.8・Visual Studio 2022）。

## CrispASR

配布の Windows 用 CUDA 12 版（NVIDIA のランタイムの DLL を同梱）を使う。

```bash
mkdir -p models/src/crispasr-bin && cd models/src/crispasr-bin
curl -L -o crispasr-cuda.zip https://github.com/CrispStrobe/CrispASR/releases/download/v0.8.54/crispasr-windows-x86_64-cuda.zip
sha256sum crispasr-cuda.zip   # 9506a83dabedd9e702922f01aefc18c63c4c6ac4aa8002a73e89d23939446ef4
unzip crispasr-cuda.zip       # → crispasr-windows-x86_64-cuda/crispasr.exe
crispasr-windows-x86_64-cuda/crispasr.exe --diagnostics   # cuda archs に 120-real（RTX 50 系）があること
```

- サーバーとして使う: `crispasr.exe --server --backend <バックエンド> -m auto --cache-dir models/crispasr --model-quant q8_0 -l ja`。
  `-m auto` は登録表の既定（多くは Q4_K）を取ってくるので、量子化をそろえるときは `--model-quant` か、ファイルの URL を
  取ってきて `-m <パス>` で渡す（`runtimes/catalog.py`）
- `parakeet-ja`・`parakeet-ctc-1.1b-ja`・`qwen3-ja-anime` は登録表のモデルの別名で、`--backend` には使えない。
  `--backend parakeet` / `fastconformer-ctc` / `qwen3-1.7b` にして、モデルのファイルを `-m` で渡す
- Fun-ASR-Nano・Fun-ASR-MLT-Nano・SenseVoice は、FunASR のモデルライセンス（funasr-v1.1）への明示的な同意
  `--accept-license funasr-v1.1` が無いと取ってこない。このリポジトリの測定では同意して使っている
- 同じく OmniVoice は `--accept-license omnivoice-terms`、Supertonic 3 は `--accept-license openrail-m` が要る（同意して使っている）
- CPU で動かすときは `-ng`
- LLM 系の TTS の温度（`-tp`）の既定は 0（毎回いちばん確からしいトークンを選ぶ）。MioTTS 0.6B はこのままだと
  20 文中 3 文が止まらずに 30 秒まで伸びたので、モデルの既定の 0.8 を渡している（CosyVoice3 にはこの設定は効かない）
- CLI で NeMo 系（haas・parakeet など）に長め（約 20 秒以上）の音声を渡すと、自動で VAD を入れて区間ごとに書き起こす。
  そのとき VAD のモデルを `~/.cache/crispasr/` に取ってくる（サーバーとして使う測定では起きない）
- NeMo 系（haas・reazonspeech・parakeet）は、発話の前後に小さく入った別の人の声まで書き起こすので、ほかの実行系より
  読みの CER が悪い。重みを F16 にしても、VAD の区切りを止めても変わらない
- 配布の `parakeet-ctc-1.1b-ja` の GGUF は、元のリポジトリのギャル語版（`parakeet-ja-gal.nemo`）から変換されたとみられる

## audio.cpp

ソースから CUDA 版のサーバー（`audiocpp_server.exe`）をビルドする。Visual Studio 2022（C++）と CUDA 12 以降が要る。

```powershell
git clone https://github.com/0xShug0/audio.cpp models/src/audio.cpp
cd models/src/audio.cpp
# CUDA 11.8 も入っている PC では、既定でそちらを拾って「CUDA 12.0 以上が要る」で止まる。12.8 を指す
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
.\scripts\build_windows.ps1 -Preset windows-cuda-release -Target audiocpp_server -Jobs 12
# → build\windows-cuda-release\bin\audiocpp_server.exe
```

- 一度 11.8 で設定に失敗したら、`build\windows-cuda-release` を消してからやり直す（`-Clean` は、ビルドが
  一度も終わっていないと失敗する）
- CUDA のアーキテクチャは自動で選ばれる（この PC では `120a-real`）

モデルは `tools/model_manager_v2.py install <パッケージ> --models-root models/audiocpp` で入れられるが、Hugging Face からの
取得が 70 KB/s 程度まで落ちることがあった。`runtimes/servers.py` は、パッケージの情報（`info <パッケージ> --json`）から
URL を組み立てて curl で直接取る（8 MB/s 程度）。

- サーバーの設定（JSON）の `models` に、モデルごとに `id`・`family`・`path`・`task`・`mode` を書く。モデルによっては
  `session_options` が要る（MioTTS 1.7B は音を作るコーデックの場所 `miotts.codec_model_path` に、別パッケージ
  `miocodec_q8_0` のフォルダーを渡す。MioTTS は参照音声も必須）
- 声のクローンは、リクエストに `voice_ref`（WAV のパス）と、要るモデルでは `reference_text` を付ける
- Nemotron Streaming の言語は `ja` ではなく `ja-JP`

## speech.cpp

公式のインストーラー（`install.ps1`）は `%LOCALAPPDATA%` に入れてユーザーの `PATH` を書き換えるので使わず、
リリースの zip を直接取る。Windows 用は Vulkan 版だけ（GPU のドライバーが入れる `vulkan-1.dll` を使う）。

```bash
mkdir -p models/src/speechcpp-bin && cd models/src/speechcpp-bin
B=https://github.com/nyosegawa/speech.cpp/releases/download/v0.11.0
curl -L -o speech.zip $B/speech-0.11.0-windows-x64-vulkan.zip
curl -L -o SHA256SUMS $B/SHA256SUMS
sha256sum speech.zip          # b64f498179fc46d08149317eaea9fe6f24e6fa0cd194557b374c97e76ccc9c67
unzip speech.zip              # → speech.exe・speech.dll
```

モデルは初めて使うときに Hugging Face から取ってくる。置き場所は既定で `%LOCALAPPDATA%\speech.cpp\models` なので、
`SPEECH_MODEL_DIR` で変える。

```bash
export SPEECH_MODEL_DIR="$PWD/models/speechcpp"
```

- 既定の `--device auto` は「最初の GPU」なので、Intel の内蔵 GPU がある PC ではそちら（`Vulkan0`）で動く。
  `speech devices` で一覧を見て、NVIDIA を `--device Vulkan1` のように指定する
- Vulkan の初回は GPU のシェーダーを作るので遅い（Haas-ja で 1 回目 13 秒、2 回目から 98 ms）。新しい長さの音声でも
  作り直すので、測る前に、測る発話をすべて 1 回通しておく
- サーバーの `model` に名前を渡すと大文字・小文字の違いで 404 になることがあるので、1 モデルだけ読み込んで `model` を省く
- Irodori は声の指定が必須。参照音声なしで使うときは `voice` に `none` を渡す

## このリポジトリから使う

モデルの名前と立ち上げ方は `runtimes/catalog.py` にまとめてある。構成の名前は `<実行系>-<名前>`
（例 `speechcpp-haas-ja`・`audiocpp-supertonic-3`）で、ASR は `asrkit`、TTS は `ttskit` からそのまま使える。
サーバーは空いているポートで自動で立ち上げ、終わると止める。モデルは初めて使うときに `models/` の下へ取ってくる。

```bash
# ASR（timing100 の 100 発話。speech.cpp はシェーダーを作るので全発話でウォームアップ）
.venv/Scripts/python benchmarks/asr/scripts/eval_asr.py --manifest data/eval/timing100/manifest.json \
    --backend speechcpp-haas-ja --device cuda --warmup 100 --out data/results/timing_gpu/speechcpp-haas-ja.json
# TTS（20 文）と読みの採点
.venv-tts/Scripts/python benchmarks/tts/scripts/bench_tts.py --engine audiocpp-supertonic-3
.venv/Scripts/python benchmarks/tts/scripts/score_tts.py audiocpp-supertonic-3
# 動くかだけ確かめる（2 発話・2 文）
.venv/Scripts/python benchmarks/asr/scripts/smoke_runtimes.py crispasr-haas-ja
.venv-tts/Scripts/python benchmarks/tts/scripts/smoke_runtimes.py crispasr-supertonic-3
```

声をクローンする TTS は、Common Voice 8.0（CC0）の 1 発話 `data/voices/ref_ja.wav`（16 kHz・5.5 秒）と、その書き起こし
`data/voices/ref_ja.txt` を参照音声に使う。Qwen3-TTS の Base は 16 kHz の参照音声を受け付けないので、24 kHz にした
`ref_ja_24k.wav` を使う。

### ストリーミング

`runtimes/streaming.py` が 3 つの実行系をストリーミングで呼ぶ（測り方は [2026-11-runtimes.md](../benchmarks/2026-11-runtimes.md#ストリーミング)）。

```bash
.venv/Scripts/python benchmarks/asr/scripts/stream_asr.py --backend audiocpp-nemotron-streaming   # 実時間で送る
.venv-tts/Scripts/python benchmarks/tts/scripts/stream_tts.py --engine speechcpp-qwen3-tts-0.6b-customvoice
```

- **audio.cpp**：モデルを `"mode": "streaming"` で登録しないとストリーミングできない（`mode: offline` のままだと
  エラー）。TTS は `"stream": true, "stream_format": "audio", "response_format": "pcm"` で、ヘッダーなしの int16 の PCM が
  チャンクで届く。サンプリングレートは応答に無いので、ふつうに 1 回合成した WAV から取る。ASR は
  `POST /v1/audio/transcriptions/live` にチャンク転送で PCM を送り、SSE で文字が返る（最後に長さ 0 のチャンクを送る）。
  MioTTS・MOSS-TTS-Local・OuteTTS はストリーミングに対応していない。VoxCPM2 は `retry_badcase: false` が要る
- **speech.cpp**：TTS は `"response_format": "pcm"` だけでチャンクで届く（`X-Sample-Rate` ヘッダー付き。`"stream"` を
  付けると 400）。ASR は WebSocket `/v1/realtime`（OpenAI Realtime の形）で、音声は **24 kHz** の PCM を base64 で送る
- **CrispASR**：TTS は `"stream": true`。逐次に音を出すのは Qwen3-TTS だけで、ほかは 1 文ずつ作ってから送る。
  ASR は `--ws-port` を付けて立ち上げると開く vLLM Realtime の WebSocket（`/v1/realtime`、16 kHz）。途中経過
  （`partial_transcription`）を出すのは、もともとストリーミングのモデル（Nemotron）だけで、ほかのモデルは commit を送って
  から全体を 1 回認識する（`session.update` でも変えられなかった）。Qwen3-ASR は、途中まで文字を送ったあと確定の文字（`completed`）を
  返さず、接続を開いたまま止まることがある（timing100 の 34 発話中 16 発話。長い発話ほど多い。同じ発話で測り直しても同じ
  ところで止まる）。サーバーの Realtime の処理（`examples/server/realtime_server.cpp`）は、受けた要求の処理で起きた例外を
  `catch (...) {}` で何も記録せずに捨てるので、確定の文字を作る途中の例外で黙って止まっているとみられる（中身は確かめていない）。0.8.54 ではこの WebSocket が HTTP と **同じ番号** で `0.0.0.0` に開き、HTTP は
  `127.0.0.1` に開くので、`127.0.0.1` へつなぐと HTTP 側に届いて 404 になる。`127.0.0.2` へつなぐと WebSocket 側に届く

## 比べるための元の実装（PyTorch）

ggml の実行系の値が、移植で変わっていないかを見るために、同じモデルを公式の実装でも測る。依存がぶつかるので、
モデルごとに専用の環境を作る（PyTorch は RTX 50 系が動く CUDA 12.8 版）。

```bash
T="torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
# Qwen3-TTS（qwen-tts は transformers 4.57.3 に固定）
uv venv .venv-qwen-tts --python 3.11 && uv pip install --python .venv-qwen-tts/Scripts/python.exe $T
uv pip install --python .venv-qwen-tts/Scripts/python.exe qwen-tts==0.1.1
# OmniVoice（transformers 5.3 以降）
uv venv .venv-omnivoice --python 3.11 && uv pip install --python .venv-omnivoice/Scripts/python.exe $T
uv pip install --python .venv-omnivoice/Scripts/python.exe omnivoice==0.2.1
# CosyVoice3（公式リポジトリ。requirements.txt は古い PyTorch に固定なので、推論に要るものだけ入れる）
git clone --recursive https://github.com/FunAudioLLM/CosyVoice models/src/CosyVoice
uv venv .venv-cosyvoice --python 3.11 && uv pip install --python .venv-cosyvoice/Scripts/python.exe $T
uv pip install --python .venv-cosyvoice/Scripts/python.exe conformer==0.3.2 diffusers==0.29.0 hydra-core==1.3.2 \
    HyperPyYAML==1.2.3 inflect==7.3.1 librosa lightning omegaconf==2.3.0 onnxruntime openai-whisper \
    x-transformers==2.11.24 wetext==0.0.4 transformers==4.51.3 numpy==1.26.4 pyarrow==18.1.0 pyworld wget \
    soundfile matplotlib huggingface_hub rich gdown
```

- 構成の名前は `qwen3-tts-0.6b-base` などで、`apps/tts_engines.toml` の節の `python` にその環境を書く
  （`apps/tts_engines.example.toml`）。モデルは `models/hf` に取ってくる
- CosyVoice3 は、公式の例のとおり日本語をカタカナで渡す（設定の `kana = true`。pyopenjtalk のある親の環境で直してから渡す）。
  漢字のまま渡すと読みが崩れる（CrispASR 版で読みの CER 0.44、カタカナで 0.016）
- Windows の torchaudio 2.9 以降の `load` は torchcodec（FFmpeg の DLL）を使うので、soundfile で読むように差し替えている
- ASR の ReazonSpeech NeMo v2 と Parakeet TDT-CTC 0.6B ja は、Haas-ja と同じ `.venv-haas`（NeMo）で動く

ほかの TTS（ggml の実行系には無いが、日本語でよく使われるもの）:

```bash
# Style-Bert-VITS2（pip 版）。pyopenjtalk-dict が numpy 1 向けなので numpy<2、BERT の型がずれるので transformers<4.50
uv venv .venv-sbv2 --python 3.11
uv pip install --python .venv-sbv2/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sbv2/Scripts/python.exe style-bert-vits2 soundfile "numpy<2" "transformers<4.50" "setuptools<81"
# GPT-SoVITS（公式リポジトリ）。jieba_fast・pyopenjtalk は公式が配る Windows 用の wheel（コンパイラ不要）を入れる
git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS models/src/GPT-SoVITS
uv venv .venv-gptsovits --python 3.12
uv pip install --python .venv-gptsovits/Scripts/python.exe torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
#   https://huggingface.co/lj1995/GPT-SoVITS-windows-package の wheels/cu128/ の jieba_fast と pyopenjtalk、langsegment を入れてから
#   requirements.txt（先頭の --no-binary=opencc を除く）を入れる。重みは XXXXRT/GPT-SoVITS-Pretrained の pretrained_models
#   から v2ProPlus に要るもの（chinese-hubert-base・chinese-roberta-wwm-ext-large・fast_langdetect・s1v3.ckpt・sv・
#   v2Pro/s2Gv2ProPlus.pth）を GPT_SoVITS/pretrained_models へ。Open JTalk の辞書を pyopenjtalk のフォルダーへ展開する
```

- Style-Bert-VITS2 の声は、公式の例の JVNV コーパス（CC BY-SA 4.0）のモデル `jvnv-F1-jp`。AivisSpeech は同じ系統の
  モデルを ONNX で動かす
- GPT-SoVITS は学習なしで、上と同じ参照音声とその書き起こしから声をクローンする（v2ProPlus）
- ASR の anime-whisper（Kotoba-Whisper v2.0 をアニメ・ゲームの台詞で追加学習）は、transformers・faster-whisper
  （有志の CTranslate2 変換）・CrispASR（有志の whisper.cpp 変換）の 3 通りで動かす

測定の結果には、読み込み時間（`load_s`）と、GPU のメモリ（読み込む前からの増え方の最大、`gpu_mem_mb`）、RAM
（測定の処理とサーバーなどの子プロセスの合計の最大、`ram_mb`）も入る（`runtimes/memwatch.py`）。
