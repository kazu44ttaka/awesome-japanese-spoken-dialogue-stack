# デモアプリ

| アプリ | 内容 |
|---|---|
| [asr_demo.py](asr_demo.py)・[asr_web.py](asr_web.py) | マイクの声をその場で認識する。VAD で区切って確定・疑似ストリーミング・真のストリーミング（Nemotron） |
| [tts_demo.py](tts_demo.py)・[tts_web.py](tts_web.py) | 文を読み上げ、最初の音が出るまでの時間と速さを見る。文ごとに合成して、できた文から再生できる |

- `*_demo.py` はデスクトップのアプリ（PySide6。[MaAI-SDS](https://github.com/kazu44ttaka/MaAI-SDS) と同じ `pyside6-essentials==6.12.0`）。マイクと再生はこの PC の sounddevice
- `*_web.py` はブラウザで使う版（FastAPI + uvicorn）。マイクと再生はブラウザで、認識・合成はこの PC で行う。起動するとブラウザが開く（ASR は http://127.0.0.1:8765 、TTS は http://127.0.0.1:8766 ）
- 中身は共通（[asr_core.py](asr_core.py)・[tts_core.py](tts_core.py)）なので、どちらでも同じモデル・同じ設定で動く
- ブラウザのマイクは安全な接続でしか使えない。この PC で開く（127.0.0.1 / localhost）なら使えるが、`--host 0.0.0.0` でほかの PC から開くときは HTTPS が要る（TTS の再生はそのまま使える）
- Web 版を同時に使えるのは 1 つのページだけ（あとから開いたページが引き継ぐ）

## ASR デモ

```
uv venv --python 3.11 .venv
uv pip install --python .venv sherpa-onnx sounddevice soundfile numpy "pyside6-essentials==6.12.0" fastapi "uvicorn[standard]"
# transformers / funasr のモデルも使うなら
uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv "transformers>=5.16" funasr
.venv/Scripts/python apps/asr_demo.py   # デスクトップ版
.venv/Scripts/python apps/asr_web.py    # Web 版（--host / --port / --no-browser）
```

sherpa-onnx のモデル（Nemotron を含む）は、選んで「読み込み」を押すと models/ に自動でダウンロードする。

追加の準備が要るエンジン（場所は `apps/asr_engines.example.toml` を `apps/asr_engines.toml` にコピーして書く）:

| エンジン | 追加で入れるもの |
|---|---|
| faster-whisper | `uv pip install --python .venv faster-whisper` |
| Nemotron（NeMo・PyTorch） | NeMo 用の別の環境（`nemo-toolkit[asr]==3.0.0`）。アプリがその環境でサーバーを起動して TCP でつなぐ |
| Qwen3-ASR（PyTorch） | `qwen-asr` は transformers 4.57 を要求して Granite（5.16 以上）とぶつかるので、別の環境 `.venv-qwen-asr` に入れて、その環境でアプリや測定を動かす |
| Voxtral Mini 4B Realtime | `"mistral-common[audio]>=1.9"`。GPU のメモリを約 9 GB 使う |
| Granite 4.0 1B Speech・moonshine-base-ja | なし（transformers で動く） |
| Haas-ja-v1.0・Parakeet CTC 1.1B 日本語 | 別の環境 `.venv-haas`（下の「Haas」）。NeMo 2.7 を使う。`.venv` の asr_demo から選ぶと、`.venv-haas` の子プロセスで自動的に動く（`asrkit/worker.py`） |

Haas（公式は Linux 専用だが、Windows 版の Triton を入れると動く。1 回目の認識はカーネルを作るので 10 秒ほどかかる）:

```
git clone https://github.com/third-intelligence/Haas-ASR models/src/Haas-ASR
uv venv --python 3.11 .venv-haas
uv pip install --python .venv-haas torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-haas "triton-windows<3.8" fugashi unidic-lite
# NeMo 2.7.3 の古い固定を、Haas-ASR の pyproject.toml と同じく上書きする（torch も上書きしないと CPU 版に入れ替わる）
printf 'fsspec[http]>=2026.6.0
hydra-core>=1.3.8
packaging>=26.3
transformers>=5.10.0
torch==2.11.0
torchaudio==2.11.0
' > haas-overrides.txt
uv pip install --python .venv-haas --override haas-overrides.txt -e models/src/Haas-ASR
```

Parakeet CTC 1.1B 日本語（有志による日本語版）も同じ環境で動く。配布元の注意どおり `parakeet-ja-gal.nemo` を使う（もう一方は重みが壊れている）。

## TTS デモ

エンジンごとに必要なものが違い、ライブラリの版もぶつかりやすいので、ASR とは別の環境（`.venv-tts`）を作る。

```
uv venv --python 3.11 .venv-tts
uv pip install --python .venv-tts torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-tts sounddevice soundfile numpy "pyside6-essentials==6.12.0" fastapi "uvicorn[standard]"
.venv-tts/Scripts/python apps/tts_demo.py   # デスクトップ版
.venv-tts/Scripts/python apps/tts_web.py    # Web 版
```

エンジンの場所は `apps/tts_engines.example.toml` を `apps/tts_engines.toml` にコピーして書き換える。

| エンジン | 追加で入れるもの | 用意するもの |
|---|---|---|
| AivisSpeech / VOICEVOX | なし | アプリを入れておく（アプリの「エンジンを起動」で既定の場所から起動できる） |
| VOICEVOX Core | `uv pip install https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.0/voicevox_core-0.17.0-cp310-abi3-win_amd64.whl` | voicevox_core のダウンローダーで取った一式（`runtime_dir`）。声の規約はキャラクターごと |
| Irodori-TTS v4.1 | 下の「Irodori-TTS」 | なし（初回に Hugging Face から取得） |
| Supertonic 3 | `supertonic` | なし（初回に取得） |
| mio-tts-cpp | なし | [mio-tts-cpp](https://github.com/mmnga/mio-tts-cpp) をビルドした `mio-tts-server` とモデル |
| Qwen3-TTS（audio.cpp） | なし | audio.cpp の `audiocpp_server` と Qwen3-TTS CustomVoice の GGUF |
| Qwen3-TTS（PyTorch） | `qwen-tts`（transformers の版がぶつかるので、さらに別の環境に入れる） | なし |
| Tsukasa Speech | 下の「Tsukasa」 | [Respair/Tsukasa_Speech](https://huggingface.co/Respair/Tsukasa_Speech) のフォルダーと、使ってよい参照音声 |
| Kokoro | `kokoro "misaki[ja]"` のあと `python -m unidic download`（辞書 約 500 MB） | なし |
| ZONOS3-Mini | なし | 下の「ZONOS3-Mini」。C++ のサーバーをビルドする |
| Kitten TTS 2 | 別の環境 `.venv-kitten` に `kittenml` | なし（初回に約 8 GB 取得）。商用で使うには Stellon Labs への登録が要る |
| LFM2.5-Audio-1.5B-JP | 別の環境 `.venv-lfm`（Python 3.12）に `liquid-audio` | なし（初回に取得）。声は選べない |
| MOSS-TTS-Nano | 別の環境 `.venv-nano`（Python 3.12）に clone した MOSS-TTS-Nano | CPU で動く。日本語の声は配布物の 1 つ（Yui） |

Irodori-TTS（学習用の gradio・wandb などは入れず、推論に要るものだけ）:

```
uv pip install --python .venv-tts "transformers>=5.12,<6" accelerate huggingface-hub safetensors "sentencepiece>=0.1.99,<0.2" peft numba llvmlite pyyaml ^
  "dacvae @ git+https://github.com/facebookresearch/dacvae" ^
  "silentcipher @ git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152"
uv pip install --python .venv-tts --no-deps "irodori-tts @ git+https://github.com/Aratako/Irodori-TTS"
```

Tsukasa:

```
uv pip install --python .venv-tts munch pydub librosa nltk matplotlib unidic-lite xlstm fugashi cutlet einops einops-exts a_unet ^
  konoha setuptools openai "monotonic_align @ git+https://github.com/resemble-ai/monotonic_align.git"
```

別の環境に入れたエンジンは、`apps/tts_engines.toml` の節に `python = ".venv-kitten/Scripts/python.exe"` のようにその環境の Python を書くと、デモアプリ（`.venv-tts`）から子プロセスとして動かせる（`ttskit/worker.py`）。

```
uv venv --python 3.11 .venv-kitten && uv pip install --python .venv-kitten torch==2.11.* --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python .venv-kitten kittenml soundfile
uv venv --python 3.12 .venv-lfm && uv pip install --python .venv-lfm torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python .venv-lfm liquid-audio soundfile
git clone https://github.com/OpenMOSS/MOSS-TTS-Nano models/src/MOSS-TTS-Nano
uv venv --python 3.12 .venv-nano && uv pip install --python .venv-nano torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python .venv-nano -e models/src/MOSS-TTS-Nano soundfile
```

torch を先に入れても、あとのパッケージが別の版の torch（GPU 非対応）に入れ替えることがある。入れ替わったら `--override` で torch の版を固定して入れ直す。
MOSS-TTS-Nano は Windows では torchaudio の音声の読み書き（torchcodec が FFmpeg の DLL を要る）が動かないので、`ttskit` の中で soundfile に置き換えている。

ZONOS3-Mini（C++/ggml のサーバー。Rust・CMake・Visual Studio 2022・CUDA 12.8 が要る）:

```
git clone https://github.com/Zyphra/ZONOS3-Mini models/src/ZONOS3-Mini
cd models/src/ZONOS3-Mini
git submodule update --init --depth 1 servers/ggml/ggml third_party/zyphra-textnorm
cd servers/ggml
cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --config Release -j 8
hf download Zyphra/ZONOS3-Mini gguf/backbone-ptq.gguf gguf/zac-keep2.gguf gguf/spk-encoder.gguf --local-dir .
hf download Zyphra/ZONOS3-Mini --include "gguf/voices/*" --local-dir .
```

`CMAKE_CUDA_ARCHITECTURES` は GPU に合わせる（RTX 50 シリーズは 120）。`zonos3-cli` は Windows で日本語の引数が文字化けするので、アプリはサーバー（`zonos3-server`）を起動して UTF-8 の JSON で頼む。日本語の既定の声は配布されていないので、ほかの言語の既定の声で日本語を読ませる。

確認した組み合わせ（2026-11、Windows 11・RTX 5070 Ti Laptop）では、上のすべてを 1 つの `.venv-tts` に入れて、Qwen3-TTS（PyTorch）以外のエンジンが同時に動いた。
