# デモアプリ

| アプリ | 内容 |
|---|---|
| [asr_demo.py](asr_demo.py) | マイクの声をその場で認識する。VAD で区切って確定・疑似ストリーミング・真のストリーミング（Nemotron） |
| [tts_demo.py](tts_demo.py) | 文を読み上げ、最初の音が出るまでの時間と速さを見る。文ごとに合成して、できた文から再生できる |

どちらも Python 3.11 以降と tkinter（Python に同梱）を使う。

## ASR デモ

```
uv venv --python 3.11 .venv
uv pip install --python .venv sherpa-onnx sounddevice soundfile numpy
# transformers / funasr のモデルも使うなら
uv pip install --python .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv "transformers>=5.16" funasr
.venv/Scripts/python apps/asr_demo.py
```

sherpa-onnx のモデル（Nemotron を含む）は、選んで「読み込み」を押すと models/ に自動でダウンロードする。

## TTS デモ

エンジンごとに必要なものが違い、ライブラリの版もぶつかりやすいので、ASR とは別の環境（`.venv-tts`）を作る。

```
uv venv --python 3.11 .venv-tts
uv pip install --python .venv-tts torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-tts sounddevice soundfile numpy
.venv-tts/Scripts/python apps/tts_demo.py
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

確認した組み合わせ（2026-11、Windows 11・RTX 5070 Ti Laptop）では、上のすべてを 1 つの `.venv-tts` に入れて、Qwen3-TTS（PyTorch）以外のエンジンが同時に動いた。
