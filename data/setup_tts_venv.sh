set -e
cd /d/awesome-japanese-speech
[ -d .venv-tts ] || uv venv -q --python 3.11 .venv-tts
P=.venv-tts/Scripts/python.exe
uv pip install -q --python $P torch==2.11.* torchaudio==2.11.* --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python $P "transformers>=5.12,<6" accelerate soundfile sounddevice numpy huggingface-hub safetensors supertonic \
  "sentencepiece>=0.1.99,<0.2" peft numba llvmlite pyyaml "dacvae @ git+https://github.com/facebookresearch/dacvae" "silentcipher @ git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152"
uv pip install -q --python $P --no-deps "irodori-tts @ git+https://github.com/Aratako/Irodori-TTS"
uv pip install -q --python $P https://github.com/VOICEVOX/voicevox_core/releases/download/0.17.0/voicevox_core-0.17.0-cp310-abi3-win_amd64.whl
uv pip install -q --python $P kokoro "misaki[ja]"
uv pip install -q --python $P munch pydub librosa nltk matplotlib unidic-lite xlstm fugashi cutlet einops einops-exts a_unet konoha setuptools openai "monotonic_align @ git+https://github.com/resemble-ai/monotonic_align.git"
echo TTS-VENV-DONE
