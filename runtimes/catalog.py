"""実行系ごとのモデルの名前と立ち上げ方（runtimes/servers.py が使う）。

ASR[実行系][名前] = その実行系での指定。名前は asrkit の構成名 `<実行系>-<名前>`（例 speechcpp-haas-ja）に使う。
同じモデルは実行系をまたいで同じ名前にする（実行系どうしで比べるため）。量子化はそろえられるものは Q8_0。

- crispasr: backend（CrispASR の --backend）と、url（決めたファイルを取ってくる）か quant（-m auto で選ぶ量子化）
- speechcpp: model（speech.cpp のカタログの名前）
- audiocpp: family と package（tools/model_manager_v2.py のパッケージ名）
"""

ASR: dict[str, dict[str, dict]] = {
    "crispasr": {
        # NeMo 系（haas・reazonspeech・parakeet）は、発話の前後に小さく入った別の人の声まで書き起こし、
        # PyTorch / speech.cpp / audio.cpp より読みの CER が 0.06〜0.13 悪い。F16 の重みでも、VAD の区切りを
        # 止めても（--chunk-seconds 60）同じなので、実装の違いとみられる
        "haas-ja": {"backend": "haas"},
        "reazonspeech-nemo-v2": {"backend": "reazonspeech"},
        # parakeet-ja などは登録表のモデルの別名で、--backend には使えない（backend は parakeet / fastconformer-ctc）
        "parakeet-tdt-ctc-0.6b-ja": {"backend": "parakeet",
                                     "url": "https://huggingface.co/cstr/parakeet-tdt-0.6b-ja-GGUF/resolve/main/parakeet-tdt-0.6b-ja-q8_0.gguf"},
        # この GGUF は元のリポジトリのギャル語版（parakeet-ja-gal.nemo）から変換されたとみられる。出力が
        # ギャル語版の PyTorch とよく似ていて、読みの CER も近い（0.183 と 0.174。通常版は 0.056）
        "parakeet-ctc-1.1b-ja": {"backend": "fastconformer-ctc",
                                 "url": "https://huggingface.co/cstr/parakeet-ctc-1.1b-ja-GGUF/resolve/main/parakeet-ctc-1.1b-ja-q8_0.gguf"},
        "qwen3-asr-0.6b": {"backend": "qwen3"},
        "qwen3-asr-1.7b": {"backend": "qwen3-1.7b"},
        "qwen3-asr-1.7b-ja-anime": {"backend": "qwen3-1.7b",
                                    "url": "https://huggingface.co/cstr/qwen3-asr-1.7b-ja-anime-GGUF/resolve/main/qwen3-asr-1.7b-ja-anime-q8_0.gguf"},
        # FunASR のモデルライセンス（funasr-v1.1）への同意が要る（ユーザーが 2026-11-17 に同意）
        "fun-asr-nano": {"backend": "funasr", "args": ["--accept-license", "funasr-v1.1"]},
        "fun-asr-mlt-nano": {"backend": "fun-asr-mlt-nano", "args": ["--accept-license", "funasr-v1.1"]},
        "sensevoice": {"backend": "sensevoice", "args": ["--accept-license", "funasr-v1.1"]},
        "cohere-transcribe": {"backend": "cohere",
                              "url": "https://huggingface.co/cstr/cohere-transcribe-03-2026-GGUF/resolve/main/cohere-transcribe-q4_k.gguf"},
        "cohere-asr-ja-v0.1": {"backend": "cohere",
                               "url": "https://huggingface.co/TransWithAI/cohere-transcribe-ja-v0.1-GGUF/resolve/main/cohere-asr-ja-v0.1-q8_0.gguf"},
        "granite-4.0-1b": {"backend": "granite",
                           "url": "https://huggingface.co/cstr/granite-speech-4.0-1b-GGUF/resolve/main/granite-speech-4.0-1b-q4_k.gguf"},
        # CrispASR の配布は Q4_K だけ。同じ句をくり返して止まらない発話がある
        "granite-4.1-2b": {"backend": "granite-4.1", "quant": None},
        "voxtral-realtime": {"backend": "voxtral4b"},
        "nemotron-streaming": {"backend": "nemotron", "quant": None},  # CrispASR の配布は F16 だけ
        # 長い録音向けのモデル。短い発話では英語の文や [Music] などを出し、読みの CER は 0.98
        "vibevoice-asr": {"backend": "vibevoice"},
        # 同じ句をくり返して止まらない発話がある
        "lfm2.5-audio-jp": {"backend": "lfm2-audio",
                            "url": "https://huggingface.co/cstr/lfm2-audio-1.5b-jp-GGUF/resolve/main/lfm2-audio-1.5b-jp-q5_k.gguf"},
        # anime-whisper の whisper.cpp 向けの変換（有志。変換した人によると先頭に余計な文字が入ることがある）
        "anime-whisper": {"backend": "whisper",
                          "url": "https://huggingface.co/Aratako/anime-whisper-ggml/resolve/main/ggml-anime-whisper-q8_0.bin"},
        "whisper-large-v3-turbo": {"backend": "whisper",
                                   "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"},
    },
    "speechcpp": {
        "haas-ja": {"model": "haas-ja-v1.0"},
        "reazonspeech-nemo-v2": {"model": "reazonspeech-v2"},
        "parakeet-tdt-ctc-0.6b-ja": {"model": "parakeet-tdt_ctc-0.6b-ja"},
        "qwen3-asr-0.6b": {"model": "qwen3-asr-0.6b"},
        "qwen3-asr-1.7b": {"model": "qwen3-asr-1.7b"},
    },
    "audiocpp": {
        # ライブ認識（ストリーミング）は language を付けると「知らないオプション」で断られる
        "haas-ja": {"family": "haas_asr", "package": "haas_ja_v1_q8_0", "live_language": False},
        "qwen3-asr-0.6b": {"family": "qwen3_asr", "package": "qwen3_asr_0_6b_q8_0"},
        "qwen3-asr-1.7b": {"family": "qwen3_asr", "package": "qwen3_asr_1_7b_q8_0"},
        "fun-asr-nano": {"family": "fun_asr_nano", "package": "fun_asr_nano_2512_q8_0"},
        "sensevoice": {"family": "sense_asr", "package": "sensevoice_small_q8"},
        "cohere-transcribe": {"family": "cohere_asr", "package": "cohere_transcribe_q8_0"},
        # granite5asr_q8_0 は日本語に対応する前の英語だけの重みとみられる（日本語の音声に "what" などを出す）ので測らない
        "voxtral-realtime": {"family": "voxtral_realtime", "package": "voxtral_realtime_q8_0"},
        "nemotron-streaming": {"family": "nemotron_asr", "package": "nemotron_asr_q8_0", "language": "ja-JP"},
        # vibevoice_asr_q8_0 は 1 発話目の応答が返らず時間切れになったので測らない
    },
}

# TTS: voices は既定の声の名前（先頭を使う）。clone は参照音声 data/voices/ref_ja.wav で声をクローンする。
# kana は文をカタカナの読みに直してから渡す（pyopenjtalk）
TTS: dict[str, dict[str, dict]] = {
    "crispasr": {
        "qwen3-tts-0.6b-base": {"backend": "qwen3-tts", "clone": True, "ref_24k": True},
        "qwen3-tts-0.6b-customvoice": {"backend": "qwen3-tts-customvoice", "voices": ["ono_anna"]},
        "qwen3-tts-1.7b-base": {"backend": "qwen3-tts-1.7b-base", "clone": True, "ref_24k": True},
        "qwen3-tts-1.7b-customvoice": {"backend": "qwen3-tts-1.7b-customvoice", "voices": ["ono_anna"]},
        # CrispASR は LLM の温度の既定が 0（貪欲）で、20 文中 3 文が止まらず 30 秒まで伸びた。
        # モデルの既定（audio.cpp と同じ 0.8）に合わせる
        "miotts-0.6b": {"backend": "miotts", "args": ["-tp", "0.8"]},
        "moss-tts-local": {"backend": "moss-tts-local", "clone": True, "ref_text": False},
        "voxcpm2": {"backend": "voxcpm2-tts", "clone": True},
        "omnivoice": {"backend": "omnivoice", "args": ["--accept-license", "omnivoice-terms",  # 同意済み（2026-11-17）
                                                       "--codec-model", "models/crispasr/omnivoice-tokenizer-q8_0.gguf"],
                      # 音声のトークナイザーはサーバーが自分では取ってこないので、先に取っておく
                      "prefetch": ["https://huggingface.co/cstr/omnivoice-GGUF/resolve/main/omnivoice-tokenizer-q8_0.gguf"]},  # 同意済み（2026-11-17）
        "supertonic-3": {"backend": "supertonic", "voices": ["F1"], "quant": None, "args": ["--accept-license", "openrail-m"]},  # 同意済み（2026-11-17）
        # 公式の例のとおり、日本語は文も参照音声の書き起こしもカタカナで渡す（kana）。漢字のまま渡すと
        # 中国語に近い読みが混ざって崩れる（読みの CER 0.44 → カタカナで 0.016）
        "cosyvoice3": {"backend": "cosyvoice3-tts", "clone": True, "quant": None, "kana": True},
        "confucius4-tts": {"backend": "confucius4-tts", "clone": True, "ref_text": False},
        "irodori-tts-500m-v3": {"backend": "irodori-tts", "quant": None},
        "irodori-tts-600m-v3-voicedesign": {"backend": "irodori-tts", "url": "https://huggingface.co/cstr/irodori-tts-voicedesign-GGUF/resolve/main/irodori-tts-600m-v3-voicedesign-q4_k.gguf"},
        # kokoro は声のファイル（jf_alpha など）を別に取ってくる必要があり、日本語も崩れやすい（2026-08 の測定）ので測らない
        # zonos（v0.1）は日本語を音素にできず（no phoneme tokens）合成しないので測らない
    },
    "speechcpp": {
        "qwen3-tts-0.6b-customvoice": {"model": "qwen3-tts-0.6b", "voices": ["ono_anna"]},
        "qwen3-tts-1.7b-customvoice": {"model": "qwen3-tts-1.7b", "voices": ["ono_anna"]},
        "irodori-tts-v4-small": {"model": "irodori-tts", "voices": ["none"]},  # none = 参照音声なし
        "irodori-tts-v4-small-mf": {"model": "irodori-tts-mf", "voices": ["none"]},
        "irodori-tts-large": {"model": "irodori-tts-large", "voices": ["none"]},
    },
    "audiocpp": {
        "qwen3-tts-0.6b-base": {"family": "qwen3_tts", "package": "qwen3_tts_0_6b_base_q8_0", "clone": True},
        "qwen3-tts-1.7b-base": {"family": "qwen3_tts", "package": "qwen3_tts_1_7b_base_q8_0", "clone": True},
        "qwen3-tts-1.7b-customvoice": {"family": "qwen3_tts", "package": "qwen3_tts_1_7b_customvoice_q8_0",
                                       "voices": ["ono_anna"]},
        "irodori-tts-v4-small": {"family": "irodori_tts", "package": "irodori_tts_v4_small_q8_0"},
        "irodori-tts-v4-small-mf": {"family": "irodori_tts", "package": "irodori_tts_v4_1_small_mf_q8_0"},
        "irodori-tts-v4-anime": {"family": "irodori_tts", "package": "irodori_tts_v4_1_anime_q8_0"},
        "irodori-tts-500m-v3": {"family": "irodori_tts", "package": "irodori_tts_500m_v3_q8_0"},
        "irodori-tts-600m-v3-voicedesign": {"family": "irodori_tts", "package": "irodori_tts_600m_v3_voicedesign_q8_0"},
        "miotts-1.7b": {"family": "miotts", "package": "miotts_1_7b_q8_0", "clone": True, "ref_text": False,
                        # 参照音声が必須。音を作るコーデック（MioCodec）は別パッケージ
                        "session_packages": {"miotts.codec_model_path": "miocodec_q8_0"}},
        "moss-tts-nano": {"family": "moss_tts_nano", "package": "moss_tts_nano_100m_q8_0", "clone": True, "ref_text": False},
        "moss-tts-local": {"family": "moss_tts_local", "package": "moss_tts_local_v1_5_q8_0", "clone": True, "ref_text": False},
        # IndexTTS2 / 2.5 は日本語に対応していない（同じ音のくり返しになり、読みの CER 0.87〜0.96）ので外した
        # ストリーミングは失敗した生成のやり直し（retry_badcase）を止めないと動かない
        "voxcpm2": {"family": "voxcpm2", "package": "voxcpm2_q8_0", "clone": True,
                    "stream_load": {"default_request_options": {"retry_badcase": False}}},
        "omnivoice": {"family": "omnivoice", "package": "omnivoice_q8_0"},
        "supertonic-3": {"family": "supertonic", "package": "supertonic_3_orig", "voices": ["F1"]},
        "confucius4-tts": {"family": "confucius4_tts", "package": "confucius4_tts_orig", "clone": True, "ref_text": False},
        "fish-s2-pro": {"family": "fish_audio", "package": "fish_audio_s2_pro_q8_0", "clone": True},
        "higgs-audio-v3-4b": {"family": "higgs_audio_tts", "package": "higgs_audio_tts_4b_q8_0", "clone": True},
        "outetts-1.0-1b": {"family": "outetts", "package": "outetts_1_0_1b_q8_0", "clone": True},
    },
}

# アプリ（apps/）の一覧に出す名前
RUNTIME_LABELS = {"crispasr": "CrispASR", "audiocpp": "audio.cpp", "speechcpp": "speech.cpp・Vulkan"}
MODEL_LABELS = {
    # ASR
    "haas-ja": "Haas-ja v1.0", "reazonspeech-nemo-v2": "ReazonSpeech NeMo v2",
    "parakeet-tdt-ctc-0.6b-ja": "Parakeet TDT-CTC 0.6B ja", "parakeet-ctc-1.1b-ja": "Parakeet CTC 1.1B ja",
    "qwen3-asr-0.6b": "Qwen3-ASR 0.6B", "qwen3-asr-1.7b": "Qwen3-ASR 1.7B", "qwen3-asr-1.7b-ja-anime": "Qwen3-ASR 1.7B ja-anime",
    "fun-asr-nano": "Fun-ASR-Nano-2512", "fun-asr-mlt-nano": "Fun-ASR-MLT-Nano-2512", "sensevoice": "SenseVoiceSmall",
    "cohere-transcribe": "Cohere Transcribe", "cohere-asr-ja-v0.1": "Cohere ASR ja v0.1",
    "granite-4.0-1b": "Granite 4.0 1B Speech", "granite-4.1-2b": "Granite Speech 4.1 2B",
    "voxtral-realtime": "Voxtral Mini 4B Realtime", "nemotron-streaming": "Nemotron 3.5 streaming 0.6B",
    "vibevoice-asr": "VibeVoice-ASR", "lfm2.5-audio-jp": "LFM2.5-Audio-1.5B-JP", "anime-whisper": "anime-whisper",
    "whisper-large-v3-turbo": "Whisper large-v3-turbo",
    # TTS
    "qwen3-tts-0.6b-base": "Qwen3-TTS 0.6B Base", "qwen3-tts-0.6b-customvoice": "Qwen3-TTS 0.6B CustomVoice",
    "qwen3-tts-1.7b-base": "Qwen3-TTS 1.7B Base", "qwen3-tts-1.7b-customvoice": "Qwen3-TTS 1.7B CustomVoice",
    "miotts-0.6b": "MioTTS 0.6B", "miotts-1.7b": "MioTTS 1.7B", "moss-tts-local": "MOSS-TTS-Local v1.5",
    "moss-tts-nano": "MOSS-TTS-Nano", "voxcpm2": "VoxCPM2", "omnivoice": "OmniVoice", "supertonic-3": "Supertonic 3",
    "cosyvoice3": "CosyVoice3 0.5B", "confucius4-tts": "Confucius4-TTS", "irodori-tts-500m-v3": "Irodori-TTS 500M v3",
    "irodori-tts-600m-v3-voicedesign": "Irodori-TTS 600M v3 VoiceDesign", "irodori-tts-v4-small": "Irodori-TTS v4 Small",
    "irodori-tts-v4-small-mf": "Irodori-TTS v4.1 Small MF", "irodori-tts-v4-anime": "Irodori-TTS v4.1 Anime",
    "irodori-tts-large": "Irodori-TTS Large", "fish-s2-pro": "Fish Audio S2 Pro", "higgs-audio-v3-4b": "Higgs Audio TTS 3 4B",
    "outetts-1.0-1b": "OuteTTS 1.0 1B",
}
# 実行系のストリーミングをアプリの真のストリーミングとして使う構成（ほかはアプリの疑似ストリーミングで動かす）
APP_STREAMING_ASR = {"audiocpp-nemotron-streaming", "audiocpp-voxtral-realtime", "crispasr-nemotron-streaming"}


def label(runtime: str, key: str, spec: dict | None = None) -> str:
    """アプリの一覧に出す名前。例: Haas-ja v1.0（speech.cpp・Vulkan）、Qwen3-TTS 0.6B Base（audio.cpp・クローン）"""
    extra = "・クローン" if spec and spec.get("clone") else ""
    return f"{MODEL_LABELS.get(key, key)}（{RUNTIME_LABELS.get(runtime, runtime)}{extra}）"
