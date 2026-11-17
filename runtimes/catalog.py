"""実行系ごとのモデルの名前と立ち上げ方（runtimes/servers.py が使う）。

ASR[実行系][名前] = その実行系での指定。名前は asrkit の構成名 `<実行系>:<名前>`（例 speechcpp:haas-ja）に使う。
"""

ASR: dict[str, dict[str, dict]] = {
    "crispasr": {
        "haas-ja": {"backend": "haas"},
    },
    "speechcpp": {
        "haas-ja": {"model": "haas-ja-v1.0"},
    },
    "audiocpp": {
        "haas-ja": {"family": "haas_asr", "path": "Haas-ja-v1.0-GGUF/haas-ja-v1.0-q8_0.gguf", "package": "haas_ja_v1_q8_0"},
    },
}

TTS: dict[str, dict[str, dict]] = {
    "crispasr": {},
    "speechcpp": {},
    "audiocpp": {},
}
