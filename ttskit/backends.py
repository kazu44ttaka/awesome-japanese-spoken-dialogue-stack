"""日本語 TTS の各エンジンを同じ形（文字列 → (波形 float32, サンプリング周波数)）で呼べるようにする。

    from ttskit.backends import BACKENDS, create
    tts = create("aivisspeech")
    voices = tts.voices()                  # [(id, 表示名), ...]
    audio, rate = tts.synthesize("こんにちは。", voice=voices[0][0])

- AivisSpeech / VOICEVOX: 別に起動したエンジンに HTTP でつなぐ（Python 側に重い依存は無い）。
  start_engine() は既定のインストール先にある run.exe を起動する（Windows）
- Supertonic 3: pip install supertonic（ONNX・CPU。初回にモデルを自動ダウンロード）
- Qwen3-TTS: pip install qwen-tts（GPU。transformers の版がぶつかりやすいので別の環境を推奨）
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    name: str
    label: str
    note: str


BACKENDS: dict[str, BackendInfo] = {b.name: b for b in [
    BackendInfo("aivisspeech", "AivisSpeech（エンジンに HTTP で接続）", "AivisSpeech を入れて起動しておく（http://127.0.0.1:10101）"),
    BackendInfo("voicevox", "VOICEVOX（エンジンに HTTP で接続）", "VOICEVOX を入れて起動しておく（http://127.0.0.1:50021）"),
    BackendInfo("supertonic", "Supertonic 3（ONNX・CPU）", "pip install supertonic"),
    BackendInfo("qwen3-tts", "Qwen3-TTS 0.6B CustomVoice（GPU）", "pip install qwen-tts（別の環境を推奨）"),
]}


def split_sentences(text: str) -> list[str]:
    """句点・感嘆符・疑問符・改行で文に分ける（文ごとに合成して、最初の文から再生するため）。"""
    parts = re.split(r"(?<=[。！？!?])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


class VoicevoxCompatible:
    """VOICEVOX 互換の HTTP API（AivisSpeech も同じ形）。"""

    def __init__(self, base_url: str, exe_candidates: list[Path]) -> None:
        self.base_url = base_url.rstrip("/")
        self.exe_candidates = exe_candidates

    def _get(self, path: str, timeout: float = 5.0) -> bytes:
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as resp:
            return resp.read()

    def _post(self, path: str, data: bytes | None = None, timeout: float = 120.0) -> bytes:
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(self.base_url + path, data=data or b"", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def available(self) -> bool:
        try:
            self._get("/version", timeout=1.0)
            return True
        except Exception:  # noqa: BLE001
            return False

    def start_engine(self, use_gpu: bool = True, wait_sec: float = 90.0) -> bool:
        """既定のインストール先のエンジンを起動し、応答するまで待つ（use_gpu で --use_gpu を付ける）。"""
        if self.available():
            return True
        exe = next((p for p in self.exe_candidates if p.is_file()), None)
        if exe is None:
            return False
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen([str(exe), *(["--use_gpu"] if use_gpu else [])], cwd=str(exe.parent), creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if self.available():
                return True
            time.sleep(1.0)
        return False

    def voices(self) -> list[tuple[str, str]]:
        speakers = json.loads(self._get("/speakers"))
        out = []
        for sp in speakers:
            for st in sp.get("styles", []):
                out.append((str(st["id"]), f"{sp.get('name', '')}（{st.get('name', '')}）"))
        return out

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        import soundfile as sf

        query = json.loads(self._post(f"/audio_query?{urllib.parse.urlencode({'text': text, 'speaker': voice})}"))
        query["speedScale"] = float(speed)
        wav = self._post(f"/synthesis?speaker={urllib.parse.quote(voice)}", json.dumps(query).encode("utf-8"))
        audio, rate = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(rate)


class Supertonic:
    """Supertonic 3（ONNX・CPU。リポジトリは 2026-07 にアーカイブ済み）。"""

    def __init__(self) -> None:
        from supertonic import TTS

        self.tts = TTS(model="supertonic-3", auto_download=True)
        self._styles: dict[str, object] = {}

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [(v, f"{'女性' if v.startswith('F') else '男性'} {v}") for v in
                ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        style = self._styles.get(voice) or self.tts.get_voice_style(voice)
        self._styles[voice] = style
        wav, _dur = self.tts.synthesize(text=text, voice_style=style, lang="ja", total_steps=8, speed=1.05 * speed)
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(self.tts.sample_rate)


class Qwen3TTS:
    """Qwen3-TTS 0.6B CustomVoice（日本語の声は Ono_Anna）。"""

    MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(
            self.MODEL, device_map="cuda:0" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    def available(self) -> bool:
        return True

    def voices(self) -> list[tuple[str, str]]:
        return [("Ono_Anna", "Ono_Anna（日本語）")]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        wavs, rate = self.model.generate_custom_voice(
            text=text, language="Japanese", speaker=voice, instruct="",
            max_new_tokens=min(1024, max(256, len(text) * 6 + 120)), do_sample=True, temperature=0.9,
        )
        return np.asarray(wavs[0], dtype=np.float32), int(rate)


def _program_files(*parts: str) -> list[Path]:
    local = os.environ.get("LOCALAPPDATA", "")
    return [Path(local, "Programs", *parts)] if local else []


def create(name: str):
    if name == "aivisspeech":
        return VoicevoxCompatible("http://127.0.0.1:10101", _program_files("AivisSpeech", "AivisSpeech-Engine", "run.exe"))
    if name == "voicevox":
        return VoicevoxCompatible("http://127.0.0.1:50021", _program_files("VOICEVOX", "vv-engine", "run.exe"))
    if name == "supertonic":
        return Supertonic()
    if name == "qwen3-tts":
        return Qwen3TTS()
    raise ValueError(f"unknown backend: {name}")
