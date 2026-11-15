"""Nemotron streaming chunk size helpers (NeMo + sherpa-onnx ONNX).

Multilingual Nemotron 3.5 uses att_context right_context 3 -> 320 ms chunk
(same as nemotron_asr.py / NVIDIA model card). k2-fsa export script originally
listed 160 ms (English-only convention); MaAI-SDS standardizes on 320 ms.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

# NeMo + sherpa-onnx multilingual: 80 / 320 / 560 / 1120 ms
NEMOTRON_CHUNK_SIZES_MS = (80, 320, 560, 1120)

CHUNK_TO_RIGHT_CONTEXT = {80: 0, 320: 3, 560: 6, 1120: 13}

_CHUNK_MS_IN_PATH_RE = re.compile(r"-(\d+)ms(?=-|/|\\|\.|$)")
_ONNX_INT8_SUFFIX = ".int8.onnx"


def swap_int8_onnx_path_to_fp32(path: str) -> str:
    """``encoder.int8.onnx`` -> ``encoder.onnx`` (directory / chunk segment unchanged)."""
    text = (path or "").strip()
    if text.endswith(_ONNX_INT8_SUFFIX):
        return text[: -len(_ONNX_INT8_SUFFIX)] + ".onnx"
    return text


def swap_fp32_onnx_path_to_int8(path: str) -> str:
    """``encoder.onnx`` -> ``encoder.int8.onnx`` (idempotent; directory unchanged)."""
    text = (path or "").strip()
    if not text or text.endswith(_ONNX_INT8_SUFFIX):
        return text
    if text.endswith(".onnx"):
        return text[: -len(".onnx")] + _ONNX_INT8_SUFFIX
    return text


def resolve_nemotron_onnx_file_path(path: str, *, prefer_fp32: bool = True) -> str:
    """Prefer fp32 ONNX when ``prefer_fp32`` and ``*.onnx`` exists beside ``*.int8.onnx``."""
    text = (path or "").strip()
    if not text:
        return text
    if not prefer_fp32:
        return text
    fp32 = swap_int8_onnx_path_to_fp32(text)
    if fp32 != text and Path(fp32).is_file():
        return fp32
    return text


def normalize_chunk_size_ms(value: object, *, default: int = 320) -> int:
    try:
        chunk_ms = int(value)
    except (TypeError, ValueError):
        chunk_ms = int(default)
    if chunk_ms not in CHUNK_TO_RIGHT_CONTEXT:
        allowed = ", ".join(str(x) for x in NEMOTRON_CHUNK_SIZES_MS)
        raise ValueError(f"未対応の chunk_size_ms={chunk_ms}。利用可能: {allowed}")
    return chunk_ms


def resolve_nemotron_onnx_chunk_ms(value: object, *, default: int = 320) -> int:
    """ONNX と NeMo で同じ chunk ms（別 ONNX ファイルを差し替え）。"""
    return normalize_chunk_size_ms(value, default=default)


def swap_chunk_ms_in_path(path: str, chunk_ms: int) -> str:
    """Replace ``-560ms-`` style segment in a model path with ``-{chunk_ms}ms-``."""
    text = (path or "").strip()
    if not text:
        return text
    if "{chunk}" in text:
        return text.replace("{chunk}", str(int(chunk_ms)))
    if _CHUNK_MS_IN_PATH_RE.search(text):
        return _CHUNK_MS_IN_PATH_RE.sub(f"-{int(chunk_ms)}ms", text, count=1)
    return text


def resolve_nemotron_onnx_model_paths(
    *,
    tokens_path: str,
    encoder_path: str,
    decoder_path: str,
    joiner_path: str,
    chunk_ms: int,
    resolve_path: Callable[[str], str],
    prefer_fp32: bool = True,
) -> dict[str, str]:
    chunk_ms = resolve_nemotron_onnx_chunk_ms(chunk_ms)
    out: dict[str, str] = {}
    for name, template in (
        ("tokens", tokens_path),
        ("encoder", encoder_path),
        ("decoder", decoder_path),
        ("joiner", joiner_path),
    ):
        swapped = swap_chunk_ms_in_path(template, chunk_ms)
        resolved = resolve_path(swapped)
        if name != "tokens":
            resolved = resolve_nemotron_onnx_file_path(resolved, prefer_fp32=prefer_fp32)
        out[name] = resolved
    return out


def read_encoder_metadata(encoder_path: str) -> dict[str, str]:
    path = (encoder_path or "").strip()
    if not path or not Path(path).is_file():
        return {}
    try:
        import onnx

        model = onnx.load(path, load_external_data=False)
        return {p.key: p.value for p in model.metadata_props}
    except Exception:  # noqa: BLE001
        return {}


def read_encoder_chunk_size_ms(encoder_path: str) -> int | None:
    meta = read_encoder_metadata(encoder_path)
    raw = meta.get("chunk_size_ms")
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


# Nemotron mel hop = preprocessor window_stride (0.01 s @ 16 kHz).
NEMOTRON_MEL_HOP_MS = 10


def finalize_tail_samples_for_chunk(
    chunk_ms: int,
    *,
    encoder_path: str = "",
    sample_rate: int = 16000,
) -> int:
    """NeMo StreamingSession.finalize pad: (precache + shift) * hop == window_size * hop (mel frames)."""
    _ = resolve_nemotron_onnx_chunk_ms(chunk_ms)
    meta = read_encoder_metadata(encoder_path)
    hop = max(1, int(sample_rate * NEMOTRON_MEL_HOP_MS / 1000))
    ws = meta.get("window_size")
    if ws is not None:
        try:
            return max(0, int(ws)) * hop
        except (TypeError, ValueError):
            pass
    cs = meta.get("chunk_shift")
    if cs is not None:
        try:
            shift = int(cs)
            # export: window_size = chunk_shift + pre_encode_cache_size (typical precache 9 @ 320ms)
            window = shift + 9
            return window * hop
        except (TypeError, ValueError):
            pass
    return 41 * hop  # 320 ms Nemotron default (metadata window_size=41)


def finalize_tail_sec_for_chunk(
    chunk_ms: int,
    *,
    encoder_path: str = "",
    sample_rate: int = 16000,
) -> float:
    samples = finalize_tail_samples_for_chunk(
        chunk_ms, encoder_path=encoder_path, sample_rate=sample_rate
    )
    return samples / float(sample_rate) if sample_rate > 0 else 0.0
