#!/usr/bin/env python
"""Nemotron 3.5 ASR streaming のサブプロセスサーバ。

専用 venv（nemo_toolkit + CUDA torch を導入）で実行し、アプリ本体（ASRService）から
ローカル TCP ソケットで接続される。モデル推論（feed/finalize/reset）だけを担い、
VAD/発話区切り/結果発行はアプリ側で行う。

プロトコルは nemotron_asr.send_frame / recv_frame（length-prefixed JSON）に準拠:
  - {"op":"feed","audio_b64": "<float32-le base64>"} -> {"text": "..."}
  - {"op":"finalize"}                                 -> {"text": "..."}
  - {"op":"reset"}                                    -> {"ok": true}
  - {"op":"ping"}                                     -> {"ok": true}

起動シーケンス:
  1. モデルをロード（数十秒）
  2. ロード完了後に bind/listen（= 接続できた時点で ready）
  3. 1 クライアントを accept し、切断/エラーで終了
"""

from __future__ import annotations

import argparse
import socket
import sys

# このスクリプトと同じディレクトリに nemotron_asr.py がある前提（cwd 非依存）。
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from nemotron_asr import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_WINDOW_SEC,
    NemotronStreamingASR,
    decode_audio_f32,
    recv_frame,
    send_frame,
)


def _log(msg: str) -> None:
    print(f"[nemotron-server] {msg}", file=sys.stderr, flush=True)


def _serve_client(conn: socket.socket, asr: NemotronStreamingASR) -> None:
    conn.settimeout(None)
    while True:
        try:
            req = recv_frame(conn)
        except (ConnectionError, OSError):
            _log("client disconnected")
            return
        except (ValueError, UnicodeDecodeError):
            # 不正/ゼロ長フレーム（JSON/UTF-8 デコード失敗）はフレーム同期が崩れているため
            # 切断扱いで抜ける（プロセス全体を巻き込んで落とさない）。
            _log("malformed frame; closing client")
            return
        if not isinstance(req, dict):
            _log("malformed frame (non-dict); closing client")
            return
        op = str(req.get("op", "")).strip().lower()
        try:
            if op == "feed":
                audio = decode_audio_f32(str(req.get("audio_b64", "")))
                send_frame(conn, {"text": asr.feed(audio)})
            elif op == "finalize":
                send_frame(conn, {"text": asr.finalize()})
            elif op == "reset":
                asr.reset()
                send_frame(conn, {"ok": True})
            elif op == "ping":
                send_frame(conn, {"ok": True})
            else:
                send_frame(conn, {"error": f"unknown op: {op}"})
        except Exception as ex:  # noqa: BLE001
            try:
                send_frame(conn, {"error": str(ex)})
            except Exception:  # noqa: BLE001
                return


def main() -> int:
    parser = argparse.ArgumentParser(description="Nemotron streaming ASR subprocess server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=320)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--max-symbols", type=int, default=DEFAULT_MAX_SYMBOLS)
    parser.add_argument("--lang", default="ja-JP")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _log(f"loading model: {args.model} chunk_size={args.chunk_size}ms ...")
    asr = NemotronStreamingASR(
        model_name=args.model,
        chunk_size_ms=args.chunk_size,
        target_lang=args.lang,
        strip_lang_tags=True,
        device=args.device,
        window_sec=args.window_sec,
        max_symbols=args.max_symbols,
        log=_log,
    )

    # モデルロード完了後に listen する（接続成功 = ready の合図）。
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    _log(f"ready, listening on {args.host}:{args.port}")
    print("READY", flush=True)

    try:
        while True:
            try:
                conn, addr = server.accept()
            except OSError:
                break
            _log(f"client connected: {addr}")
            try:
                _serve_client(conn, asr)
            finally:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            # 1 クライアント切断で終了（アプリの ASR 停止に追従）。
            break
    finally:
        try:
            server.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
