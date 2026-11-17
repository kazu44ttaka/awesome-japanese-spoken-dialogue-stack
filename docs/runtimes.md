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
export SPEECH_MODEL_DIR=D:/awesome-japanese-speech/models/speechcpp
```

- 既定の `--device auto` は「最初の GPU」なので、Intel の内蔵 GPU がある PC ではそちら（`Vulkan0`）で動く。
  `speech devices` で一覧を見て、NVIDIA を `--device Vulkan1` のように指定する
- Vulkan の初回は GPU のシェーダーを作るので遅い（Haas-ja で 1 回目 13 秒、2 回目から 98 ms）。測る前にウォームアップする
