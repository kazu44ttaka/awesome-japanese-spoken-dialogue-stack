# LLM

日本語の知識や推論の力は、既存のリーダーボード（Nejumi・Swallow）に任せます。ここでは音声対話で効く値を測っています。

- **最初の文字までの時間（TTFT）**：話し終わってから返事が始まるまでの時間に直結する
- **文字の出る速さ（tokens/s）**：読み上げ（TTS）より遅いと途中で詰まる

| 内容 | ページ |
|---|---|
| 10 モデルの TTFT と速さ（LM Studio） | [2026-11-lmstudio.md](2026-11-lmstudio.md) |
| 同じモデルを LM Studio・llama.cpp（CUDA・Vulkan）で比べた結果 | [実行系の比較](../2026-11-runtimes.md#llm) |
