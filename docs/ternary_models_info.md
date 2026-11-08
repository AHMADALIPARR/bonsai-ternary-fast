# Ternary Models & PrismML Bonsai Architectural Overview

This document summarizes known ternary models, the PrismML Bonsai model family, their underlying architecture, quantization layouts, and parameters.

---

## 1. Overview of Ternary Models

Ternary neural networks constrain weights to values in $\{-1, 0, +1\}$ (or scaled versions $-s, 0, +s$). Because weights take only 3 states, they can theoretically be stored in $\log_2(3) \approx 1.58$ bits per weight and computed with addition/subtraction rather than floating-point multiplications.

Key families of ternary models include:
1. **PrismML Bonsai Series**:
   - High-performance ternary LLMs developed by PrismML using hybrid Gated DeltaNet (linear attention) + standard multi-head attention (full attention) layers.
   - Employs Walsh-Hadamard Transforms (sign-folding) to condition activations and weights.
   - Uses `PTQ1_0` (Prism Ternary Quantization), `PQ2_0`, and `Q2_0` formats in GGUF.
2. **Microsoft BitNet b1.58 Series**:
   - `1bitLLM/bitnet_b1_58-large`, `bitnet_b1_58-3B`, `bitnet-b1.58-2B-4T`.
   - Uses Absmean quantization where activations are quantized to 8-bit integers and weights are constrained to $\{-1, 0, +1\}$.
3. **Community Pruned & Quantized Ternary Models**:
   - `henceiusegentoo/Ternary-Bonsai-2-24B-PTQ1_0-GGUF` (pruned 64 to 56 layers).
   - `burnboomm/Bonsai-27B-56blk-trim98k`.

---

## 2. PrismML Bonsai Model Family

### Generation 2 Models (`Ternary-Bonsai-2-*`)
| Model Name | Total Parameters | Layers | Hidden Size | Intermediate Size | Attention Heads (Q / KV) | Head Dim | Full Attn Interval |
|---|---|---|---|---|---|---|---|
| **Ternary-Bonsai-2-27B** | ~27B | 64 | 5120 | 17408 | 24 / 4 | 256 | Every 4th layer (16 full, 48 linear) |
| **Ternary-Bonsai-2-9B** | ~9B | 32 | 4096 | 12288 | 16 / 4 | 256 | Every 4th layer (8 full, 24 linear) |
| **Ternary-Bonsai-2-4B** | ~4B | 32 | 2560 | 9216 | 16 / 4 | 256 | Every 4th layer (8 full, 24 linear) |
| **Ternary-Bonsai-2-2B** | ~2B | 24 | 2048 | 6144 | 8 / 2 | 256 | Every 4th layer (6 full, 18 linear) |

### Generation 1 Models (`Ternary-Bonsai-*`)
| Model Name | Total Parameters | Layers | Hidden Size | Intermediate Size | Attention Heads (Q / KV) | Head Dim |
|---|---|---|---|---|---|---|
| **Ternary-Bonsai-27B** | ~27B | 64 | 5120 | 17408 | 24 / 4 | 256 |
| **Ternary-Bonsai-8B** | ~8B | 36 | 4096 | 12288 | 32 / 8 | 128 |
| **Ternary-Bonsai-4B** | ~4B | 36 | 2560 | 9728 | 32 / 8 | 128 |
| **Ternary-Bonsai-1.7B** | ~1.7B | 28 | 2048 | 6144 | 16 / 8 | 128 |

---

## 3. Hybrid Architecture (Qwen 3.5 / Gated DeltaNet)

Bonsai models use a 3:1 hybrid design:
- **3 of every 4 layers**: Linear attention (Gated DeltaNet / SSM-like recurrence).
  - State dimension `d_state = 128`.
  - Recurrent conv1d kernel size = 4.
  - Linear key head dim = 128, value head dim = 128.
  - Time-step rank `dt_rank = 16`.
- **1 of every 4 layers**: Full attention (Sliding window / Standard multi-head self-attention with RoPE).

This hybrid architecture drastically reduces KV-cache memory bandwidth during autoregressive decoding while preserving long-context reasoning capabilities.

---

## 4. Quantization Schemes & Layouts

1. **PTQ1_0 (Prism Ternary Quantization 1.0)**:
   - Block size: 128 weights.
   - Storage per block: 16-bit half-precision float scale (`fp16`) + packed ternary bits.
   - Values are ternary $\{-1, 0, 1\}$.
   - In `bonsai-ternary-fast`, weights are repacked into a GPU-resident 2-bit format where 4 weights occupy a byte, enabling fast `dp4a` dot-product instructions with INT8 activations.

2. **PQ2_0 / Q2_0**:
   - 2-bit integer formats storing affine scale and 2-bit packed codes.

3. **Hadamard Transformations**:
   - Normalized Sylvester-Walsh-Hadamard transform applied along input dimensions to eliminate activation outliers prior to quantization.

---

## 5. 15B Dense Bonsai Model Downscaling & Quantization

To create a 15B dense Bonsai model from the 27B architecture:
- Maintain hidden dimensions ($E = 5120$) and intermediate dimensions ($F = 17408$) to keep model capacity and channel representation intact.
- Downscale layer count from 64 layers to 36 layers (27 linear attention layers + 9 full attention layers with 4-layer periodicity).
- Parameter total: $\approx 14.53\text{B}$ parameters.
- Quantize all feed-forward and attention projection layers to `PTQ1_0` ternary format.
