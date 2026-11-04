<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab -->

# Packed ternary layout (T2)

## Source: PTQ1_0 in the GGUF

PrismML's PTQ1_0 block holds 128 weights in 28 bytes:

| bytes | content |
|---|---|
| 0..15 (`qs`) | 5 trits per byte (base-3 digits), elements `n*16 + m` for digit n = 0..4 |
| 16..23 | 5 trits per byte, elements 80..119 |
| 24..25 (`qh`) | 4 trits per byte, elements 120..127 |
| 26..27 | fp16 scale `d` |

A trit is read the way prism's dequantizer reads it: `((uint8)(b * 3^k) * 3) >> 8` gives `trit + 1` in {0, 1, 2}.
That's 1.75 bits per weight including the scale.

## Resident form on the GPU

Each `[N][K]` matrix is stored as two arrays:

```
codes   uint8  [N][K/4]     2 bits per weight, value = trit + 1 in {0,1,2}
scales  fp16   [N][K/128]   the block scale, unchanged
```

That's 2.125 bits per weight, or 34 bytes per 28-byte source block (×1.214).

Inside one 128-weight block (32 code bytes), column `e` sits in:

```
byte  = (e / 32) * 8 + ((e / 16) % 2) * 4 + (e % 4)
shift = ((e / 4) % 4) * 2
```

Each 32-column sub-block is therefore two little-endian uint32 words. In the word for columns `16h .. 16h+15`, slot `k` (`(word >> 2k) & 0x03030303`) holds columns `16h + 4k + 0..3`, one per byte. That matches the int8x4 layout of a Q8 activation word for the same 4 columns, so one `__dp4a` multiplies 4 codes by 4 activations with no unpacking:

```
code_dot = sum_k dp4a((w >> 2k) & 0x03030303, act_word[k])     (8 dp4a per 32 columns)
trit_dot = code_dot - sum(q)                                   (code = trit + 1)
y       += (fp16 scale * activation scale d) * trit_dot        (fp32)
```

`sum(q)` per 32-element activation block is computed once in the activation prep. The two-word sub-block lets one lane load 8 bytes (`uint2`) per row per sub-block, so a warp's loads are fully coalesced.

## Activations (Q8_0)

Activations are quantized exactly like ggml's `quantize_row_q8_0`:

- blocks of 32
- `d = fp16(amax / 127)`
- `q = nearbyint(x * 127 / amax)`

The prep kernel also stores `sum(q)` per block. Before quantization it applies the prism Hadamard input transform of the weight it feeds: the optional tiled→grouped permutation, the sign flip, then the normalized blockwise WHT.

## Repack

`btf::repack_rows` (CPU) and `repack_kernel` (GPU) share `pack_block_get` in `include/btf/packed.hpp`. The GPU reads each source block as 7 uint32 words and writes 2 × uint4 of codes plus the scale. `test_packed` checks every packed code against the reference trit decoder (`bd::ptq1_0_trit`) and checks the packed matvec against `bd::matvec_raw`. On the real model, `btf_plan --repack` repacks all 401 PTQ1_0 tensors on the CPU and checks the first row of each chunk against the dequantizer (`--verify-all` checks every element).
