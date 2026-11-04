<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab -->

# Building and running

You need:

- CMake ≥ 3.20 and a C++17 compiler
- the CUDA toolkit (developed with 12.9) for the GPU build
- the model file `Ternary-Bonsai-2-27B-PTQ1_0.gguf`

```sh
MODEL=/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

## Memory

Numbers are from `btf_plan -m $MODEL` and are exact byte counts for this model:

| n_ctx | device arena | pinned host |
|---|---|---|
| 4096 (default) | 7,289,316,352 B (6.79 GiB) | 471,932,928 B |
| 32768 | 9,168,364,544 B (8.54 GiB) | 471,932,928 B |
| 4096, `--embd-device` | 7,627,031,552 B (7.10 GiB) | 134,217,728 B |

Breakdown at n_ctx 4096:

| item | bytes |
|---|---|
| 401 packed matrices | 6,799,564,800 |
| KV cache (16 layers, fp16) | 268,435,456 |
| GDN states (48 layers) | 156,893,184 |
| GDN alpha/beta | 47,185,920 |
| fp32 norms, conv weights, gates | 10,582,016 |
| scratch | 6,540,288 |
| packed token_embd (pinned host, read zero-copy) | 337,715,200 |
| staging (2 × 64 MiB, pinned host, freed after load) | 134,217,728 |

KV cache grows by 65,536 B per token of context. Run `btf_plan -m $MODEL -c N` for any other size. The decoder refuses to start if the arena plus 64 MiB is more than `cudaMemGetInfo` reports free.

## Desktop / datacenter GPU (sm_80, 86, 89, 90)

This needs 7.29 GB of free VRAM at n_ctx 4096, plus a 64 MiB margin and the CUDA context. 8 GB cards are marginal at best; use a 10 GB or larger card.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBTF_MODEL_GGUF=$MODEL
cmake --build build -j
ctest --test-dir build --output-on-failure        # all tests, incl. golden_greeting
build/btf_golden -m $MODEL --ref-tokens tests/data/greeting_ref.txt   # GPU must reproduce the saved CPU tokens
build/btf_bench  -m $MODEL -p 512 -n 128 -r 3      # prefill + decode tok/s
```

To build one arch only, which compiles faster, add e.g. `-DCMAKE_CUDA_ARCHITECTURES=89`. If nvcc isn't on `PATH`, pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

## Jetson Orin (sm_87, unified memory)

On Jetson the "device arena" and "pinned host" both come out of the same DRAM, so you need about 7.8 GB for the model plus the OS. That rules out 8 GB Orin boards; use a 16 GB or larger Orin.

```sh
sudo nvpmodel -m 0 && sudo jetson_clocks          # max clocks (optional)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBTF_MODEL_GGUF=$MODEL
cmake --build build -j4
ctest --test-dir build --output-on-failure
build/btf_golden -m $MODEL --ref-tokens tests/data/greeting_ref.txt
build/btf_bench  -m $MODEL -p 512 -n 128 -r 3
```

Notes for unified memory:

- The loader streams the GGUF through two 64 MiB pinned, mapped slots and repacks on the GPU, so it never holds a second full copy of the weights in RAM. After upload it calls `posix_fadvise(DONTNEED)` on the GGUF, which gives the page cache back. The 5.9 GB of mmapped file pages would otherwise compete with the arena.
- token_embd stays in pinned mapped memory by default. On Jetson that's the same DRAM the GPU reads anyway, so `--embd-device` costs 338 MB more and gains nothing.
- Use `--ref-tokens` with btf_golden. Without it the tool also runs the 27B CPU reference, which keeps the 5.9 GB of GGUF pages resident next to the arena.
- The decode loop copies no weights. `btf_bench` prints `weight transfer host->device` per token, and it should be 0 B/token. Compare its `GPU compute` and `total` lines with the adapter's ~12 s transfer and ~16 s compute per token. `btf_plan -m $MODEL --bw 204.8` prints the AGX Orin calculated ceiling (102.4 for Orin NX).

## Tools

| tool | what it does |
|---|---|
| `btf_plan -m $MODEL [-c N] [--repack] [--bw GBPS]` | memory plan and minimum decode bytes per token; `--bw` prints the calculated ceiling bandwidth / bytes (not a measurement); `--repack` times and verifies the CPU repack of all 401 tensors |
| `btf_golden -m $MODEL [-p TEXT] [-n N]` | runs the CPU reference greedily (slow), then the GPU decoder, and compares tokens and timing |
| `btf_golden -m $MODEL --save-ref F --ref-only` | CPU reference tokens to F (slow: whole 27B model on CPU) |
| `btf_golden -m $MODEL --ref-tokens F [--emu]` | GPU (or CPU emulator) must reproduce F's tokens |
| `btf_bench -m $MODEL [-p 512] [-n 128] [-r 3] [--no-graph] [--emu]` | prefill and decode throughput, best of `-r` reps, plus a per-token breakdown: weight transfer host->device (s and B per token), GPU compute (CUDA events), host overhead, total s/token and tok/s, minimum traffic and effective bandwidth |

## CPU only

Without nvcc, CMake builds the CPU emulator, tools and CPU tests. The GPU tests aren't built. On a CUDA build running on a machine without a GPU, the GPU tests exit 77 and CTest reports them as skipped.

```sh
cmake -S . -B build -DBTF_CUDA=OFF && cmake --build build -j && ctest --test-dir build
```
