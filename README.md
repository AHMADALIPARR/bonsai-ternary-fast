# bonsai-ternary-fast

![license](https://img.shields.io/badge/license-AGPL--3.0--only%20OR%20SnapKitty--Opaque--1.0-blue)
![CUDA](https://img.shields.io/badge/CUDA-sm__80%20%7C%2086%20%7C%2087%20%7C%2089%20%7C%2090-76b900)
![C++17](https://img.shields.io/badge/C%2B%2B-17-informational)
![python](https://img.shields.io/badge/python-none-lightgrey)

Fast GPU decode for PrismML **Ternary-Bonsai-2-27B** (`PTQ1_0` GGUF):

- The weights are repacked once at load into a 2-bit ternary layout and stay resident on the GPU.
- Every matmul is a fused ternary GEMV/GEMM: `dp4a` on packed codes against Q8 activations, with the fp16 block scale applied in the epilogue.
- One decode step is one CUDA graph launch.

The model needs 7.29 GB of device memory at 4K context. It targets desktop and datacenter GPUs (sm_80/86/89/90) and Jetson Orin (sm_87, unified memory).

## Quick start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBTF_MODEL_GGUF=/path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
cmake --build build -j
ctest --test-dir build --output-on-failure
build/btf_bench -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

See [docs/running.md](docs/running.md) for Jetson, memory and the tools.

## Benchmarks

Measured numbers only. GPU cells stay empty until they've been measured on real hardware.

| what | hardware | result |
|---|---|---|
| decode, batch 1, CUDA graph (tok/s) | GPU | |
| prefill, 512 tokens (tok/s) | GPU | |
| load + repack to GPU (s) | GPU | |
| CPU repack of all 401 PTQ1_0 tensors | 8 vCPU Intel Xeon, 16 GB | 5.878 GB in 2.21 s (2.65 GB/s) |
| CPU reference (bonsai-decoder), 54-token prompt + 54 generated | 8 vCPU Intel Xeon, 16 GB | 135.7 s |
| CPU emulator of the GPU math, same run | 8 vCPU Intel Xeon, 16 GB | 194.6 s, 54/54 greedy tokens equal to the reference |
| per-token breakdown, CPU emulator (`btf_bench --emu -p 8 -n 8 -r 1`) | 8 vCPU Intel Xeon, 16 GB | 0 s weight transfer + 2.342 s compute = 2.342 s/token (0.427 tok/s) |

### Against the current adapter

`btf_bench` prints the same split as the adapter numbers: weight transfer per token, GPU compute per token (CUDA events), total s/token and tok/s.

| decoder | hardware | weight transfer (s/token) | GPU compute (s/token) | total (s/token) | tok/s |
|---|---|---|---|---|---|
| current adapter (measured by the user) | user's GPU (Jetson-class) | ~12 | ~16 | ~30 | ~0.033 |
| bonsai-ternary-fast | Jetson Orin | | | | |
| bonsai-ternary-fast | desktop GPU | | | | |

Weights are copied host-to-device once, at load. Decode steps copy no weight bytes (`test_gpu_decoder` checks this), so the weight-transfer column should read 0.

### Calculated ceiling (not a measurement)

Decode at batch 1 is memory-bound. The minimum traffic per token, from `btf_plan`, is 7,173,221,712 B plus 65,536 B per token in context. That covers the packed matrices, the GDN state read and write, and the small vectors. The ceiling is spec-sheet bandwidth divided by those bytes. Real runs will be below it.

| GPU | spec bandwidth (GB/s) | ceiling at empty context (tok/s) | ceiling at 4096 context (tok/s) |
|---|---|---|---|
| Jetson AGX Orin 32/64 GB | 204.8 | 28.6 | 27.5 |
| Jetson Orin NX 16 GB | 102.4 | 14.3 | 13.8 |
| NVIDIA L4 | 300 | 41.8 | 40.3 |
| RTX 3060 12 GB | 360 | 50.2 | 48.4 |
| RTX 4080 | 716.8 | 99.9 | 96.3 |
| RTX 3090 | 936 | 130.5 | 125.8 |
| RTX 4090 | 1008 | 140.5 | 135.5 |
| A100 80 GB SXM | 2039 | 284.3 | 274.0 |
| H100 SXM | 3350 | 467.0 | 450.2 |

Run `btf_plan -m MODEL --bw GBPS` to get the ceiling for any other bandwidth. The packed matrices are 6.80 GB, not 5.88 GB: the 2-bit layout is 1.214x the file's PTQ1_0 data. That costs 14% more bytes in exchange for decoding codes with plain shifts and `dp4a`.

Device memory (from `btf_plan`): 7,289,316,352 B at n_ctx 4096 and 9,168,364,544 B at n_ctx 32768, plus 337,715,200 B of pinned host memory for the packed token embedding.

## Status

The CPU side is tested on the real model:

- repack
- memory plan
- the emulator, which matches the CPU reference token for token on the golden prompt

The CUDA kernels build for all five archs with 0 spills and 0 stack (checked by CTest). The GPU tests (`test_gpu_kernels`, `test_gpu_decoder`, `golden_greeting`) haven't been run on a GPU yet.

## Docs

- [docs/kernel-design.md](docs/kernel-design.md): decode step, kernels, correctness chain
- [docs/packed-layout.md](docs/packed-layout.md): PTQ1_0 and the resident packed format
- [docs/running.md](docs/running.md): build, test, bench, Jetson and desktop

## License

AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0. See [LICENSE](LICENSE). `ref/` holds a copy of the bonsai-decoder CPU reference ([ref/README.md](ref/README.md)).
