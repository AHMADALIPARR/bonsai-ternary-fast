<!-- SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0 -->
<!-- Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab -->

# Kernel design

Decode at batch 1 is bound by weight bandwidth. Every token reads every matrix once. The design keeps those bytes as small as possible and reads each one once:

- the 401 PTQ1_0 matrices stay resident in packed ternary form (2.125 bits per weight)
- the matvec runs straight on the packed codes
- the rest of the step stays on the device, with one sync per token

## Load

`btf::make_gpu_decoder` (`src/cuda/gpu_decoder.cu`) makes one `cudaMalloc` arena sized by `btf::plan` (`src/repack.cpp`). It checks the size against `cudaMemGetInfo` before allocating.

Each matrix streams through two pinned, mapped 64 MiB staging slots:

1. memcpy from the mmapped GGUF into one slot
2. `repack_kernel` reads that slot over PCIe or, on Jetson, straight from DRAM, and writes packed codes and scales into the arena
3. meanwhile the other slot fills

After upload, `posix_fadvise(DONTNEED)` drops the GGUF from the page cache. Nothing is repacked or re-uploaded per token.

## One decode step

The step runs `embed`, then:

| layer | kernel sequence |
|---|---|
| GDN (48) | prep(norm+WHT) · gemv[qkv,z] · ab_gates · conv_silu · gdn · prep(WHT) · gemv+add[out] |
| attention (16) | prep(norm+WHT) · gemv[q,k,v] · attn_prep · attention · attn_combine · prep(WHT) · gemv+add[o] |
| every layer | prep(norm+WHT) · gemv swiglu[gate,up] · prep(WHT) · gemv+add[down] |

At the end come prep(norm) · gemv[output] · argmax · pos_add.

The position lives on the device (`d_pos`) and is bumped by `pos_add`, so the same sequence is valid for every token. It's captured once as a CUDA graph, including the H2D copy of the token id from a pinned host word and the D2H copy of the argmax. Each `step()` is then one `cudaGraphLaunch` plus one stream sync. `--no-graph` launches the same sequence eagerly. `test_gpu_decoder` checks that both give bit-identical logits.

## Bytes per token and the ceiling

Weights are copied host-to-device exactly once. The decoder counts every weight byte it uploads (`wx_bytes_`). `stats()` reports how much of that happened during decode steps, and `test_gpu_decoder` requires it to be 0. On Jetson, the arena is ordinary `cudaMalloc` memory in the shared DRAM, so there is no per-token copy and no managed-memory migration. The only host memory read per step is one 1,360 B packed row of `token_embd`, read zero-copy.

Batch-1 decode reads every resident matrix once per token. `btf_plan` prints the minimum traffic (`MemoryPlan::decode_bytes_fixed` and `decode_bytes_per_ctx`):

- 6,799,564,800 B of packed codes and scales: the 401 `PTQ1_0` tensors at 2.125 bpw
- the alpha/beta matrices and f32 norm and bias vectors
- the GDN recurrent and conv state, read and written: 2 × 48 × (48·128·128 + 3·10240) × 4 B
- a 2 × 248320 × 4 B logits/argmax pass

That totals 7,173,221,712 B per token, plus 65,536 B × tokens in context for the fp16 KV read.

The calculated ceiling is `tok/s ≤ memory bandwidth / bytes per token`. It's a bound, not a measurement. `btf_plan --bw GBPS` prints it; the README has a table for common GPUs.

The file holds 5,877,760,000 B of `PTQ1_0` data, so the resident T2 layout is 14% larger than a native-format kernel would need. T2 was chosen because each code decodes with a shift and a mask straight into `dp4a`, with no base-3 division in the inner loop. A native base-3 GEMV could cut traffic by about 0.92 GB per token, at the cost of more ALU work per byte.

## Kernels (`src/cuda/kernels.cu`)

**prep_kernel** turns the activation into Q8 for the next matmul. The tensor's Hadamard transform and the int8 quantization are fused into one pass. One block per WHT span (up to 1024 floats in smem) does:

1. optional RMSNorm (double sum of squares)
2. the tiled→grouped gather for `gdn_v_grouped`
3. explicit signs
4. an in-smem fast WHT
5. ×1/√block
6. per-32 `amax`, fp16 `d`, int8 `q`, and `sum(q)`

The rounding order matches ggml (`__fmul_rn`, `__float2int_rn`), so the CPU emulator can match it bit for bit.

**gemv_kernel<NB, MODE>** is the fused ternary GEMV/GEMM. It runs 256 threads (`__launch_bounds__(256)`).

- The NB ≤ 8 activation rows (q, d, sum) go into dynamic shared memory once per block. At K = 17408 and NB = 8 that's 174 KB, using the opt-in smem limit.
- Each warp owns two output rows. Lane `s` walks 32-column sub-blocks `s, s+32, …`.
- Per sub-block and row, a lane does one coalesced `uint2` load of 8 code bytes and one fp16 scale load. Then, for each activation row, it does 8 `dp4a` per weight row.
- The epilogue per sub-block is `acc += (scale_fp16 * d_act) * (code_dot - sum_q)`, then a warp shuffle reduction.

There are three modes:

| mode | output | used for |
|---|---|---|
| store | `y = W x` | |
| add | `y += W x` | fused residual add for out/o/down |
| swiglu | `y = silu(W_gate x) * (W_up x)` | gate and up rows share the activation |

One launch can cover up to three matrices with the same K (q/k/v, qkv/z, gate/up) through a cumulative unit index, so the activation is staged once. M > 8 prompt rows are split into launches of ≤ 8. The same kernel is the batch-1 GEMV (NB = 1) and the small-batch GEMM (NB = 2, 4, 8).

**ab_gates_kernel** handles the GDN alpha/beta projections. These are small bf16/f32 tensors, read as stored and not repacked. It also computes the decay gate `softplus(alpha·x + dt) · a` and `sigmoid(beta·x)`.

**conv_silu_kernel** runs the causal depthwise conv1d (kernel 4) over the qkv stream, then SiLU. The 3-row conv state is carried in registers across the M rows and written back.

**gdn_kernel<DK>** is the gated delta-net recurrence. It uses one block per value head with 4 threads per state row. The 128×128 fp32 state stays in registers for the whole M-row chunk, and the q/k L2-norm runs in smem. After the delta-rule update it produces the output, applies the per-head RMSNorm, and gates it with silu(z).

**attn_prep_kernel**:

- per-head q/k RMSNorm
- partial RoPE on the first 64 dims, base 1e7
- writes K/V for this position into the fp16 KV cache
- writes q (optionally rounded to fp16, as ggml's F16 vec-dot does)

The angle uses `sincospi(theta/π)` in double. That keeps every kernel free of local memory.

**attention_kernel / attn_combine_kernel** do flash-decoding over the KV cache:

- grid (H, nsplit = 16, M)
- each split scores `ceil(T/nsplit)` keys into smem, with T read from `d_pos` at run time so the graph stays valid
- each split writes max, sum, and a partial V
- the combine pass merges the splits and applies the sigmoid output gate

GQA maps 24 q heads onto 4 KV heads.

**embed_kernel** gathers one row of the packed token_embd (the table is ternary too), dequantizes it, and undoes its WHT+signs. The table is in mapped host memory by default (`--embd-device` puts it in the arena). Only 5120/4 bytes per token cross the bus.

**argmax_stage1/2** is a two-stage argmax over V = 248320 with the lowest index winning ties, like the reference sampler.

**repack_kernel** is described in [packed-layout.md](packed-layout.md).

## Resource checks

`cmake/check_kernel_resources.cmake` (CTest `kernel_resources`) compiles `kernels.cu` with `-Xptxas -v` for every arch in `CMAKE_CUDA_ARCHITECTURES`. It fails if any entry reports a stack frame, spill stores or loads, or local memory.

With archs `80;86;87;89;90` it covers 140 entries (28 kernels × 5 archs). All of them report `0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads`. Register counts on sm_86 run from 8 (`pos_add`) to 100 (`gdn<128>`); the GEMV variants use 54–79.

## Correctness chain

1. `test_packed`: the packed codes equal the reference trits, and the packed matvec matches `bd::matvec_raw`.
2. `test_emu_vs_cpu`: `src/emu_decoder.cpp` is a CPU implementation of exactly the GPU math: same packing, same Q8, same accumulation order where it matters. It's checked against bonsai-decoder's CPU decoder on synthetic qwen35 models. Hidden-state error has to stay within 2× the reference's own fp noise baseline, and greedy tokens have to agree.
3. `test_gpu_kernels`: each CUDA kernel against a host reference.
4. `test_gpu_decoder`: the full GPU decoder against the CPU reference and the emulator, plus graph vs eager and prefill vs per-token.
5. `golden_greeting`: the real 27B model on the GPU must reproduce the greedy tokens saved in `tests/data/greeting_ref.txt` by the CPU reference.

GPU tests exit 77 (CTest "skipped") when no CUDA device is present.
