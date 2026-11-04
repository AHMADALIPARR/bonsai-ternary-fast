// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Host-side launchers of the CUDA kernels (kernels.cu). Plain C++ so the
// decoder and the kernel tests can call them; every pointer is a device
// pointer unless it says host. All launches are asynchronous on `st` and
// legal inside CUDA graph capture (no allocation, no sync, no attribute
// calls; kernels_init() does those once).
#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>

namespace btf {
namespace cuda {

// Call once per device before any launch (sets shared-memory opt-ins).
void kernels_init(int device);

// Largest batch the ternary GEMV can keep in shared memory for width K.
int gemv_max_nb(int K);

// ---- one-time repack: nblocks PTQ1_0 blocks (28 B each, src may be mapped
// host memory) -> codes [nblocks][32 B] + scales [nblocks] (fp16 bits).
void launch_repack(const uint8_t* src, int64_t nblocks, uint8_t* codes, uint16_t* scales, cudaStream_t st);

// Q8_0 activations of M rows of width K.
struct Q8Act {
  int8_t* q = nullptr;  // [M][K]
  float* d = nullptr;   // [M][K/32]
  int* s = nullptr;     // [M][K/32] sum of q per block
};

// Input transform of a folded weight (block 0: none).
struct Hx {
  int block = 0;
  int perm_hd = 0, perm_nk = 0, perm_rep = 0;  // rep > 1: tiled -> grouped
  const float* signs = nullptr;                // [n] or nullptr
};

// ---- fused prep: [RMSNorm * w] -> perm -> signs -> blockwise WHT -> Q8_0.
// x [M][ldx]; normw nullptr = no norm; xn_out (optional) gets the normed rows.
void launch_prep(const float* x, int ldx, int M, int n, const float* normw, float eps, const Hx& hx,
                 float* xn_out, int ldxn, const Q8Act& a, cudaStream_t st);

// ---- ternary GEMV / small-batch GEMM on packed codes.
struct GemvSeg {
  const uint8_t* codes = nullptr;   // [N][K/4]
  const uint16_t* scales = nullptr; // [N][K/128]
  int N = 0;
  float* y = nullptr;  // [M][ldy]
  int ldy = 0;
};
enum GemvMode { kStore = 0, kAdd = 1, kSwiglu = 2 };
// kStore / kAdd: y = W x or y += W x for every segment (all share K and x).
// kSwiglu: segs[0] = gate, segs[1] = up (same N); segs[0].y = silu(g) * u.
void launch_gemv(const GemvSeg* segs, int nseg, int K, int M, const Q8Act& a, GemvMode mode, cudaStream_t st);

// ---- GDN pieces
// beta = sigmoid(Wb xn), g = softplus(Wa xn + dt) * ssm_a; W [NV][E] bf16 or f32.
void launch_ab_gates(const void* wa, const void* wb, bool bf16, const float* xn, int ldx, int M, int E, int NV,
                     const float* dt, const float* ssm_a, float* beta, float* g, cudaStream_t st);
// causal conv (kernel 4) over M rows + silu; state [3][C] is updated.
void launch_conv_silu(const float* x, int ldx, int M, int C, const float* w, float* state, float* out, int ldo,
                      cudaStream_t st);
// gated delta rule + gated RMSNorm for M tokens; state [NV][DV][DK].
void launch_gdn(const float* cv, int ldc, const float* z, int ldz, const float* g, const float* beta, int M,
                int NV, int NK, int DK, int DV, const float* ssm_norm, float eps, float* state, float* out, int ldo,
                cudaStream_t st);

// ---- attention pieces (positions come from *d_pos so graphs can be replayed)
// q/k RMSNorm + RoPE, q rounding, K/V written to the fp16 cache at pos + m.
void launch_attn_prep(const float* qf, int ldq, const float* k, int ldk, const float* v, int ldv, int M, int H,
                      int HK, int D, int R, float rope_base, float eps, const float* q_norm, const float* k_norm,
                      bool round_q, const int* d_pos, float* q_out, uint16_t* kc, uint16_t* vc, cudaStream_t st);
// split-K decode attention: part [M][H][nsplit][D + 2] (o unnormalized, max, sum)
void launch_attention(const float* q, const uint16_t* kc, const uint16_t* vc, const int* d_pos, int M, int H,
                      int HK, int D, int nsplit, int max_chunk, float* part, cudaStream_t st);
// merges the splits, normalizes, multiplies by sigmoid(gate); gate = qf + D per head.
void launch_attn_combine(const float* part, const float* qf, int ldq, int M, int H, int D, int nsplit, float* out,
                         int ldo, cudaStream_t st);

// ---- embedding lookup from the packed table (may live in mapped host memory),
// then the inverse transform (WHT, then signs).
void launch_embed(const uint8_t* codes, const uint16_t* scales, const int* d_tok, int M, int E, const Hx& inv,
                  float* x, cudaStream_t st);

// ---- greedy pick (smallest index on ties) -> *d_tok_out; part needs
// argmax_part_bytes(V) of scratch.
size_t argmax_part_bytes(int V);
void launch_argmax(const float* logits, int V, void* part, int* d_tok_out, cudaStream_t st);

void launch_pos_add(int* d_pos, int M, cudaStream_t st);

}  // namespace cuda
}  // namespace btf
