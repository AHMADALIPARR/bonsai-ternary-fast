// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CUDA kernels of the fast ternary decode path. docs/kernel-design.md walks
// through each one. Rules kept here: every kernel has __launch_bounds__, all
// per-thread arrays are indexed by compile-time constants (no local memory),
// no fast-math, and the float math follows the CPU reference op order where
// it is cheap to do so.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "btf/packed.hpp"
#include "kernels.cuh"

namespace btf {
namespace cuda {

namespace {

int g_optin_smem = 48 * 1024;
int g_sm_count = 1;
int g_smem_per_sm = 64 * 1024;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
#define BTF_LAUNCH_CHECK(name) check(cudaGetLastError(), name)

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ int warp_sum(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over the block (blockDim multiple of 32, <= 1024). red: 32 doubles.
// Result is valid in every thread. Ends with a barrier so red can be reused.
__device__ __forceinline__ double block_sum(double v, double* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  double t = lane < nw ? red[lane] : 0.0;
  t = warp_sum(t);
  __syncthreads();
  return t;
}
__device__ __forceinline__ float block_sum_f(float v, float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : 0.f;
  t = warp_sum(t);
  __syncthreads();
  return t;
}
__device__ __forceinline__ float block_max(float v, float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = lane < nw ? red[lane] : -INFINITY;
  t = warp_max(t);
  __syncthreads();
  return t;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
__device__ __forceinline__ float round_half(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// In-place Walsh-Hadamard butterflies of every `block`-sized piece of buf[0, S)
// in the reference stage order (len = 1, 2, 4, ...). No final 1/sqrt scaling.
__device__ __forceinline__ void smem_fwht(float* buf, int S, int block) {
  for (int len = 1; len < block; len <<= 1) {
    for (int p = threadIdx.x; p < (S >> 1); p += blockDim.x) {
      const int j = (p / len) * (len << 1) + (p % len);
      const float a = buf[j], b = buf[j + len];
      buf[j] = a + b;
      buf[j + len] = a - b;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- repack
__global__ void __launch_bounds__(256) repack_kernel(const uint8_t* __restrict__ src, int64_t nblocks,
                                                     uint4* __restrict__ codes, uint16_t* __restrict__ scales) {
  for (int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; b < nblocks;
       b += (int64_t)gridDim.x * blockDim.x) {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(src + b * kBlockBytes);
    const uint32_t w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3], w4 = p[4], w5 = p[5], w6 = p[6];
    auto get = [&](int i) -> uint32_t {
      const int wi = i >> 2;
      const uint32_t w = wi == 0 ? w0 : wi == 1 ? w1 : wi == 2 ? w2 : wi == 3 ? w3 : wi == 4 ? w4 : wi == 5 ? w5 : w6;
      return (w >> ((i & 3) * 8)) & 0xffu;
    };
    uint32_t o[8];
    uint16_t sc;
    pack_block_get(get, o, &sc);
    codes[2 * b] = make_uint4(o[0], o[1], o[2], o[3]);
    codes[2 * b + 1] = make_uint4(o[4], o[5], o[6], o[7]);
    scales[b] = sc;
  }
}

// ---------------------------------------------------------------- prep
struct PrepArgs {
  const float* x;
  int ldx, n, S;
  const float* normw;
  float eps;
  int block, perm_hd, perm_nk, perm_rep;
  const float* signs;
  float* xn_out;
  int ldxn;
  int8_t* q;
  float* d;
  int* s;
};

constexpr int kPrepThreads = 256;

template <bool NORM>
__global__ void __launch_bounds__(kPrepThreads) prep_kernel(PrepArgs a) {
  extern __shared__ float buf[];
  __shared__ double red[32];
  const int m = blockIdx.y, base = blockIdx.x * a.S;
  const float* xr = a.x + (size_t)m * a.ldx;
  float scale = 1.0f;
  if (NORM) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < a.n; i += blockDim.x) acc += (double)xr[i] * (double)xr[i];
    const double sum = block_sum(acc, red);
    const float mean = (float)(sum / a.n);
    scale = 1.0f / sqrtf(mean + a.eps);
  }
  for (int o = threadIdx.x; o < a.S; o += blockDim.x) {
    const int og = base + o;
    int src = og;
    if (a.perm_rep > 1) {
      const int dd = og % a.perm_hd, t = og / a.perm_hd;
      const int r = t % a.perm_rep, k = t / a.perm_rep;
      src = dd + a.perm_hd * (k + a.perm_nk * r);
    }
    float v = xr[src];
    if (NORM) v = __fmul_rn(__fmul_rn(v, scale), a.normw[src]);
    if (a.signs) v = __fmul_rn(v, a.signs[og]);
    buf[o] = v;
    if (a.xn_out) {
      float u = xr[og];
      if (NORM) u = __fmul_rn(__fmul_rn(u, scale), a.normw[og]);
      a.xn_out[(size_t)m * a.ldxn + og] = u;
    }
  }
  __syncthreads();
  float hs = 1.0f;
  if (a.block > 0) {
    smem_fwht(buf, a.S, a.block);
    hs = 1.0f / sqrtf((float)a.block);
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int g = warp; g < a.S / 32; g += nw) {
    float v = buf[g * 32 + lane];
    if (a.block > 0) v = __fmul_rn(v, hs);
    const float amax = warp_max(fabsf(v));
    const float dq = __half2float(__float2half_rn(amax / 127.0f));
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const int qv = __float2int_rn(__fmul_rn(v, id));
    const size_t col = (size_t)base + g * 32 + lane;
    a.q[(size_t)m * a.n + col] = (int8_t)qv;
    const int sum = warp_sum(qv);
    if (lane == 0) {
      const size_t bi = (size_t)m * (a.n / 32) + (base / 32 + g);
      a.d[bi] = dq;
      a.s[bi] = sum;
    }
  }
}

int span_of(int n) {
  int s = 1;
  while (s < 1024 && n % (s * 2) == 0) s *= 2;
  return s;
}

// ---------------------------------------------------------------- GEMV
struct GemvArgs {
  const uint8_t* c0; const uint8_t* c1; const uint8_t* c2;
  const uint16_t* s0; const uint16_t* s1; const uint16_t* s2;
  float* y0; float* y1; float* y2;
  int n0, n1, n2;
  int ld0, ld1, ld2;
  int u1, u2, utotal;  // cumulative unit counts
  int K, nb, m0;
  const int8_t* q;
  const float* d;
  const int* s;
};

constexpr int kGemvThreads = 256;
constexpr int kGemvWarps = kGemvThreads / 32;

__device__ __forceinline__ int dot16(uint32_t w, uint4 x) {
  int acc = 0;
  acc = __dp4a((int)(w & 0x03030303u), (int)x.x, acc);
  acc = __dp4a((int)((w >> 2) & 0x03030303u), (int)x.y, acc);
  acc = __dp4a((int)((w >> 4) & 0x03030303u), (int)x.z, acc);
  acc = __dp4a((int)((w >> 6) & 0x03030303u), (int)x.w, acc);
  return acc;
}

size_t gemv_smem(int nb, int K) { return (size_t)nb * (K / 32) * (32 + 4 + 4); }

template <int NB, int MODE>
__global__ void __launch_bounds__(kGemvThreads) gemv_kernel(GemvArgs a) {
  extern __shared__ uint4 sm[];
  const int nsub = a.K / 32;
  uint4* p0 = sm;
  uint4* p1 = sm + NB * nsub;
  float* sd = reinterpret_cast<float*>(sm + 2 * NB * nsub);
  int* ss = reinterpret_cast<int*>(sd + NB * nsub);
  for (int idx = threadIdx.x; idx < NB * nsub; idx += blockDim.x) {
    const int b = idx / nsub, s = idx - b * nsub;
    if (b < a.nb) {
      const size_t row = (size_t)(a.m0 + b);
      const uint4* src = reinterpret_cast<const uint4*>(a.q + row * a.K);
      p0[idx] = src[2 * s];
      p1[idx] = src[2 * s + 1];
      sd[idx] = a.d[row * nsub + s];
      ss[idx] = a.s[row * nsub + s];
    } else {
      p0[idx] = make_uint4(0, 0, 0, 0);
      p1[idx] = make_uint4(0, 0, 0, 0);
      sd[idx] = 0.f;
      ss[idx] = 0;
    }
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rowb = a.K / 4;   // code bytes per row
  const int nsc = a.K / 128;  // scales per row
  for (int u = blockIdx.x * kGemvWarps + warp; u < a.utotal; u += gridDim.x * kGemvWarps) {
    const int k = (u >= a.u1) + (u >= a.u2);
    const int lu = u - (k == 0 ? 0 : k == 1 ? a.u1 : a.u2);
    const uint8_t* codes = k == 0 ? a.c0 : k == 1 ? a.c1 : a.c2;
    const uint16_t* scl = k == 0 ? a.s0 : k == 1 ? a.s1 : a.s2;
    float* y = k == 0 ? a.y0 : k == 1 ? a.y1 : a.y2;
    const int N = k == 0 ? a.n0 : k == 1 ? a.n1 : a.n2;
    const int ldy = k == 0 ? a.ld0 : k == 1 ? a.ld1 : a.ld2;
    int r0, r1;
    const uint8_t* cb;
    const uint16_t* sb;
    if (MODE == kSwiglu) {
      r0 = lu;
      r1 = lu;
      cb = a.c1;
      sb = a.s1;
    } else {
      r0 = 2 * lu;
      r1 = 2 * lu + 1 < N ? 2 * lu + 1 : 2 * lu;
      cb = codes;
      sb = scl;
    }
    const uint2* ca = reinterpret_cast<const uint2*>(codes + (size_t)r0 * rowb);
    const uint2* cc = reinterpret_cast<const uint2*>(cb + (size_t)r1 * rowb);
    const uint16_t* sa_ = scl + (size_t)r0 * nsc;
    const uint16_t* sc_ = sb + (size_t)r1 * nsc;
    float acc0[NB], acc1[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc0[b] = acc1[b] = 0.f;
    for (int s = lane; s < nsub; s += 32) {
      const uint2 w0 = __ldg(ca + s);
      const uint2 w1 = __ldg(cc + s);
      const float f0 = h2f(__ldg(sa_ + (s >> 2)));
      const float f1 = h2f(__ldg(sc_ + (s >> 2)));
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        const uint4 x0 = p0[b * nsub + s], x1 = p1[b * nsub + s];
        const float dd = sd[b * nsub + s];
        const int sv = ss[b * nsub + s];
        const int i0 = dot16(w0.x, x0) + dot16(w0.y, x1) - sv;
        const int i1 = dot16(w1.x, x0) + dot16(w1.y, x1) - sv;
        acc0[b] += (f0 * dd) * (float)i0;
        acc1[b] += (f1 * dd) * (float)i1;
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      acc0[b] = warp_sum(acc0[b]);
      acc1[b] = warp_sum(acc1[b]);
    }
    if (lane == 0) {
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        if (b < a.nb) {
          const size_t m = (size_t)(a.m0 + b);
          if (MODE == kSwiglu) {
            y[m * ldy + r0] = silu_f(acc0[b]) * acc1[b];
          } else if (MODE == kAdd) {
            y[m * ldy + r0] += acc0[b];
            if (r1 != r0) y[m * ldy + r1] += acc1[b];
          } else {
            y[m * ldy + r0] = acc0[b];
            if (r1 != r0) y[m * ldy + r1] = acc1[b];
          }
        }
      }
    }
  }
}

template <int NB, int MODE>
void set_gemv_attr() {
  check(cudaFuncSetAttribute(gemv_kernel<NB, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, g_optin_smem),
        "cudaFuncSetAttribute(gemv)");
}
template <int MODE>
void set_gemv_attrs() {
  set_gemv_attr<1, MODE>();
  set_gemv_attr<2, MODE>();
  set_gemv_attr<4, MODE>();
  set_gemv_attr<8, MODE>();
}

template <int MODE>
void gemv_dispatch(int NB, dim3 grid, size_t smem, cudaStream_t st, const GemvArgs& a) {
  switch (NB) {
    case 1: gemv_kernel<1, MODE><<<grid, kGemvThreads, smem, st>>>(a); break;
    case 2: gemv_kernel<2, MODE><<<grid, kGemvThreads, smem, st>>>(a); break;
    case 4: gemv_kernel<4, MODE><<<grid, kGemvThreads, smem, st>>>(a); break;
    default: gemv_kernel<8, MODE><<<grid, kGemvThreads, smem, st>>>(a); break;
  }
}

// ---------------------------------------------------------------- GDN gates
template <bool BF16>
__global__ void __launch_bounds__(256) ab_gates_kernel(const void* __restrict__ wa, const void* __restrict__ wb,
                                                       const float* __restrict__ xn, int ldx, int M, int E, int NV,
                                                       const float* __restrict__ dt, const float* __restrict__ ssm_a,
                                                       float* __restrict__ beta, float* __restrict__ g) {
  const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (w >= NV * M) return;
  const int h = w % NV, m = w / NV;
  const float* xr = xn + (size_t)m * ldx;
  float sa = 0.f, sb = 0.f;
  for (int i = lane; i < E; i += 32) {
    float xv = xr[i];
    float va, vb;
    if (BF16) {
      xv = __bfloat162float(__float2bfloat16_rn(xv));
      va = bd::bf16_to_f32(static_cast<const uint16_t*>(wa)[(size_t)h * E + i]);
      vb = bd::bf16_to_f32(static_cast<const uint16_t*>(wb)[(size_t)h * E + i]);
    } else {
      va = static_cast<const float*>(wa)[(size_t)h * E + i];
      vb = static_cast<const float*>(wb)[(size_t)h * E + i];
    }
    sa += va * xv;
    sb += vb * xv;
  }
  sa = warp_sum(sa);
  sb = warp_sum(sb);
  if (lane == 0) {
    beta[(size_t)m * NV + h] = sigmoid_f(sb);
    g[(size_t)m * NV + h] = softplus_f(sa + dt[h]) * ssm_a[h];
  }
}

// ---------------------------------------------------------------- conv + silu
__global__ void __launch_bounds__(256) conv_silu_kernel(const float* __restrict__ x, int ldx, int M, int C,
                                                        const float* __restrict__ w, float* __restrict__ state,
                                                        float* __restrict__ out, int ldo) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s0 = state[c], s1 = state[C + c], s2 = state[2 * C + c];
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int m = 0; m < M; ++m) {
    const float xin = x[(size_t)m * ldx + c];
    float s = 0.f;
    s = __fadd_rn(s, __fmul_rn(s0, w0));
    s = __fadd_rn(s, __fmul_rn(s1, w1));
    s = __fadd_rn(s, __fmul_rn(s2, w2));
    s = __fadd_rn(s, __fmul_rn(xin, w3));
    s0 = s1;
    s1 = s2;
    s2 = xin;
    out[(size_t)m * ldo + c] = silu_f(s);
  }
  state[c] = s0;
  state[C + c] = s1;
  state[2 * C + c] = s2;
}

// ---------------------------------------------------------------- GDN recurrence
__device__ __forceinline__ int padi(int e) { return e + (e >> 5); }

template <int DK>
__global__ void __launch_bounds__(512) gdn_kernel(const float* __restrict__ cv, int ldc, const float* __restrict__ z,
                                                  int ldz, const float* __restrict__ gg,
                                                  const float* __restrict__ beta, int M, int NV, int NK, int DV,
                                                  const float* __restrict__ ssm_norm, float eps,
                                                  float* __restrict__ state, float* __restrict__ out, int ldo) {
  constexpr int PI = DK / 4;
  __shared__ float sq[DK + DK / 32], sk[DK + DK / 32];
  __shared__ double red[32];
  const int h = blockIdx.x, t = threadIdx.x, j = t >> 2, p = t & 3;
  const int kh = h % NK, QK = NK * DK;
  float* sp = state + ((size_t)h * DV + j) * DK + p * PI;
  float st[PI];
#pragma unroll
  for (int i = 0; i < PI; ++i) st[i] = sp[i];
  const float qscale = 1.0f / sqrtf((float)DK);
  for (int m = 0; m < M; ++m) {
    const float* cr = cv + (size_t)m * ldc;
    double aq = 0.0, ak = 0.0;
    for (int e = t; e < 2 * DK; e += blockDim.x) {
      if (e < DK) {
        const float v = cr[kh * DK + e];
        aq += (double)v * v;
      } else {
        const float v = cr[QK + kh * DK + (e - DK)];
        ak += (double)v * v;
      }
    }
    const double sumq = block_sum(aq, red);
    const double sumk = block_sum(ak, red);
    const float scq = 1.0f / fmaxf(sqrtf((float)sumq), eps);
    const float sck = 1.0f / fmaxf(sqrtf((float)sumk), eps);
    for (int e = t; e < 2 * DK; e += blockDim.x) {
      if (e < DK)
        sq[padi(e)] = cr[kh * DK + e] * scq;
      else
        sk[padi(e - DK)] = cr[QK + kh * DK + (e - DK)] * sck;
    }
    __syncthreads();
    const float v = cr[2 * QK + h * DV + j];
    const float decay = expf(gg[(size_t)m * NV + h]);
    const float bt = beta[(size_t)m * NV + h];
    float kv = 0.f;
#pragma unroll
    for (int i = 0; i < PI; ++i) {
      st[i] *= decay;
      kv += st[i] * sk[padi(p * PI + i)];
    }
    kv += __shfl_xor_sync(0xffffffffu, kv, 1);
    kv += __shfl_xor_sync(0xffffffffu, kv, 2);
    const float dl = (v - kv) * bt;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < PI; ++i) {
      st[i] += sk[padi(p * PI + i)] * dl;
      acc += st[i] * sq[padi(p * PI + i)];
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
    const float o = acc * qscale;
    const double so = block_sum(p == 0 ? (double)o * o : 0.0, red);
    const float sc = 1.0f / sqrtf((float)(so / DV) + eps);
    if (p == 0) {
      const float zz = z[(size_t)m * ldz + h * DV + j];
      out[(size_t)m * ldo + h * DV + j] = __fmul_rn(__fmul_rn(__fmul_rn(o, sc), ssm_norm[j]), silu_f(zz));
    }
    // block_sum ended with a barrier, so sq / sk may be rewritten next token
  }
#pragma unroll
  for (int i = 0; i < PI; ++i) sp[i] = st[i];
}

// ---------------------------------------------------------------- attention prep
__global__ void __launch_bounds__(256) attn_prep_kernel(const float* __restrict__ qf, int ldq,
                                                        const float* __restrict__ k, int ldk,
                                                        const float* __restrict__ v, int ldv, int H, int HK, int D,
                                                        int R, float theta_scale, float eps,
                                                        const float* __restrict__ q_norm,
                                                        const float* __restrict__ k_norm, int round_q,
                                                        const int* __restrict__ d_pos, float* __restrict__ q_out,
                                                        uint16_t* __restrict__ kc, uint16_t* __restrict__ vc) {
  __shared__ float hv[256];
  __shared__ double red[32];
  const int hh = blockIdx.x, m = blockIdx.y, i = threadIdx.x;
  const int pos = *d_pos + m;
  const bool isq = hh < H;
  const float x = isq ? qf[(size_t)m * ldq + (size_t)hh * 2 * D + i] : k[(size_t)m * ldk + (size_t)(hh - H) * D + i];
  const double sum = block_sum((double)x * x, red);
  const float sc = 1.0f / sqrtf((float)(sum / D) + eps);
  hv[i] = __fmul_rn(__fmul_rn(x, sc), isq ? q_norm[i] : k_norm[i]);
  __syncthreads();
  float r0 = 0.f, r1 = 0.f;
  const int half = R / 2;
  if (i < half) {
    float theta = (float)pos;
    for (int c = 0; c < i; ++c) theta *= theta_scale;
    // cosf/sinf have a Payne-Hanek slow path that uses local memory; theta is
    // a float, so theta/pi in double is exact enough and sincospi needs no
    // large-argument reduction (rounded to float it agrees with libm to ~1 ulp).
    double sd, cd;
    sincospi((double)theta * 0.31830988618379067154, &sd, &cd);
    const float cs = (float)cd, sn = (float)sd;
    const float x0 = hv[i], x1 = hv[i + half];
    r0 = __fsub_rn(__fmul_rn(x0, cs), __fmul_rn(x1, sn));
    r1 = __fadd_rn(__fmul_rn(x0, sn), __fmul_rn(x1, cs));
  }
  __syncthreads();
  if (i < half) {
    hv[i] = r0;
    hv[i + half] = r1;
  }
  __syncthreads();
  float val = hv[i];
  if (isq) {
    if (round_q) val = round_half(val);
    q_out[((size_t)m * H + hh) * D + i] = val;
  } else {
    const int g = hh - H;
    const size_t o = ((size_t)pos * HK + g) * D + i;
    kc[o] = __half_as_ushort(__float2half_rn(val));
    vc[o] = __half_as_ushort(__float2half_rn(v[(size_t)m * ldv + (size_t)g * D + i]));
  }
}

// ---------------------------------------------------------------- attention
constexpr int kAttnThreads = 128;

__global__ void __launch_bounds__(kAttnThreads) attention_kernel(const float* __restrict__ q,
                                                                 const uint16_t* __restrict__ kc,
                                                                 const uint16_t* __restrict__ vc,
                                                                 const int* __restrict__ d_pos, int H, int HK, int D,
                                                                 int nsplit, float scale, float* __restrict__ part) {
  extern __shared__ float smem[];
  __shared__ float red[32];
  const int h = blockIdx.x, sp = blockIdx.y, m = blockIdx.z;
  const int T = *d_pos + m + 1;
  const int chunk = (T + nsplit - 1) / nsplit;
  const int t0 = sp * chunk, t1 = min(T, t0 + chunk), n = max(0, t1 - t0);
  const int g = h / (H / HK), kvw = HK * D;
  float* sqv = smem;      // [D]
  float* sc = smem + D;   // [chunk]
  float* po = part + (((size_t)m * H + h) * nsplit + sp) * (D + 2);
  const float* qr = q + ((size_t)m * H + h) * D;
  for (int i = threadIdx.x; i < D; i += blockDim.x) sqv[i] = qr[i];
  __syncthreads();
  if (n == 0) {
    for (int i = threadIdx.x; i < D; i += blockDim.x) po[i] = 0.f;
    if (threadIdx.x == 0) {
      po[D] = -INFINITY;
      po[D + 1] = 0.f;
    }
    return;
  }
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int t = t0 + warp; t < t1; t += nw) {
    const __half2* kr = reinterpret_cast<const __half2*>(kc + (size_t)t * kvw + (size_t)g * D);
    float dot = 0.f;
    for (int i = lane * 2; i < D; i += 64) {
      const float2 kk = __half22float2(kr[i >> 1]);
      dot += sqv[i] * kk.x + sqv[i + 1] * kk.y;
    }
    dot = warp_sum(dot);
    if (lane == 0) sc[t - t0] = dot * scale;
  }
  __syncthreads();
  float mx = -INFINITY;
  for (int t = threadIdx.x; t < n; t += blockDim.x) mx = fmaxf(mx, sc[t]);
  mx = block_max(mx, red);
  float ls = 0.f;
  for (int t = threadIdx.x; t < n; t += blockDim.x) {
    const float e = expf(sc[t] - mx);
    sc[t] = e;
    ls += e;
  }
  const float tot = block_sum_f(ls, red);  // ends with a barrier: sc is ready
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const uint16_t* vr = vc + (size_t)t0 * kvw + (size_t)g * D + i;
    float acc = 0.f;
    for (int t = 0; t < n; ++t) acc += sc[t] * h2f(vr[(size_t)t * kvw]);
    po[i] = acc;
  }
  if (threadIdx.x == 0) {
    po[D] = mx;
    po[D + 1] = tot;
  }
}

__global__ void __launch_bounds__(128) attn_combine_kernel(const float* __restrict__ part,
                                                           const float* __restrict__ qf, int ldq, int H, int D,
                                                           int nsplit, float* __restrict__ out, int ldo) {
  const int h = blockIdx.x, m = blockIdx.y;
  const float* pb = part + ((size_t)m * H + h) * nsplit * (D + 2);
  float gmax = -INFINITY;
  for (int s = 0; s < nsplit; ++s) gmax = fmaxf(gmax, pb[s * (D + 2) + D]);
  float tot = 0.f;
  for (int s = 0; s < nsplit; ++s) {
    const float sm = pb[s * (D + 2) + D + 1];
    if (sm > 0.f) tot += sm * expf(pb[s * (D + 2) + D] - gmax);
  }
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float o = 0.f;
    for (int s = 0; s < nsplit; ++s) {
      const float sm = pb[s * (D + 2) + D + 1];
      if (sm > 0.f) o += pb[s * (D + 2) + i] * expf(pb[s * (D + 2) + D] - gmax);
    }
    o = o / tot;
    const float gate = qf[(size_t)m * ldq + (size_t)h * 2 * D + D + i];
    out[(size_t)m * ldo + (size_t)h * D + i] = o * sigmoid_f(gate);
  }
}

// ---------------------------------------------------------------- embedding
__global__ void __launch_bounds__(256) embed_kernel(const uint8_t* __restrict__ codes,
                                                    const uint16_t* __restrict__ scales,
                                                    const int* __restrict__ d_tok, int E, int S, int block,
                                                    const float* __restrict__ signs, float* __restrict__ x) {
  extern __shared__ float buf[];  // [S] floats, then S/4 code bytes
  uint32_t* cw = reinterpret_cast<uint32_t*>(buf + S);
  const int m = blockIdx.y, base = blockIdx.x * S;
  const size_t tok = (size_t)d_tok[m];
  const uint32_t* rowc = reinterpret_cast<const uint32_t*>(codes + tok * (E / 4) + base / 4);
  for (int i = threadIdx.x; i < S / 16; i += blockDim.x) cw[i] = rowc[i];
  __syncthreads();
  const uint16_t* rows = scales + tok * (E / 128);
  for (int o = threadIdx.x; o < S; o += blockDim.x) {
    const int e = base + o, eb = e & 127;
    const int byte = (e >> 7) * 32 + code_byte(eb) - base / 4;
    const uint32_t c = (cw[byte >> 2] >> ((byte & 3) * 8 + code_shift(eb))) & 3u;
    buf[o] = (float)((int)c - 1) * h2f(__ldg(rows + (e >> 7)));
  }
  __syncthreads();
  float hs = 1.0f;
  if (block > 0) {
    smem_fwht(buf, S, block);
    hs = 1.0f / sqrtf((float)block);
  }
  for (int o = threadIdx.x; o < S; o += blockDim.x) {
    float v = buf[o];
    if (block > 0) v = __fmul_rn(v, hs);
    if (signs) v = __fmul_rn(v, signs[base + o]);
    x[(size_t)m * E + base + o] = v;
  }
}

// ---------------------------------------------------------------- argmax
constexpr int kArgBlocks = 128;

__device__ __forceinline__ void arg_better(float& bv, int& bi, float v, int i) {
  if (v > bv || (v == bv && i < bi)) {
    bv = v;
    bi = i;
  }
}
__device__ __forceinline__ void warp_arg(float& bv, int& bi) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float v = __shfl_xor_sync(0xffffffffu, bv, o);
    const int i = __shfl_xor_sync(0xffffffffu, bi, o);
    arg_better(bv, bi, v, i);
  }
}
__device__ __forceinline__ void block_arg(float& bv, int& bi) {
  __shared__ float rv[32];
  __shared__ int ri[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  warp_arg(bv, bi);
  if (lane == 0) {
    rv[warp] = bv;
    ri[warp] = bi;
  }
  __syncthreads();
  bv = lane < nw ? rv[lane] : -INFINITY;
  bi = lane < nw ? ri[lane] : 0x7fffffff;
  warp_arg(bv, bi);
}

__global__ void __launch_bounds__(256) argmax_stage1(const float* __restrict__ x, int V, float2* __restrict__ part) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) arg_better(bv, bi, x[i], i);
  block_arg(bv, bi);
  if (threadIdx.x == 0) part[blockIdx.x] = make_float2(bv, __int_as_float(bi));
}
__global__ void __launch_bounds__(128) argmax_stage2(const float2* __restrict__ part, int n, int* __restrict__ out) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += blockDim.x) arg_better(bv, bi, part[i].x, __float_as_int(part[i].y));
  block_arg(bv, bi);
  if (threadIdx.x == 0) *out = bi == 0x7fffffff ? 0 : bi;
}

__global__ void __launch_bounds__(32) pos_add_kernel(int* d_pos, int M) {
  if (threadIdx.x == 0) *d_pos += M;
}

}  // namespace

// ================================================================ launchers
void kernels_init(int device) {
  check(cudaDeviceGetAttribute(&g_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "optin smem");
  check(cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, device), "sm count");
  check(cudaDeviceGetAttribute(&g_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device), "smem/sm");
  // leave room for the static shared memory of each kernel
  g_optin_smem -= 1024;
  set_gemv_attrs<kStore>();
  set_gemv_attrs<kAdd>();
  set_gemv_attrs<kSwiglu>();
  check(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, g_optin_smem),
        "cudaFuncSetAttribute(attention)");
}

int gemv_max_nb(int K) {
  for (int nb : {8, 4, 2, 1})
    if (gemv_smem(nb, K) <= (size_t)g_optin_smem) return nb;
  return 0;
}

void launch_repack(const uint8_t* src, int64_t nblocks, uint8_t* codes, uint16_t* scales, cudaStream_t st) {
  if (nblocks <= 0) return;
  const int64_t want = (nblocks + 255) / 256;
  const int grid = (int)std::min<int64_t>(want, (int64_t)g_sm_count * 32);
  repack_kernel<<<grid, 256, 0, st>>>(src, nblocks, reinterpret_cast<uint4*>(codes), scales);
  BTF_LAUNCH_CHECK("repack_kernel");
}

void launch_prep(const float* x, int ldx, int M, int n, const float* normw, float eps, const Hx& hx, float* xn_out,
                 int ldxn, const Q8Act& a, cudaStream_t st) {
  if (M <= 0) return;
  if (n % 32) throw std::runtime_error("prep: width must be a multiple of 32");
  PrepArgs p{};
  p.x = x;
  p.ldx = ldx;
  p.n = n;
  p.S = span_of(n);
  p.normw = normw;
  p.eps = eps;
  p.block = hx.block;
  p.perm_hd = hx.perm_hd;
  p.perm_nk = hx.perm_nk;
  p.perm_rep = hx.perm_rep;
  p.signs = hx.signs;
  p.xn_out = xn_out;
  p.ldxn = ldxn;
  p.q = a.q;
  p.d = a.d;
  p.s = a.s;
  if (hx.block > 0 && (p.S % hx.block)) throw std::runtime_error("prep: hadamard block does not divide the span");
  const dim3 grid(n / p.S, M);
  const size_t smem = (size_t)p.S * sizeof(float);
  if (normw)
    prep_kernel<true><<<grid, kPrepThreads, smem, st>>>(p);
  else
    prep_kernel<false><<<grid, kPrepThreads, smem, st>>>(p);
  BTF_LAUNCH_CHECK("prep_kernel");
}

void launch_gemv(const GemvSeg* segs, int nseg, int K, int M, const Q8Act& act, GemvMode mode, cudaStream_t st) {
  if (M <= 0) return;
  if (nseg < 1 || nseg > 3) throw std::runtime_error("gemv: 1..3 segments");
  if (mode == kSwiglu && (nseg != 2 || segs[0].N != segs[1].N)) throw std::runtime_error("gemv: swiglu needs gate+up");
  if (K % 128) throw std::runtime_error("gemv: K must be a multiple of 128");
  const int maxnb = gemv_max_nb(K);
  if (maxnb == 0) throw std::runtime_error("gemv: K too large for shared memory");
  GemvArgs a{};
  const GemvSeg z{};
  const GemvSeg& s0 = segs[0];
  const GemvSeg& s1 = nseg > 1 ? segs[1] : z;
  const GemvSeg& s2 = nseg > 2 ? segs[2] : z;
  a.c0 = s0.codes; a.c1 = s1.codes; a.c2 = s2.codes;
  a.s0 = s0.scales; a.s1 = s1.scales; a.s2 = s2.scales;
  a.y0 = s0.y; a.y1 = s1.y; a.y2 = s2.y;
  a.n0 = s0.N; a.n1 = s1.N; a.n2 = s2.N;
  a.ld0 = s0.ldy; a.ld1 = s1.ldy; a.ld2 = s2.ldy;
  if (mode == kSwiglu) {
    a.u1 = a.u2 = a.utotal = s0.N;
  } else {
    const int n0 = (s0.N + 1) / 2, n1 = (s1.N + 1) / 2, n2 = (s2.N + 1) / 2;
    a.u1 = n0;
    a.u2 = n0 + n1;
    a.utotal = n0 + n1 + n2;
  }
  a.K = K;
  a.q = act.q;
  a.d = act.d;
  a.s = act.s;
  for (int m0 = 0; m0 < M; m0 += maxnb) {
    const int nb = std::min(maxnb, M - m0);
    const int NB = nb <= 1 ? 1 : nb <= 2 ? 2 : nb <= 4 ? 4 : 8;
    a.nb = nb;
    a.m0 = m0;
    const size_t smem = gemv_smem(NB, K);
    const int per_sm = std::max(1, std::min(8, (int)(g_smem_per_sm / std::max<size_t>(smem + 1024, 1))));
    const int want = (a.utotal + kGemvWarps - 1) / kGemvWarps;
    const dim3 grid(std::max(1, std::min(want, g_sm_count * per_sm)));
    if (mode == kStore)
      gemv_dispatch<kStore>(NB, grid, smem, st, a);
    else if (mode == kAdd)
      gemv_dispatch<kAdd>(NB, grid, smem, st, a);
    else
      gemv_dispatch<kSwiglu>(NB, grid, smem, st, a);
    BTF_LAUNCH_CHECK("gemv_kernel");
  }
}

void launch_ab_gates(const void* wa, const void* wb, bool bf16, const float* xn, int ldx, int M, int E, int NV,
                     const float* dt, const float* ssm_a, float* beta, float* g, cudaStream_t st) {
  if (M <= 0) return;
  const int warps = NV * M;
  const dim3 grid((warps + 7) / 8);
  if (bf16)
    ab_gates_kernel<true><<<grid, 256, 0, st>>>(wa, wb, xn, ldx, M, E, NV, dt, ssm_a, beta, g);
  else
    ab_gates_kernel<false><<<grid, 256, 0, st>>>(wa, wb, xn, ldx, M, E, NV, dt, ssm_a, beta, g);
  BTF_LAUNCH_CHECK("ab_gates_kernel");
}

void launch_conv_silu(const float* x, int ldx, int M, int C, const float* w, float* state, float* out, int ldo,
                      cudaStream_t st) {
  if (M <= 0) return;
  conv_silu_kernel<<<(C + 255) / 256, 256, 0, st>>>(x, ldx, M, C, w, state, out, ldo);
  BTF_LAUNCH_CHECK("conv_silu_kernel");
}

void launch_gdn(const float* cv, int ldc, const float* z, int ldz, const float* g, const float* beta, int M, int NV,
                int NK, int DK, int DV, const float* ssm_norm, float eps, float* state, float* out, int ldo,
                cudaStream_t st) {
  if (M <= 0) return;
  if (DV % 8 || DV > 128) throw std::runtime_error("gdn: DV must be a multiple of 8 and <= 128");
  const int threads = 4 * DV;
  switch (DK) {
    case 32: gdn_kernel<32><<<NV, threads, 0, st>>>(cv, ldc, z, ldz, g, beta, M, NV, NK, DV, ssm_norm, eps, state, out, ldo); break;
    case 64: gdn_kernel<64><<<NV, threads, 0, st>>>(cv, ldc, z, ldz, g, beta, M, NV, NK, DV, ssm_norm, eps, state, out, ldo); break;
    case 128: gdn_kernel<128><<<NV, threads, 0, st>>>(cv, ldc, z, ldz, g, beta, M, NV, NK, DV, ssm_norm, eps, state, out, ldo); break;
    default: throw std::runtime_error("gdn: DK must be 32, 64 or 128");
  }
  BTF_LAUNCH_CHECK("gdn_kernel");
}

void launch_attn_prep(const float* qf, int ldq, const float* k, int ldk, const float* v, int ldv, int M, int H, int HK,
                      int D, int R, float rope_base, float eps, const float* q_norm, const float* k_norm, bool round_q,
                      const int* d_pos, float* q_out, uint16_t* kc, uint16_t* vc, cudaStream_t st) {
  if (M <= 0) return;
  if (D % 32 || D > 256 || R > D || R % 2) throw std::runtime_error("attn_prep: unsupported head geometry");
  const float theta_scale = (float)std::pow((double)rope_base, -2.0 / R);
  attn_prep_kernel<<<dim3(H + HK, M), D, 0, st>>>(qf, ldq, k, ldk, v, ldv, H, HK, D, R, theta_scale, eps, q_norm,
                                                   k_norm, round_q ? 1 : 0, d_pos, q_out, kc, vc);
  BTF_LAUNCH_CHECK("attn_prep_kernel");
}

void launch_attention(const float* q, const uint16_t* kc, const uint16_t* vc, const int* d_pos, int M, int H, int HK,
                      int D, int nsplit, int max_chunk, float* part, cudaStream_t st) {
  if (M <= 0) return;
  const size_t smem = (size_t)(D + max_chunk) * sizeof(float);
  if (smem > (size_t)g_optin_smem) throw std::runtime_error("attention: chunk too large for shared memory");
  const float scale = 1.0f / sqrtf((float)D);
  attention_kernel<<<dim3(H, nsplit, M), kAttnThreads, smem, st>>>(q, kc, vc, d_pos, H, HK, D, nsplit, scale, part);
  BTF_LAUNCH_CHECK("attention_kernel");
}

void launch_attn_combine(const float* part, const float* qf, int ldq, int M, int H, int D, int nsplit, float* out,
                         int ldo, cudaStream_t st) {
  if (M <= 0) return;
  attn_combine_kernel<<<dim3(H, M), 128, 0, st>>>(part, qf, ldq, H, D, nsplit, out, ldo);
  BTF_LAUNCH_CHECK("attn_combine_kernel");
}

void launch_embed(const uint8_t* codes, const uint16_t* scales, const int* d_tok, int M, int E, const Hx& inv,
                  float* x, cudaStream_t st) {
  if (M <= 0) return;
  const int S = span_of(E);
  if (S < 128) throw std::runtime_error("embed: width must be a multiple of 128");
  if (inv.block > 0 && S % inv.block) throw std::runtime_error("embed: hadamard block does not divide the span");
  const size_t smem = (size_t)S * sizeof(float) + S / 4;
  embed_kernel<<<dim3(E / S, M), 256, smem, st>>>(codes, scales, d_tok, E, S, inv.block, inv.signs, x);
  BTF_LAUNCH_CHECK("embed_kernel");
}

size_t argmax_part_bytes(int) { return kArgBlocks * sizeof(float2); }

void launch_argmax(const float* logits, int V, void* part, int* d_tok_out, cudaStream_t st) {
  argmax_stage1<<<kArgBlocks, 256, 0, st>>>(logits, V, static_cast<float2*>(part));
  BTF_LAUNCH_CHECK("argmax_stage1");
  argmax_stage2<<<1, 128, 0, st>>>(static_cast<const float2*>(part), kArgBlocks, d_tok_out);
  BTF_LAUNCH_CHECK("argmax_stage2");
}

void launch_pos_add(int* d_pos, int M, cudaStream_t st) {
  pos_add_kernel<<<1, 32, 0, st>>>(d_pos, M);
  BTF_LAUNCH_CHECK("pos_add_kernel");
}

}  // namespace cuda
}  // namespace btf
