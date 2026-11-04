// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Every CUDA kernel against a host reference written from the CPU decoder's
// formulas. Exits 77 (skipped) when no CUDA device is present.
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "../src/cuda/kernels.cuh"
#include "bd/cpu_ops.hpp"
#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "btf/packed.hpp"
#include "test_common.hpp"

namespace {

std::mt19937 rng(1234);
float rn() { return std::normal_distribution<float>(0.f, 1.f)(rng); }
float ru(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

void ck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <class T>
T* dev(const std::vector<T>& h) {
  T* p = nullptr;
  ck(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(1, h.size()) * sizeof(T)));
  if (!h.empty()) ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}
template <class T>
T* dev0(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(1, n) * sizeof(T)));
  ck(cudaMemset(p, 0, std::max<size_t>(1, n) * sizeof(T)));
  return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
  std::vector<T> h(n);
  ck(cudaDeviceSynchronize());
  ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
std::vector<float> randv(size_t n, float s = 1.f) {
  std::vector<float> v(n);
  for (auto& x : v) x = rn() * s;
  return v;
}
float r16(float x) { return bd::fp16_to_f32(bd::f32_to_fp16(x)); }

int fails = 0;
void report(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  std::printf("  %-44s %s  err %.3e (tol %.0e)\n", what, ok ? "ok  " : "FAIL", err, tol);
  if (!ok) ++fails;
}

std::vector<uint8_t> random_ptq(int N, int K) {
  std::vector<float> w((size_t)N * K);
  for (auto& v : w) v = rn() * (ru(0, 1) < 0.3f ? 0.f : 1.f);
  std::vector<uint8_t> src((size_t)N * (K / 128) * 28);
  for (int r = 0; r < N; ++r) bd::quant_row(bd::T_PTQ1_0, &w[(size_t)r * K], &src[(size_t)r * (K / 128) * 28], K);
  return src;
}

// ---- repack (source in mapped host memory, as at load time)
void test_repack() {
  const int N = 41, K = 1024;
  const std::vector<uint8_t> src = random_ptq(N, K);
  uint8_t* hsrc = nullptr;
  ck(cudaHostAlloc(reinterpret_cast<void**>(&hsrc), src.size(), cudaHostAllocMapped));
  std::memcpy(hsrc, src.data(), src.size());
  uint8_t* dsrc = nullptr;
  ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dsrc), hsrc, 0));
  std::vector<uint8_t> hc(btf::packed_codes_bytes(N, K));
  std::vector<uint16_t> hs(btf::packed_scales_bytes(N, K) / 2);
  btf::repack_rows(src.data(), N, K, hc.data(), hs.data());
  uint8_t* dc = dev0<uint8_t>(hc.size());
  uint16_t* ds = dev0<uint16_t>(hs.size());
  btf::cuda::launch_repack(dsrc, (int64_t)N * (K / 128), dc, ds, 0);
  const auto gc = host(dc, hc.size());
  const auto gs = host(ds, hs.size());
  size_t bad = 0;
  for (size_t i = 0; i < hc.size(); ++i) bad += gc[i] != hc[i];
  for (size_t i = 0; i < hs.size(); ++i) bad += gs[i] != hs[i];
  report("repack (mapped source) code/scale mismatches", (double)bad, 0);
  cudaFree(dc);
  cudaFree(ds);
  cudaFreeHost(hsrc);
}

// ---- fused prep
void test_prep(int n, int block, bool norm, bool perm, bool signs, int M) {
  const float eps = 1e-6f;
  std::vector<float> x = randv((size_t)M * n, 3.f), w = randv(n), sg(n);
  for (auto& s : sg) s = ru(0, 1) < 0.5f ? -1.f : 1.f;
  bd::HadamardXform hx;
  hx.active = block > 0;
  hx.block = block;
  hx.signs = signs ? &sg : nullptr;
  if (perm) {
    hx.perm_hd = 32;
    hx.perm_rep = 2;
    hx.perm_nk = n / 64;
  }
  std::vector<int8_t> rq((size_t)M * n);
  std::vector<float> rd((size_t)M * n / 32), rxn((size_t)M * n);
  std::vector<int> rs((size_t)M * n / 32);
  std::vector<float> t(n), xn(n);
  for (int m = 0; m < M; ++m) {
    const float* xr = &x[(size_t)m * n];
    if (norm)
      bd::rmsnorm(xr, w.data(), n, eps, xn.data());
    else
      std::memcpy(xn.data(), xr, n * 4);
    std::memcpy(&rxn[(size_t)m * n], xn.data(), n * 4);
    bd::hadamard_input(hx, xn.data(), n, t.data());
    for (int b = 0; b < n / 32; ++b)
      rs[(size_t)m * n / 32 + b] = btf::q8_block_host(&t[b * 32], &rq[(size_t)m * n + b * 32], &rd[(size_t)m * n / 32 + b]);
  }
  float* dx = dev(x);
  float* dw = dev(w);
  float* dsg = dev(sg);
  float* dxn = dev0<float>((size_t)M * n);
  btf::cuda::Q8Act a;
  a.q = dev0<int8_t>((size_t)M * n);
  a.d = dev0<float>((size_t)M * n / 32);
  a.s = dev0<int>((size_t)M * n / 32);
  btf::cuda::Hx h;
  if (hx.active) {
    h.block = block;
    h.signs = signs ? dsg : nullptr;
    h.perm_hd = hx.perm_hd;
    h.perm_nk = hx.perm_nk;
    h.perm_rep = hx.perm_rep;
  }
  btf::cuda::launch_prep(dx, n, M, n, norm ? dw : nullptr, eps, h, dxn, n, a, 0);
  const auto gq = host(a.q, rq.size());
  const auto gd = host(a.d, rd.size());
  const auto gxn = host(dxn, rxn.size());
  size_t qbad = 0;
  int qmax = 0;
  for (size_t i = 0; i < rq.size(); ++i) {
    qbad += gq[i] != rq[i];
    qmax = std::max(qmax, std::abs((int)gq[i] - (int)rq[i]));
  }
  char lbl[128];
  std::snprintf(lbl, sizeof lbl, "prep n=%d hb=%d norm=%d perm=%d signs=%d M=%d q", n, block, norm, perm, signs, M);
  // a q value within float rounding of a .5 boundary may round the other way
  report(lbl, qmax > 1 ? 1e9 : (double)qbad / rq.size(), 1e-3);
  report("  ... d (rel rms)", btft::rel_rms(gd.data(), rd.data(), rd.size()), 1e-4);
  report("  ... normed rows (rel rms)", btft::rel_rms(gxn.data(), rxn.data(), rxn.size()), 1e-6);
  cudaFree(dx); cudaFree(dw); cudaFree(dsg); cudaFree(dxn); cudaFree(a.q); cudaFree(a.d); cudaFree(a.s);
}

// ---- ternary GEMV / GEMM
struct HostMat {
  int N, K;
  std::vector<uint8_t> codes;
  std::vector<uint16_t> scales;
};
HostMat random_mat(int N, int K) {
  HostMat h{N, K, {}, {}};
  const std::vector<uint8_t> src = random_ptq(N, K);
  h.codes.resize(btf::packed_codes_bytes(N, K));
  h.scales.resize(btf::packed_scales_bytes(N, K) / 2);
  btf::repack_rows(src.data(), N, K, h.codes.data(), h.scales.data());
  return h;
}
std::vector<double> ref_gemv(const HostMat& W, const std::vector<int8_t>& q, const std::vector<float>& d, int M) {
  std::vector<double> y((size_t)M * W.N);
  for (int m = 0; m < M; ++m)
    for (int r = 0; r < W.N; ++r) {
      double acc = 0;
      for (int b = 0; b < W.K / 32; ++b) {
        long dot = 0;
        for (int i = 0; i < 32; ++i) {
          const int e = b * 32 + i;
          dot += (long)(btf::packed_code(&W.codes[(size_t)r * (W.K / 4)], e) - 1) * q[(size_t)m * W.K + e];
        }
        acc += (double)bd::fp16_to_f32(W.scales[(size_t)r * (W.K / 128) + b / 4]) * d[(size_t)m * W.K / 32 + b] * dot;
      }
      y[(size_t)m * W.N + r] = acc;
    }
  return y;
}
double rel(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (a[i] - b[i]) * (a[i] - b[i]);
    den += b[i] * b[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

void test_gemv(int K, int M) {
  std::vector<int8_t> q((size_t)M * K);
  std::vector<float> d((size_t)M * K / 32);
  std::vector<int> s((size_t)M * K / 32);
  for (auto& v : q) v = (int8_t)std::uniform_int_distribution<int>(-127, 127)(rng);
  for (auto& v : d) v = ru(0.001f, 0.05f);
  for (size_t b = 0; b < s.size(); ++b) {
    int t = 0;
    for (int i = 0; i < 32; ++i) t += q[b * 32 + i];
    s[b] = t;
  }
  btf::cuda::Q8Act a;
  a.q = dev(q);
  a.d = dev(d);
  a.s = dev(s);
  const int Ns[3] = {37, 64, 9};
  HostMat W[3] = {random_mat(Ns[0], K), random_mat(Ns[1], K), random_mat(Ns[2], K)};
  uint8_t* dc[3];
  uint16_t* ds[3];
  float* dy[3];
  std::vector<double> ry[3];
  for (int i = 0; i < 3; ++i) {
    dc[i] = dev(W[i].codes);
    ds[i] = dev(W[i].scales);
    dy[i] = dev0<float>((size_t)M * (Ns[i] + 3));
    ry[i] = ref_gemv(W[i], q, d, M);
  }
  // store into three outputs with a row stride wider than N
  btf::cuda::GemvSeg seg[3];
  for (int i = 0; i < 3; ++i) seg[i] = {dc[i], ds[i], Ns[i], dy[i], Ns[i] + 3};
  btf::cuda::launch_gemv(seg, 3, K, M, a, btf::cuda::kStore, 0);
  char lbl[128];
  for (int i = 0; i < 3; ++i) {
    const auto g = host(dy[i], (size_t)M * (Ns[i] + 3));
    std::vector<float> gg((size_t)M * Ns[i]);
    for (int m = 0; m < M; ++m)
      for (int r = 0; r < Ns[i]; ++r) gg[(size_t)m * Ns[i] + r] = g[(size_t)m * (Ns[i] + 3) + r];
    std::snprintf(lbl, sizeof lbl, "gemv store K=%d M=%d seg %d (N=%d)", K, M, i, Ns[i]);
    report(lbl, rel(gg, ry[i]), 1e-5);
  }
  // add: y = y0 + W x
  {
    std::vector<float> y0 = randv((size_t)M * Ns[0]);
    float* d0 = dev(y0);
    btf::cuda::GemvSeg sa{dc[0], ds[0], Ns[0], d0, Ns[0]};
    btf::cuda::launch_gemv(&sa, 1, K, M, a, btf::cuda::kAdd, 0);
    const auto g = host(d0, y0.size());
    std::vector<double> r(y0.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = y0[i] + ry[0][i];
    std::snprintf(lbl, sizeof lbl, "gemv add K=%d M=%d", K, M);
    report(lbl, rel(g, r), 1e-5);
    cudaFree(d0);
  }
  // swiglu: gate = W[1], up = W[1] shifted copy -> use W[1] and a fresh N=64 matrix
  {
    HostMat U = random_mat(Ns[1], K);
    uint8_t* uc = dev(U.codes);
    uint16_t* us = dev(U.scales);
    const std::vector<double> ru_ = ref_gemv(U, q, d, M);
    float* dg = dev0<float>((size_t)M * Ns[1]);
    btf::cuda::GemvSeg sw[2] = {{dc[1], ds[1], Ns[1], dg, Ns[1]}, {uc, us, Ns[1], nullptr, 0}};
    btf::cuda::launch_gemv(sw, 2, K, M, a, btf::cuda::kSwiglu, 0);
    const auto g = host(dg, (size_t)M * Ns[1]);
    std::vector<double> r(g.size());
    for (size_t i = 0; i < r.size(); ++i) r[i] = ry[1][i] / (1 + std::exp(-ry[1][i])) * ru_[i];
    std::snprintf(lbl, sizeof lbl, "gemv swiglu K=%d M=%d", K, M);
    report(lbl, rel(g, r), 1e-5);
    cudaFree(uc); cudaFree(us); cudaFree(dg);
  }
  for (int i = 0; i < 3; ++i) {
    cudaFree(dc[i]); cudaFree(ds[i]); cudaFree(dy[i]);
  }
  cudaFree(a.q); cudaFree(a.d); cudaFree(a.s);
}

// ---- GDN gates, conv, recurrence
void test_ab(bool bf16) {
  const int E = 256, NV = 8, M = 3;
  std::vector<float> wa = randv((size_t)NV * E, 0.05f), wb = randv((size_t)NV * E, 0.05f), xn = randv((size_t)M * E);
  std::vector<float> dt = randv(NV), sa = randv(NV);
  for (auto& v : sa) v = -std::fabs(v);
  std::vector<uint16_t> wa16(wa.size()), wb16(wb.size());
  if (bf16) {
    for (size_t i = 0; i < wa.size(); ++i) {
      wa16[i] = bd::f32_to_bf16(wa[i]);
      wb16[i] = bd::f32_to_bf16(wb[i]);
      wa[i] = bd::bf16_to_f32(wa16[i]);
      wb[i] = bd::bf16_to_f32(wb16[i]);
    }
  }
  std::vector<float> rb((size_t)M * NV), rg((size_t)M * NV);
  for (int m = 0; m < M; ++m) {
    std::vector<float> xr(&xn[(size_t)m * E], &xn[(size_t)(m + 1) * E]);
    if (bf16)
      for (auto& v : xr) v = bd::bf16_to_f32(bd::f32_to_bf16(v));
    for (int h = 0; h < NV; ++h) {
      float a = 0, b = 0;
      for (int i = 0; i < E; ++i) {
        a += wa[(size_t)h * E + i] * xr[i];
        b += wb[(size_t)h * E + i] * xr[i];
      }
      rb[(size_t)m * NV + h] = bd::sigmoid(b);
      rg[(size_t)m * NV + h] = bd::softplus(a + dt[h]) * sa[h];
    }
  }
  void* dwa = bf16 ? (void*)dev(wa16) : (void*)dev(wa);
  void* dwb = bf16 ? (void*)dev(wb16) : (void*)dev(wb);
  float *dx = dev(xn), *ddt = dev(dt), *dsa = dev(sa), *db = dev0<float>(rb.size()), *dg = dev0<float>(rg.size());
  btf::cuda::launch_ab_gates(dwa, dwb, bf16, dx, E, M, E, NV, ddt, dsa, db, dg, 0);
  const auto gb = host(db, rb.size()), gg = host(dg, rg.size());
  report(bf16 ? "ab_gates bf16 beta" : "ab_gates f32 beta", btft::rel_rms(gb.data(), rb.data(), rb.size()), 1e-5);
  report(bf16 ? "ab_gates bf16 g" : "ab_gates f32 g", btft::rel_rms(gg.data(), rg.data(), rg.size()), 1e-5);
  cudaFree(dwa); cudaFree(dwb); cudaFree(dx); cudaFree(ddt); cudaFree(dsa); cudaFree(db); cudaFree(dg);
}

void test_conv() {
  const int C = 300, M = 3, K = 4, ld = C + 5;
  std::vector<float> w = randv((size_t)C * K), st((size_t)(K - 1) * C, 0.f), x = randv((size_t)2 * M * ld);
  std::vector<float> rout((size_t)2 * M * C);
  for (int m = 0; m < 2 * M; ++m)
    for (int c = 0; c < C; ++c) {
      float s = 0.f;
      for (int i = 0; i < K - 1; ++i) s += st[(size_t)i * C + c] * w[(size_t)c * K + i];
      const float xin = x[(size_t)m * ld + c];
      s += xin * w[(size_t)c * K + K - 1];
      for (int i = 0; i + 1 < K - 1; ++i) st[(size_t)i * C + c] = st[(size_t)(i + 1) * C + c];
      st[(size_t)(K - 2) * C + c] = xin;
      rout[(size_t)m * C + c] = bd::silu(s);
    }
  float *dw = dev(w), *dst = dev0<float>(st.size()), *dx = dev(x), *dout = dev0<float>((size_t)2 * M * C);
  // two chunks of M rows: the state must carry over
  btf::cuda::launch_conv_silu(dx, ld, M, C, dw, dst, dout, C, 0);
  btf::cuda::launch_conv_silu(dx + (size_t)M * ld, ld, M, C, dw, dst, dout + (size_t)M * C, C, 0);
  const auto g = host(dout, rout.size()), gs = host(dst, st.size());
  report("conv_silu (2 chunks) out", btft::rel_rms(g.data(), rout.data(), rout.size()), 1e-6);
  report("conv_silu state", btft::rel_rms(gs.data(), st.data(), st.size()), 0);
  cudaFree(dw); cudaFree(dst); cudaFree(dx); cudaFree(dout);
}

void test_gdn(int DK, int DV, int NK, int NV, int M) {
  const int QK = NK * DK, C = 2 * QK + NV * DV, DI = NV * DV;
  const float eps = 1e-6f;
  std::vector<float> cv = randv((size_t)M * C), z = randv((size_t)M * DI), g((size_t)M * NV), beta((size_t)M * NV);
  for (auto& v : g) v = -ru(0.01f, 1.f);
  for (auto& v : beta) v = ru(0.f, 1.f);
  std::vector<float> nw = randv(DV), st = randv((size_t)NV * DK * DV, 0.1f);
  float *dcv = dev(cv), *dz = dev(z), *dg = dev(g), *db = dev(beta), *dnw = dev(nw), *dst = dev(st);
  float* dout = dev0<float>((size_t)M * DI);
  std::vector<float> o((size_t)M * DI), q(DK), k(DK);
  const float qscale = 1.0f / std::sqrt((float)DK);
  for (int h = 0; h < NV; ++h) {
    const int kh = h % NK;
    float* S = &st[(size_t)h * DK * DV];
    for (int m = 0; m < M; ++m) {
      const float* cr = &cv[(size_t)m * C];
      bd::l2norm(cr + (size_t)kh * DK, DK, eps, q.data());
      bd::l2norm(cr + QK + (size_t)kh * DK, DK, eps, k.data());
      const float* v = cr + 2 * QK + (size_t)h * DV;
      const float decay = std::exp(g[(size_t)m * NV + h]), bt = beta[(size_t)m * NV + h];
      float* oh = &o[(size_t)m * DI + (size_t)h * DV];
      for (int j = 0; j < DV; ++j) {
        float* col = S + (size_t)j * DK;
        float kv = 0.f;
        for (int i = 0; i < DK; ++i) {
          col[i] *= decay;
          kv += col[i] * k[i];
        }
        const float dl = (v[j] - kv) * bt;
        float acc = 0.f;
        for (int i = 0; i < DK; ++i) {
          col[i] += k[i] * dl;
          acc += col[i] * q[i];
        }
        oh[j] = acc * qscale;
      }
      bd::rmsnorm(oh, nw.data(), DV, eps, oh);
      for (int j = 0; j < DV; ++j) oh[j] *= bd::silu(z[(size_t)m * DI + (size_t)h * DV + j]);
    }
  }
  btf::cuda::launch_gdn(dcv, C, dz, DI, dg, db, M, NV, NK, DK, DV, dnw, eps, dst, dout, DI, 0);
  const auto go = host(dout, o.size()), gs = host(dst, st.size());
  char lbl[128];
  std::snprintf(lbl, sizeof lbl, "gdn DK=%d DV=%d NK=%d NV=%d M=%d out", DK, DV, NK, NV, M);
  report(lbl, btft::rel_rms(go.data(), o.data(), o.size()), 1e-4);
  report("  ... state", btft::rel_rms(gs.data(), st.data(), st.size()), 1e-4);
  cudaFree(dcv); cudaFree(dz); cudaFree(dg); cudaFree(db); cudaFree(dnw); cudaFree(dst); cudaFree(dout);
}

// ---- attention: prep (norm, RoPE, cache write) + split decode + combine
void test_attention(int D, int R, int H, int HK, int pos0, int M, int nsplit) {
  const int n_ctx = 64, kvw = HK * D, ldq = 2 * H * D + 7;
  const float eps = 1e-6f, base = 1e7f;
  std::vector<float> qf = randv((size_t)M * ldq), k = randv((size_t)M * kvw), v = randv((size_t)M * kvw);
  std::vector<float> qn = randv(D), kn = randv(D);
  std::vector<uint16_t> kc((size_t)n_ctx * kvw), vc((size_t)n_ctx * kvw);
  for (int t = 0; t < pos0; ++t)
    for (int i = 0; i < kvw; ++i) {
      kc[(size_t)t * kvw + i] = bd::f32_to_fp16(rn());
      vc[(size_t)t * kvw + i] = bd::f32_to_fp16(rn());
    }
  // reference
  const float ts = std::pow(base, -2.0f / R);
  auto rope = [&](float* hv, int p) {
    float theta = (float)p;
    for (int ic = 0; ic < R / 2; ++ic) {
      const float c = std::cos(theta), s = std::sin(theta), x0 = hv[ic], x1 = hv[ic + R / 2];
      hv[ic] = x0 * c - x1 * s;
      hv[ic + R / 2] = x0 * s + x1 * c;
      theta *= ts;
    }
  };
  std::vector<uint16_t> rkc = kc, rvc = vc;
  std::vector<float> q((size_t)M * H * D), o((size_t)M * H * D);
  for (int m = 0; m < M; ++m) {
    const int p = pos0 + m;
    for (int h = 0; h < H; ++h) {
      float* qh = &q[((size_t)m * H + h) * D];
      bd::rmsnorm(&qf[(size_t)m * ldq + (size_t)h * 2 * D], qn.data(), D, eps, qh);
      rope(qh, p);
      for (int i = 0; i < D; ++i) qh[i] = r16(qh[i]);
    }
    for (int h = 0; h < HK; ++h) {
      std::vector<float> kh(D);
      bd::rmsnorm(&k[(size_t)m * kvw + (size_t)h * D], kn.data(), D, eps, kh.data());
      rope(kh.data(), p);
      for (int i = 0; i < D; ++i) {
        rkc[(size_t)p * kvw + h * D + i] = bd::f32_to_fp16(kh[i]);
        rvc[(size_t)p * kvw + h * D + i] = bd::f32_to_fp16(v[(size_t)m * kvw + h * D + i]);
      }
    }
  }
  const float scale = 1.0f / std::sqrt((float)D);
  for (int m = 0; m < M; ++m) {
    const int T = pos0 + m + 1;
    for (int h = 0; h < H; ++h) {
      const int g = h / (H / HK);
      std::vector<double> s(T);
      double mx = -1e300;
      for (int t = 0; t < T; ++t) {
        double dd = 0;
        for (int i = 0; i < D; ++i) dd += (double)q[((size_t)m * H + h) * D + i] * bd::fp16_to_f32(rkc[(size_t)t * kvw + g * D + i]);
        s[t] = dd * scale;
        mx = std::max(mx, s[t]);
      }
      double sum = 0;
      for (auto& x : s) sum += (x = std::exp(x - mx));
      for (int i = 0; i < D; ++i) {
        double acc = 0;
        for (int t = 0; t < T; ++t) acc += s[t] * bd::fp16_to_f32(rvc[(size_t)t * kvw + g * D + i]);
        o[((size_t)m * H + h) * D + i] = (float)(acc / sum) * bd::sigmoid(qf[(size_t)m * ldq + (size_t)h * 2 * D + D + i]);
      }
    }
  }
  // GPU
  float *dqf = dev(qf), *dk = dev(k), *dv = dev(v), *dqn = dev(qn), *dkn = dev(kn);
  uint16_t *dkc = dev(kc), *dvc = dev(vc);
  int* dpos = dev(std::vector<int>{pos0});
  float* dq = dev0<float>(q.size());
  float* dpart = dev0<float>((size_t)M * H * nsplit * (D + 2));
  float* dout = dev0<float>(o.size());
  btf::cuda::launch_attn_prep(dqf, ldq, dk, kvw, dv, kvw, M, H, HK, D, R, base, eps, dqn, dkn, true, dpos, dq, dkc, dvc, 0);
  btf::cuda::launch_attention(dq, dkc, dvc, dpos, M, H, HK, D, nsplit, (n_ctx + nsplit - 1) / nsplit, dpart, 0);
  btf::cuda::launch_attn_combine(dpart, dqf, ldq, M, H, D, nsplit, dout, H * D, 0);
  const auto gq = host(dq, q.size()), go = host(dout, o.size());
  const auto gk = host(dkc, kc.size()), gv = host(dvc, vc.size());
  std::vector<float> fk(kc.size()), frk(kc.size());
  size_t vbad = 0;
  for (size_t i = 0; i < kc.size(); ++i) {
    fk[i] = bd::fp16_to_f32(gk[i]);
    frk[i] = bd::fp16_to_f32(rkc[i]);
    vbad += gv[i] != rvc[i];
  }
  char lbl[128];
  std::snprintf(lbl, sizeof lbl, "attn_prep D=%d R=%d H=%d HK=%d pos=%d M=%d q", D, R, H, HK, pos0, M);
  report(lbl, btft::rel_rms(gq.data(), q.data(), q.size()), 1e-4);
  report("  ... K cache", btft::rel_rms(fk.data(), frk.data(), fk.size()), 1e-4);
  report("  ... V cache mismatches", (double)vbad, 0);
  std::snprintf(lbl, sizeof lbl, "attention + combine nsplit=%d", nsplit);
  report(lbl, btft::rel_rms(go.data(), o.data(), o.size()), 1e-4);
  cudaFree(dqf); cudaFree(dk); cudaFree(dv); cudaFree(dqn); cudaFree(dkn); cudaFree(dkc); cudaFree(dvc);
  cudaFree(dpos); cudaFree(dq); cudaFree(dpart); cudaFree(dout);
}

// ---- embedding (packed table in mapped host memory) + inverse transform
void test_embed(int E, int block, bool signs) {
  const int V = 50, M = 3;
  HostMat T = random_mat(V, E);
  std::vector<float> sg(E);
  for (auto& s : sg) s = ru(0, 1) < 0.5f ? -1.f : 1.f;
  bd::HadamardXform hx;
  hx.active = block > 0;
  hx.block = block;
  hx.signs = signs ? &sg : nullptr;
  const std::vector<int> toks = {7, 0, 49};
  std::vector<float> rx((size_t)M * E);
  for (int m = 0; m < M; ++m) {
    float* xr = &rx[(size_t)m * E];
    for (int i = 0; i < E; ++i)
      xr[i] = (float)(btf::packed_code(&T.codes[(size_t)toks[m] * (E / 4)], i) - 1) *
              bd::fp16_to_f32(T.scales[(size_t)toks[m] * (E / 128) + i / 128]);
    bd::hadamard_inverse(hx, xr, E);
  }
  uint8_t* hbuf = nullptr;
  ck(cudaHostAlloc(reinterpret_cast<void**>(&hbuf), T.codes.size() + T.scales.size() * 2, cudaHostAllocMapped));
  std::memcpy(hbuf, T.codes.data(), T.codes.size());
  std::memcpy(hbuf + T.codes.size(), T.scales.data(), T.scales.size() * 2);
  uint8_t* dbuf = nullptr;
  ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dbuf), hbuf, 0));
  float* dsg = dev(sg);
  int* dt = dev(toks);
  float* dx = dev0<float>(rx.size());
  btf::cuda::Hx h;
  if (hx.active) {
    h.block = block;
    h.signs = signs ? dsg : nullptr;
  }
  btf::cuda::launch_embed(dbuf, reinterpret_cast<const uint16_t*>(dbuf + T.codes.size()), dt, M, E, h, dx, 0);
  const auto g = host(dx, rx.size());
  char lbl[128];
  std::snprintf(lbl, sizeof lbl, "embed E=%d hb=%d signs=%d (mapped table)", E, block, signs);
  report(lbl, btft::rel_rms(g.data(), rx.data(), rx.size()), 1e-6);
  cudaFree(dsg); cudaFree(dt); cudaFree(dx); cudaFreeHost(hbuf);
}

void test_argmax() {
  const int V = 248320;
  std::vector<float> x = randv(V);
  x[123457] = 9.5f;
  x[200001] = 9.5f;  // tie: the smaller index wins
  x[5] = 9.25f;
  float* dx = dev(x);
  void* part = dev0<uint8_t>(btf::cuda::argmax_part_bytes(V));
  int* out = dev0<int>(1);
  btf::cuda::launch_argmax(dx, V, part, out, 0);
  const int g = host(out, 1)[0];
  report("argmax (tie -> smallest index)", g == 123457 ? 0 : 1, 0);
  int* pos = dev0<int>(1);
  btf::cuda::launch_pos_add(pos, 5, 0);
  btf::cuda::launch_pos_add(pos, 1, 0);
  report("pos_add", std::abs(host(pos, 1)[0] - 6), 0);
  cudaFree(dx); cudaFree(part); cudaFree(out); cudaFree(pos);
}

}  // namespace

int main() {
  std::string why;
  if (!btf::gpu_available(&why, 0)) {
    std::printf("test_gpu_kernels: skipped (%s)\n", why.c_str());
    return 77;
  }
  try {
    btf::cuda::kernels_init(0);
    test_repack();
    test_prep(1024, 64, true, false, true, 1);
    test_prep(5120, 1024, true, false, true, 3);
    test_prep(2048, 128, false, true, true, 8);
    test_prep(256, 0, true, false, false, 2);
    test_prep(17408, 1024, false, false, true, 1);
    for (int M : {1, 3, 8, 13}) test_gemv(1024, M);
    test_gemv(5120, 1);
    test_gemv(17408, 2);
    test_ab(true);
    test_ab(false);
    test_conv();
    test_gdn(128, 128, 2, 4, 3);
    test_gdn(64, 32, 2, 4, 2);
    test_gdn(32, 16, 1, 2, 1);
    test_attention(256, 64, 4, 1, 0, 1, 16);
    test_attention(256, 64, 4, 2, 37, 3, 4);
    test_attention(64, 32, 8, 2, 20, 8, 16);
    test_embed(1024, 64, true);
    test_embed(5120, 1024, true);
    test_embed(512, 0, false);
    test_argmax();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_kernels: %s\n", e.what());
    return 1;
  }
  std::printf("test_gpu_kernels: %d failure(s)\n", fails);
  return fails ? 1 : 0;
}
