// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Packed layout: every code and scale after repack matches the PTQ1_0 source,
// the Q8 quantizer matches bd::round_q8_0, and the integer packed dot matches
// the reference PTQ1_0 matvec with ggml activation rounding.
#include <cstring>
#include <random>

#include "bd/cpu_ops.hpp"
#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "btf/packed.hpp"
#include "test_common.hpp"

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  const int N = 37, K = 1024;
  std::vector<float> w((size_t)N * K);
  for (auto& v : w) v = nd(rng) * (std::uniform_real_distribution<float>(0, 1)(rng) < 0.3f ? 0.f : 1.f);
  std::vector<uint8_t> src((size_t)N * (K / 128) * 28);
  for (int r = 0; r < N; ++r) bd::quant_row(bd::T_PTQ1_0, &w[(size_t)r * K], &src[(size_t)r * (K / 128) * 28], K);

  // 1. repack: every element
  std::vector<uint8_t> codes(btf::packed_codes_bytes(N, K));
  std::vector<uint16_t> scales(btf::packed_scales_bytes(N, K) / 2);
  btf::repack_rows(src.data(), N, K, codes.data(), scales.data());
  int hist[3] = {0, 0, 0};
  for (int r = 0; r < N; ++r) {
    for (int e = 0; e < K; ++e) {
      const uint8_t* blk = &src[((size_t)r * (K / 128) + e / 128) * 28];
      const int t = bd::ptq1_0_trit(blk, e % 128);
      const int c = btf::packed_code(&codes[(size_t)r * (K / 4)], e);
      BD_CHECK(c == t + 1);
      hist[c]++;
    }
    for (int b = 0; b < K / 128; ++b) {
      const uint8_t* blk = &src[((size_t)r * (K / 128) + b) * 28];
      BD_CHECK(scales[(size_t)r * (K / 128) + b] == (uint16_t)(blk[26] | (blk[27] << 8)));
    }
  }
  BD_CHECK(hist[0] > 0 && hist[1] > 0 && hist[2] > 0);

  // 2. Q8 quantizer vs reference rounding (including an all-zero block)
  std::vector<float> x(K);
  for (auto& v : x) v = nd(rng) * 3.f;
  for (int i = 64; i < 96; ++i) x[i] = 0.f;
  std::vector<float> xr(K);
  bd::round_q8_0(x.data(), K, xr.data());
  std::vector<int8_t> q(K);
  std::vector<float> da(K / 32);
  std::vector<int> sa(K / 32);
  for (int b = 0; b < K / 32; ++b) {
    sa[b] = btf::q8_block_host(&x[b * 32], &q[b * 32], &da[b]);
    int s = 0;
    for (int i = 0; i < 32; ++i) {
      BD_CHECK((float)q[b * 32 + i] * da[b] == xr[b * 32 + i]);
      s += q[b * 32 + i];
    }
    BD_CHECK(s == sa[b]);
  }

  // 3. packed integer dot vs the reference PTQ1_0 matvec (ggml_act)
  bd::GgufTensor t;
  t.name = "w";
  t.ne = {K, N};
  t.type = bd::T_PTQ1_0;
  t.data = src.data();
  bd::ThreadPool pool(1);
  std::vector<float> yref(N), y(N);
  bd::matvec_raw(pool, t, x.data(), yref.data(), true);
  for (int r = 0; r < N; ++r) {
    float acc = 0.f;
    const uint32_t* aw = (const uint32_t*)q.data();
    for (int s = 0; s < K / 32; ++s) {
      uint32_t w0, w1;
      std::memcpy(&w0, &codes[(size_t)r * (K / 4) + s * 8], 4);
      std::memcpy(&w1, &codes[(size_t)r * (K / 4) + s * 8 + 4], 4);
      const int idot = btf::sub_dot_host(w0, w1, aw + s * 8) - sa[s];
      acc += bd::fp16_to_f32(scales[(size_t)r * (K / 128) + s / 4]) * da[s] * (float)idot;
    }
    y[r] = acc;
  }
  const double rr = btft::rel_rms(y.data(), yref.data(), N);
  std::printf("repack: %d x %d all codes/scales match (codes 0/1/2: %d/%d/%d); q8 exact; packed dot rel rms %.2e\n",
              N, K, hist[0], hist[1], hist[2], rr);
  BD_CHECK(rr < 1e-5);
  return 0;
}
