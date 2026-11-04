// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Sources for each op (prism branch, 62c418c):
//   rmsnorm   ggml/src/ggml-cpu/ops.cpp:4203-4260 (sum in double, mean + eps)
//   l2norm    ggml/src/ggml-cpu/ops.cpp:4582-4625 (scale = 1/max(sqrt(sum), eps))
//   WHT       src/llama-model.cpp:2342-2352 (matrix), src/llama-impl.h:57 (blockwise)
//   transform src/llama-graph.cpp:1684-1715 (perm, signs, WHT), 2676-2692 (inverse)
#include "bd/cpu_ops.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "bd/types.hpp"

namespace bd {

// ---------------------------------------------------------------- pool
ThreadPool::ThreadPool(int n) {
  if (n <= 0) n = (int)std::thread::hardware_concurrency();
  if (n <= 0) n = 1;
  n_ = n;
  for (int i = 1; i < n_; ++i) th_.emplace_back(&ThreadPool::worker, this, i);
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : th_) t.join();
}

void ThreadPool::worker(int id) {
  uint64_t seen = 0;
  for (;;) {
    const std::function<void(int64_t, int64_t)>* fn;
    int64_t total;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [&] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      fn = fn_;
      total = total_;
    }
    const int64_t b = total * id / n_, e = total * (id + 1) / n_;
    if (b < e) (*fn)(b, e);
    {
      std::lock_guard<std::mutex> l(mu_);
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
  if (n_ == 1 || n < 2 * n_) {
    fn(0, n);
    return;
  }
  {
    std::lock_guard<std::mutex> l(mu_);
    fn_ = &fn;
    total_ = n;
    pending_ = n_ - 1;
    ++gen_;
  }
  cv_.notify_all();
  fn(0, n / n_);
  std::unique_lock<std::mutex> l(mu_);
  done_cv_.wait(l, [&] { return pending_ == 0; });
}

// ---------------------------------------------------------------- hadamard
void fwht(float* x, int n) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += len << 1) {
      for (int j = i; j < i + len; ++j) {
        const float a = x[j], b = x[j + len];
        x[j] = a + b;
        x[j + len] = a - b;
      }
    }
  }
  const float s = 1.0f / std::sqrt((float)n);
  for (int i = 0; i < n; ++i) x[i] *= s;
}

void hadamard_input(const HadamardXform& hx, const float* x, int64_t n, float* out) {
  if (!hx.active) {
    if (out != x) std::memcpy(out, x, n * sizeof(float));
    return;
  }
  if (hx.perm_rep > 1) {
    // tiled [hd, nk, rep] -> grouped [hd, rep, nk] (llama-graph.cpp:1699-1706)
    const int hd = hx.perm_hd, nk = hx.perm_nk, rep = hx.perm_rep;
    if ((int64_t)hd * nk * rep != n) throw std::runtime_error("hadamard perm geometry");
    std::vector<float> tmp(x, x + n);
    for (int k = 0; k < nk; ++k)
      for (int r = 0; r < rep; ++r)
        for (int d = 0; d < hd; ++d) out[d + hd * (r + rep * k)] = tmp[d + hd * (k + nk * r)];
  } else if (out != x) {
    std::memcpy(out, x, n * sizeof(float));
  }
  if (hx.signs) {
    if ((int64_t)hx.signs->size() != n) throw std::runtime_error("hadamard sign width");
    for (int64_t i = 0; i < n; ++i) out[i] *= (*hx.signs)[i];
  }
  for (int64_t b = 0; b < n; b += hx.block) fwht(out + b, hx.block);
}

void hadamard_inverse(const HadamardXform& hx, float* x, int64_t n) {
  if (!hx.active) return;
  for (int64_t b = 0; b < n; b += hx.block) fwht(x + b, hx.block);
  if (hx.signs)
    for (int64_t i = 0; i < n; ++i) x[i] *= (*hx.signs)[i];
}

// ---------------------------------------------------------------- norms
void rmsnorm(const float* x, const float* w, int64_t n, float eps, float* y) {
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
  const float mean = (float)(sum / n);
  const float scale = 1.0f / std::sqrt(mean + eps);
  for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

void l2norm(const float* x, int64_t n, float eps, float* y) {
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += (double)x[i] * x[i];
  const float scale = 1.0f / std::fmax(std::sqrt((float)sum), eps);
  for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale;
}

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// ---------------------------------------------------------------- weights
std::vector<float> to_f32(const Weight& w) {
  const int64_t n = w.t->n_elements();
  std::vector<float> v(n);
  dequant_row(w.t->type, w.t->data, v.data(), n);
  return v;
}

void row_f32(const Weight& w, int64_t r, float* out) {
  const int64_t in = w.t->dim(0);
  dequant_row(w.t->type, w.t->data + r * row_bytes(w.t->type, in), out, in);
}

namespace {
struct Ptq1Lut {
  float v[256][5];
  Ptq1Lut() {
    static const uint32_t p3[5] = {1, 3, 9, 27, 81};
    for (int b = 0; b < 256; ++b)
      for (int n = 0; n < 5; ++n) v[b][n] = (float)((int)(((uint32_t)(uint8_t)(b * p3[n]) * 3u) >> 8) - 1);
  }
};
const Ptq1Lut& lut() {
  static Ptq1Lut l;
  return l;
}

// dot of one PTQ1_0 row with x (layout: types.hpp ptq1_0_trit)
float dot_ptq1_0(const uint8_t* row, const float* x, int64_t n) {
  const auto& L = lut().v;
  double acc = 0.0;
  for (int64_t b = 0; b < n / 128; ++b, row += 28, x += 128) {
    float s = 0.f;
    for (int m = 0; m < 16; ++m) {
      const float* t = L[row[m]];
      s += t[0] * x[m] + t[1] * x[16 + m] + t[2] * x[32 + m] + t[3] * x[48 + m] + t[4] * x[64 + m];
    }
    for (int m = 0; m < 8; ++m) {
      const float* t = L[row[16 + m]];
      s += t[0] * x[80 + m] + t[1] * x[88 + m] + t[2] * x[96 + m] + t[3] * x[104 + m] + t[4] * x[112 + m];
    }
    for (int h = 0; h < 2; ++h) {
      const float* t = L[row[24 + h]];
      s += t[0] * x[120 + h] + t[1] * x[122 + h] + t[2] * x[124 + h] + t[3] * x[126 + h];
    }
    acc += (double)(s * fp16_to_f32((uint16_t)(row[26] | (row[27] << 8))));
  }
  return (float)acc;
}
}  // namespace

void round_q8_0(const float* x, int64_t n, float* out) {
  for (int64_t b = 0; b < n; b += 32) {
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b + i]));
    const float d = fp16_to_f32(f32_to_fp16(amax / 127.f));
    const float id = amax != 0.f ? 127.f / amax : 0.f;
    for (int i = 0; i < 32; ++i) out[b + i] = std::nearbyint(x[b + i] * id) * d;
  }
}

void matvec_raw(ThreadPool& pool, const GgufTensor& t, const float* x0, float* y, bool ggml_act) {
  const int64_t in = t.dim(0), out = t.dim(1);
  std::vector<float> xq;
  const float* x = x0;
  if (ggml_act) {
    xq.resize(in);
    if ((t.type == T_PTQ1_0 || t.type == T_Q8_0) && in % 32 == 0) {
      round_q8_0(x0, in, xq.data());
      x = xq.data();
    } else if (t.type == T_BF16) {
      for (int64_t i = 0; i < in; ++i) xq[i] = bf16_to_f32(f32_to_bf16(x0[i]));
      x = xq.data();
    } else if (t.type == T_F16) {
      for (int64_t i = 0; i < in; ++i) xq[i] = fp16_to_f32(f32_to_fp16(x0[i]));
      x = xq.data();
    }
  }
  const uint64_t rb = row_bytes(t.type, in);
  if (t.type == T_PTQ1_0) {
    pool.parallel_for(out, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; ++r) y[r] = dot_ptq1_0(t.data + r * rb, x, in);
    });
    return;
  }
  if (t.type == T_F32) {
    pool.parallel_for(out, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; ++r) {
        const float* w = (const float*)(t.data + r * rb);
        double acc = 0.0;
        for (int64_t i = 0; i < in; ++i) acc += (double)w[i] * x[i];
        y[r] = (float)acc;
      }
    });
    return;
  }
  pool.parallel_for(out, [&](int64_t b, int64_t e) {
    std::vector<float> buf(in);
    for (int64_t r = b; r < e; ++r) {
      dequant_row(t.type, t.data + r * rb, buf.data(), in);
      double acc = 0.0;
      for (int64_t i = 0; i < in; ++i) acc += (double)buf[i] * x[i];
      y[r] = (float)acc;
    }
  });
}

void matvec(ThreadPool& pool, const Weight& w, const float* x, float* y, bool ggml_act) {
  if (!w.hx.active) {
    matvec_raw(pool, *w.t, x, y, ggml_act);
    return;
  }
  std::vector<float> xt(w.in());
  hadamard_input(w.hx, x, w.in(), xt.data());
  matvec_raw(pool, *w.t, xt.data(), y, ggml_act);
}

}  // namespace bd
