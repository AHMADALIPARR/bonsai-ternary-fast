// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// fp32 CPU building blocks for the reference decoder.
#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "bd/model.hpp"

namespace bd {

class ThreadPool {
 public:
  explicit ThreadPool(int n_threads);  // n_threads <= 0: hardware concurrency
  ~ThreadPool();
  int size() const { return n_; }
  // Runs fn(begin, end) over [0, n) split in contiguous chunks; blocks.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

 private:
  void worker(int id);
  int n_;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
  int64_t total_ = 0;
  uint64_t gen_ = 0;
  int pending_ = 0;
  bool stop_ = false;
};

// In-place normalized Sylvester Walsh-Hadamard transform of n (power of 2)
// values: x <- H x with H[r][c] = (-1)^popcount(r&c) / sqrt(n).
void fwht(float* x, int n);
// Folded-weight input transform: out = blockWHT(signs * perm(x)).
void hadamard_input(const HadamardXform& hx, const float* x, int64_t n, float* out);
// Embedding inverse (llama-graph.cpp build_embd_rows): x <- signs * blockWHT(x).
void hadamard_inverse(const HadamardXform& hx, float* x, int64_t n);

// y = x / sqrt(mean(x^2) + eps) * w   (w may be nullptr), over n values
void rmsnorm(const float* x, const float* w, int64_t n, float eps, float* y);
// y = x / max(||x||_2, eps)
void l2norm(const float* x, int64_t n, float eps, float* y);
inline float silu(float x) { return x / (1.0f + __builtin_expf(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }
// ggml softplus with threshold 20 (as in the fused gated-delta-net path)
float softplus(float x);

// Whole tensor to fp32 (small tensors: norms, conv, ssm_a, ...)
std::vector<float> to_f32(const Weight& w);
// Row r of a 2-D weight to fp32.
void row_f32(const Weight& w, int64_t r, float* out);

// y[out] = W x for a GGUF weight stored [in, out] (ggml order); applies the
// Hadamard input transform when the weight is folded. fp32 accumulation.
// ggml_act: round the activation the way ggml-cpu does before its dot product
// (Q8_0 blocks of 32 for PTQ1_0 / Q8_0 weights, bf16 for BF16 weights, fp16
// for F16 weights) so traces can be compared with llama.cpp op by op.
void matvec(ThreadPool& pool, const Weight& w, const float* x, float* y, bool ggml_act = false);
// Same but without the transform (x already in the weight's basis).
void matvec_raw(ThreadPool& pool, const GgufTensor& t, const float* x, float* y, bool ggml_act = false);
// Q8_0 round trip of n values (n % 32 == 0), ggml-cpu x86 quantize_row_q8_0.
void round_q8_0(const float* x, int64_t n, float* out);

}  // namespace bd
