// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Token sampler that follows prism llama.cpp's default CPU chain
// (src/llama-sampler.cpp, common/sampling.cpp):
//   top-k -> top-p -> min-p -> temperature -> dist
// top-p and min-p look at temperature-1 probabilities, and the temperature
// only changes the final draw. dist uses std::mt19937 and
// uniform_real_distribution<double>, as prism does. With the same logits and
// seed the draws are the same, but only if ties sort the same way.
#pragma once
#include <cstdint>
#include <random>

namespace bd {

class Gguf;

struct SamplerParams {
  float temperature = 0.f;  // <= 0: greedy (argmax, lowest id on ties, like llama_sampler_greedy)
  int top_k = 40;           // <= 0: off           (common default 40)
  float top_p = 0.95f;      // >= 1: off           (common default 0.95)
  float min_p = 0.05f;      // <= 0: off           (common default 0.05)
  uint32_t seed = 0;
  // Fills temperature/top_k/top_p/min_p from general.sampling.* when the
  // GGUF has them, the way common_params_sampling does. The Bonsai file has
  // top_k 20, top_p 0.95 and temp 1.
  void apply_gguf_defaults(const Gguf& g);
};

class Sampler {
 public:
  explicit Sampler(const SamplerParams& p) : p_(p), rng_(p.seed) {}
  int32_t sample(const float* logits, int32_t n_vocab);
  static int32_t argmax(const float* logits, int32_t n_vocab);

 private:
  SamplerParams p_;
  std::mt19937 rng_;
};

}  // namespace bd
