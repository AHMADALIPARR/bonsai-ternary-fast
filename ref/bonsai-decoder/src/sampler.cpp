// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
#include "bd/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "bd/gguf.hpp"

namespace bd {

void SamplerParams::apply_gguf_defaults(const Gguf& g) {
  if (g.has("general.sampling.top_k")) top_k = (int)g.get_i("general.sampling.top_k");
  if (g.has("general.sampling.top_p")) top_p = (float)g.get_f("general.sampling.top_p");
  if (g.has("general.sampling.min_p")) min_p = (float)g.get_f("general.sampling.min_p");
  if (g.has("general.sampling.temp")) temperature = (float)g.get_f("general.sampling.temp");
}

int32_t Sampler::argmax(const float* l, int32_t n) {
  int32_t best = 0;
  for (int32_t i = 1; i < n; ++i)
    if (l[i] > l[best]) best = i;
  return best;
}

int32_t Sampler::sample(const float* l, int32_t n) {
  if (p_.temperature <= 0.f) return argmax(l, n);
  struct C { float logit; int32_t id; };
  std::vector<C> c(n);
  for (int32_t i = 0; i < n; ++i) c[i] = {l[i], i};
  auto desc = [](const C& a, const C& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };
  // top-k (llama_sampler_top_k_impl); with top-k off, sort everything for top-p
  size_t k = (p_.top_k > 0 && p_.top_k < n) ? (size_t)p_.top_k : (size_t)n;
  std::partial_sort(c.begin(), c.begin() + k, c.end(), desc);
  c.resize(k);
  // top-p (llama_sampler_top_p_apply): float softmax at temperature 1, keep
  // through the first index where the cumulative sum reaches p
  if (p_.top_p < 1.f) {
    const float mx = c[0].logit;
    std::vector<float> p(c.size());
    float sum = 0.f;
    for (size_t i = 0; i < c.size(); ++i) sum += p[i] = std::exp(c[i].logit - mx);
    float cum = 0.f;
    size_t last = c.size();
    for (size_t i = 0; i < c.size(); ++i) {
      cum += p[i] / sum;
      if (cum >= p_.top_p) { last = i + 1; break; }
    }
    c.resize(last);
  }
  // min-p (llama_sampler_min_p_apply, sorted path): logit >= max + log(p)
  if (p_.min_p > 0.f) {
    const float min_logit = c[0].logit + std::log(p_.min_p);
    size_t i = 1;
    for (; i < c.size(); ++i)
      if (c[i].logit < min_logit) break;
    c.resize(i);
  }
  // temperature (llama_sampler_temp_impl)
  for (auto& x : c) x.logit /= p_.temperature;
  // dist (llama_sampler_dist_apply)
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  if (c.size() == 1) { (void)dist(rng_); return c[0].id; }
  const float mx = c[0].logit;
  std::vector<float> p(c.size());
  double sum = 0.0;
  for (size_t i = 0; i < c.size(); ++i) sum += p[i] = std::exp(c[i].logit - mx);
  const double tgt = sum * dist(rng_);
  double run = 0.0;
  for (size_t i = 0; i < c.size(); ++i) {
    run += p[i];
    if (run >= tgt) return c[i].id;
  }
  return c.back().id;
}

}  // namespace bd
