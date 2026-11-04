// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Synthetic models and comparison helpers shared by the tests.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "bd/model.hpp"
#include "bd/synth.hpp"
#include "bd/test_util.hpp"

namespace btft {

struct Config {
  const char* label;
  bd::SynthOptions so;
};

// Small models covering every convention of the real file, plus one with the
// real per-head geometry (head_dim 256, n_rot 64, d_state 128, block 1024).
inline std::vector<Config> configs() {
  std::vector<Config> v;
  bd::SynthOptions base;
  v.push_back({"default (block 64)", base});
  bd::SynthOptions s = base;
  s.explicit_signs = false;
  v.push_back({"identity signs", s});
  s = base;
  s.hadamard = false;
  v.push_back({"no hadamard", s});
  s = base;
  s.gdn_v_grouped = false;
  v.push_back({"gdn v tiled", s});
  s = base;
  s.small_mat_type = 0;  // F32 alpha/beta
  v.push_back({"f32 alpha/beta", s});
  s = base;
  s.n_layer = 4;
  s.n_embd = 1024;
  s.n_ff = 2048;
  s.n_head = 4;
  s.n_head_kv = 1;
  s.head_dim = 256;
  s.n_rot = 64;
  s.d_state = 128;
  s.n_group = 2;
  s.dt_rank = 8;
  s.hadamard_block = 1024;
  v.push_back({"real head geometry", s});
  return v;
}

inline std::unique_ptr<bd::Model> make_model(const bd::SynthOptions& so) {
  return bd::Model::from_gguf(bd::Gguf::from_memory(bd::make_synth_qwen35(so)));
}

inline double rel_rms(const float* a, const float* ref, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - ref[i];
    num += d * d;
    den += (double)ref[i] * ref[i];
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

inline int argmax(const std::vector<float>& v) { return (int)(std::max_element(v.begin(), v.end()) - v.begin()); }

inline const std::vector<int>& prompt() {
  static const std::vector<int> p = {259, 272, 44, 32, 259, 261, 103, 97, 264};
  return p;
}

}  // namespace btft
