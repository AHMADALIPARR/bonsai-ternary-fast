// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// qwen35 (Qwen3.5 hybrid: gated delta net + gated full attention) model view
// over a GGUF, plus the prism Hadamard fold metadata. No weights are copied;
// every Weight points into the mmap. See docs/graph-conventions.md.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "bd/gguf.hpp"

namespace bd {

struct HParams {
  int n_layer = 0, n_embd = 0, n_ff = 0, n_head = 0, n_head_kv = 0;
  int head_dim = 0;   // attention.key_length (== value_length)
  int n_rot = 0;      // rope.dimension_count
  int sections[4] = {0, 0, 0, 0};
  float rope_base = 10000.f, rms_eps = 1e-6f;
  int conv_kernel = 0, d_state = 0, n_group = 0, dt_rank = 0, d_inner = 0;
  int full_attn_interval = 4;
  int n_vocab = 0;
  int n_ctx_train = 0;
  bool is_recurrent(int il) const { return ((il + 1) % full_attn_interval) != 0; }
  // linear attention derived sizes
  int gdn_head_k() const { return d_state; }            // 128
  int gdn_head_v() const { return d_inner / dt_rank; }  // 128
  int gdn_qk_dim() const { return d_state * n_group; }  // 2048
  int gdn_conv_ch() const { return 2 * gdn_qk_dim() + d_inner; }  // 10240
};

// Activation transform applied to a matmul input before a folded weight:
// optional tiled->grouped permutation, sign flip, blockwise normalized WHT.
struct HadamardXform {
  bool active = false;
  int block = 0;
  const std::vector<float>* signs = nullptr;  // nullptr: identity sign mode
  int perm_hd = 0, perm_nk = 0, perm_rep = 0;   // perm_rep > 1: apply perm
};

struct Weight {
  const GgufTensor* t = nullptr;
  HadamardXform hx;  // input transform (folded weights)
  int64_t in() const { return t ? t->dim(0) : 0; }
  int64_t out() const { return t ? t->dim(1) : 0; }
  explicit operator bool() const { return t != nullptr; }
};

struct LayerWeights {
  bool recurrent = false;
  Weight attn_norm, post_norm, ffn_gate, ffn_up, ffn_down;
  // full attention
  Weight wq, wk, wv, wo, q_norm, k_norm;
  // gated delta net
  Weight wqkv, wgate, ssm_alpha, ssm_beta, ssm_a, ssm_dt, conv1d, ssm_norm, ssm_out;
};

struct HadamardMeta {
  int version = 0;
  int block = 0;
  std::string sign_mode;
  std::map<int, std::vector<float>> signs;  // by input width
  std::set<std::string> weight_names;
  std::set<std::string> inverse_names;
  bool gdn_v_grouped = false;
  bool tied_output = false;
};

class Model {
 public:
  // Opens path (mmap) and binds every tensor. Throws on any inconsistency.
  static std::unique_ptr<Model> load(const std::string& path);
  // Binds a model over an already parsed GGUF (synthetic tests).
  static std::unique_ptr<Model> from_gguf(std::unique_ptr<Gguf> g);

  std::unique_ptr<Gguf> gguf;
  std::string arch;
  HParams hp;
  HadamardMeta hdmd;
  Weight token_embd, output_norm, output;
  HadamardXform embd_inverse;  // inverse transform after the embedding lookup
  std::vector<LayerWeights> layers;

  std::string describe() const;

 private:
  void bind();
  Weight w(const std::string& name, bool required = true);
};

}  // namespace bd
