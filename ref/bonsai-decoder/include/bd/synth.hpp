// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Synthetic tiny qwen35 GGUF (in memory or on disk) covering every layer type
// and every convention of the real file: gated delta net + gated full
// attention, prism Hadamard fold with explicit signs, GDN grouped-v
// permutation, inverse embedding transform, PTQ1_0 / BF16 / F32 tensors and a
// byte-level BPE vocab with the chat special tokens.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace bd {

struct SynthOptions {
  int n_layer = 8;            // full attention at il % 4 == 3
  bool hadamard = true;
  bool explicit_signs = true;  // false: sign_mode identity
  bool gdn_v_grouped = true;
  bool tied = false;           // true: no output.weight, hadamard v2
  uint32_t mat_type = 143;     // big matrices (default PTQ1_0)
  uint32_t small_mat_type = 30;  // ssm_alpha / ssm_beta (default BF16)
  uint32_t embd_type = 143;
  uint64_t seed = 1;
  // geometry (defaults are the smallest shapes that keep PTQ1_0 rows whole)
  int n_embd = 128, n_ff = 256, n_head = 4, n_head_kv = 2, head_dim = 32, n_rot = 16;
  int d_state = 32, n_group = 2, dt_rank = 4, conv_kernel = 4;
  int hadamard_block = 64;
};

std::vector<uint8_t> make_synth_qwen35(const SynthOptions& o);
// Vocab the synthetic file carries (byte tokens, a few merges, specials).
std::vector<std::string> synth_vocab_tokens();

}  // namespace bd
