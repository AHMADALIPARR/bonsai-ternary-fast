// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Derived sizes shared by the memory plan, the CPU emulator and the CUDA decoder.
#pragma once
#include <algorithm>
#include <cstdint>

#include "bd/model.hpp"
#include "btf/fast.hpp"

namespace btf {

struct Dims {
  int L = 0, E = 0, F = 0, V = 0, H = 0, HK = 0, D = 0, R = 0;
  int C = 0, DK = 0, DV = 0, NK = 0, NV = 0, QK = 0, DI = 0, KC = 0;
  int maxM = 1, n_ctx = 0, nsplit = 1;
  int Kmax = 0;       // widest input of any packed matrix
  int n_attn = 0, n_gdn = 0;
  float eps = 1e-6f, rope_base = 1e4f;

  Dims(const bd::Model& m, const FastOptions& o) {
    const bd::HParams& hp = m.hp;
    L = hp.n_layer; E = hp.n_embd; F = hp.n_ff; V = hp.n_vocab; H = hp.n_head; HK = hp.n_head_kv;
    D = hp.head_dim; R = hp.n_rot; C = hp.gdn_conv_ch(); DK = hp.gdn_head_k(); DV = hp.gdn_head_v();
    NK = hp.n_group; NV = hp.dt_rank; QK = hp.gdn_qk_dim(); DI = hp.d_inner; KC = hp.conv_kernel;
    eps = hp.rms_eps; rope_base = hp.rope_base;
    maxM = std::max(1, std::min(8, o.max_batch));
    n_ctx = o.n_ctx;
    nsplit = std::max(1, std::min(64, o.attn_splits));
    for (int il = 0; il < L; ++il) (hp.is_recurrent(il) ? n_gdn : n_attn)++;
    Kmax = std::max({E, F, n_gdn ? DI : 0, n_attn ? H * D : 0});
  }
  // scratch widths (floats per batch row)
  int w_a() const { return std::max(n_gdn ? C : 0, n_attn ? 2 * H * D : 0); }    // qkv | q+gate
  int w_b() const { return std::max(n_gdn ? DI : 0, n_attn ? HK * D : 0); }      // z | k
  int w_c() const { return std::max(n_gdn ? C : 0, n_attn ? HK * D : 0); }       // conv out | v
  int w_o() const { return std::max(n_gdn ? DI : 0, n_attn ? H * D : 0); }       // mixer out
  int w_q() const { return n_attn ? H * D : 0; }                                 // roped q
  int chunk_keys() const { return (n_ctx + nsplit - 1) / nsplit; }
};

}  // namespace btf
