// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Host repack (PTQ1_0 -> packed T2), support checks and the memory plan.
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>

#include "btf/fast.hpp"
#include "btf/packed.hpp"
#include "dims.hpp"

namespace btf {

void repack_rows(const uint8_t* src, int64_t rows, int64_t K, uint8_t* codes, uint16_t* scales) {
  const int64_t nb = K / kBlock;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t b = 0; b < nb; ++b) {
      uint32_t w[8];
      uint16_t s;
      pack_block(src + (r * nb + b) * kBlockBytes, w, &s);
      std::memcpy(codes + (r * nb + b) * kPackedBlockBytes, w, sizeof w);
      scales[r * nb + b] = s;
    }
  }
}

PackedMatrix repack(const bd::Weight& w, bd::ThreadPool* pool) {
  if (!w.t || w.t->type != bd::T_PTQ1_0) throw std::runtime_error("repack: not a PTQ1_0 tensor");
  PackedMatrix p;
  p.K = w.in();
  p.N = w.out();
  p.hx = w.hx;
  if (p.K % kBlock) throw std::runtime_error("repack: K % 128 != 0 for " + w.t->name);
  p.codes.resize(packed_codes_bytes(p.N, p.K));
  p.scales.resize(packed_scales_bytes(p.N, p.K) / 2);
  const uint64_t rb = (uint64_t)(p.K / kBlock) * kBlockBytes;
  auto run = [&](int64_t b, int64_t e) {
    repack_rows(w.t->data + b * rb, e - b, p.K, p.codes.data() + b * (p.K / 4), p.scales.data() + b * (p.K / kBlock));
  };
  if (pool) pool->parallel_for(p.N, run);
  else run(0, p.N);
  return p;
}

namespace {

bool same_hx(const bd::HadamardXform& a, const bd::HadamardXform& b) {
  return a.active == b.active && a.block == b.block && a.signs == b.signs && a.perm_hd == b.perm_hd &&
         a.perm_nk == b.perm_nk && a.perm_rep == b.perm_rep;
}

void need(bool c, const std::string& what) {
  if (!c) throw std::runtime_error("fast path does not support this model: " + what);
}

void check_mat(const bd::Weight& w, const char* nm) {
  need(w.t != nullptr, std::string("missing ") + nm);
  need(w.t->type == bd::T_PTQ1_0, w.t->name + " is " + bd::type_name(w.t->type) + ", need PTQ1_0");
  need(w.in() % kBlock == 0, w.t->name + ": input width % 128 != 0");
  if (w.hx.active) {
    const int b = w.hx.block;
    need(b >= 32 && b <= 1024 && (b & (b - 1)) == 0, w.t->name + ": hadamard block must be a power of 2 in [32,1024]");
    need(w.in() % b == 0, w.t->name + ": input width % hadamard block != 0");
  }
}

}  // namespace

void check_supported(const bd::Model& m, const FastOptions& o) {
  const Dims d(m, o);
  need(o.n_ctx > 0, "n_ctx must be > 0");
  need(m.token_embd.t && m.token_embd.t->type == bd::T_PTQ1_0, "token_embd must be PTQ1_0");
  need(m.output.t && m.output.t != m.token_embd.t, "tied output head");
  check_mat(m.output, "output");
  if (m.embd_inverse.active) {
    const int b = m.embd_inverse.block;
    need(b >= 32 && b <= 1024 && (b & (b - 1)) == 0 && d.E % b == 0, "embedding inverse hadamard block");
  }
  need(d.E % 128 == 0, "n_embd % 128");
  for (int il = 0; il < d.L; ++il) {
    const bd::LayerWeights& W = m.layers[il];
    check_mat(W.ffn_gate, "ffn_gate");
    check_mat(W.ffn_up, "ffn_up");
    check_mat(W.ffn_down, "ffn_down");
    need(same_hx(W.ffn_gate.hx, W.ffn_up.hx), "ffn_gate/ffn_up input transforms differ");
    if (W.recurrent) {
      check_mat(W.wqkv, "attn_qkv");
      check_mat(W.wgate, "attn_gate");
      check_mat(W.ssm_out, "ssm_out");
      need(same_hx(W.wqkv.hx, W.wgate.hx), "attn_qkv/attn_gate input transforms differ");
      for (const bd::Weight* w : {&W.ssm_alpha, &W.ssm_beta}) {
        need(w->t && (w->t->type == bd::T_BF16 || w->t->type == bd::T_F32), "ssm_alpha/ssm_beta must be BF16 or F32");
        need(!w->hx.active, "ssm_alpha/ssm_beta with an input transform");
      }
      need(d.KC == 4, "ssm conv kernel must be 4");
      need(d.DK == 32 || d.DK == 64 || d.DK == 128, "ssm d_state must be 32, 64 or 128");
      need(d.DV % 8 == 0 && d.DV <= 128, "ssm head_v_dim must be a multiple of 8 and <= 128");
      need(d.NV % d.NK == 0, "dt_rank % n_group");
    } else {
      check_mat(W.wq, "attn_q");
      check_mat(W.wk, "attn_k");
      check_mat(W.wv, "attn_v");
      check_mat(W.wo, "attn_output");
      need(same_hx(W.wq.hx, W.wk.hx) && same_hx(W.wq.hx, W.wv.hx), "attn_q/k/v input transforms differ");
      need(d.D % 32 == 0 && d.D <= 256, "head_dim must be a multiple of 32 and <= 256");
      need(d.R % 2 == 0 && d.R <= d.D, "rope dims");
      need(d.H % d.HK == 0, "n_head % n_head_kv");
    }
  }
}

uint64_t MemoryPlan::device_bytes() const {
  uint64_t s = 0;
  for (auto& i : items) s += i.host ? 0 : i.bytes;
  return s;
}
uint64_t MemoryPlan::host_bytes() const {
  uint64_t s = 0;
  for (auto& i : items) s += i.host ? i.bytes : 0;
  return s;
}

std::string MemoryPlan::table() const {
  std::ostringstream ss;
  char buf[256];
  for (auto& i : items) {
    std::snprintf(buf, sizeof buf, "  %-44s %-6s %14llu B  %9.2f MiB\n", i.what.c_str(), i.host ? "host" : "device",
                  (unsigned long long)i.bytes, i.bytes / 1048576.0);
    ss << buf;
  }
  std::snprintf(buf, sizeof buf, "  %-44s %-6s %14llu B  %9.2f MiB (%.3f GB)\n", "total resident", "device",
                (unsigned long long)device_bytes(), device_bytes() / 1048576.0, device_bytes() / 1e9);
  ss << buf;
  std::snprintf(buf, sizeof buf, "  %-44s %-6s %14llu B  %9.2f MiB (%.3f GB)\n", "total pinned", "host",
                (unsigned long long)host_bytes(), host_bytes() / 1048576.0, host_bytes() / 1e9);
  ss << buf;
  return ss.str();
}

MemoryPlan plan_memory(const bd::Model& m, const FastOptions& o) {
  check_supported(m, o);
  const Dims d(m, o);
  MemoryPlan p;
  auto add = [&](const std::string& w, uint64_t b, bool host = false) { p.items.push_back({w, b, host}); };
  // packed matrices
  uint64_t mats = 0, nmats = 0;
  auto mat = [&](const bd::Weight& w) {
    mats += packed_codes_bytes(w.out(), w.in()) + packed_scales_bytes(w.out(), w.in());
    ++nmats;
  };
  uint64_t ab = 0, small = 0;
  std::set<const std::vector<float>*> signs;
  auto sg = [&](const bd::HadamardXform& hx) {
    if (hx.active && hx.signs) signs.insert(hx.signs);
  };
  for (const auto& W : m.layers) {
    for (const bd::Weight* w : {&W.ffn_gate, &W.ffn_up, &W.ffn_down}) mat(*w), sg(w->hx);
    small += 2ull * d.E * 4;  // attn_norm, post_norm
    if (W.recurrent) {
      for (const bd::Weight* w : {&W.wqkv, &W.wgate, &W.ssm_out}) mat(*w), sg(w->hx);
      for (const bd::Weight* w : {&W.ssm_alpha, &W.ssm_beta})
        ab += (uint64_t)w->in() * w->out() * (w->t->type == bd::T_BF16 ? 2 : 4);
      small += (2ull * d.NV + (uint64_t)d.KC * d.C + d.DV) * 4;
    } else {
      for (const bd::Weight* w : {&W.wq, &W.wk, &W.wv, &W.wo}) mat(*w), sg(w->hx);
      small += 2ull * d.D * 4;
    }
  }
  mat(m.output);
  sg(m.output.hx);
  sg(m.embd_inverse);
  small += (uint64_t)d.E * 4;
  uint64_t sb = 0;
  for (auto* s : signs) sb += s->size() * 4;
  add("packed ternary matrices (" + std::to_string(nmats) + ", codes+scales)", mats);
  add("ssm_alpha + ssm_beta (BF16/F32)", ab);
  add("norms, ssm_a, ssm_dt, conv1d (F32)", small);
  add("hadamard signs (F32)", sb);
  const uint64_t embd = packed_codes_bytes(d.V, d.E) + packed_scales_bytes(d.V, d.E);
  add("token_embd packed", embd, !o.embd_on_device);
  add("KV cache fp16 (" + std::to_string(d.n_attn) + " layers x " + std::to_string(d.n_ctx) + " tokens)",
      (uint64_t)d.n_attn * 2 * d.n_ctx * d.HK * d.D * 2);
  add("GDN state + conv state (" + std::to_string(d.n_gdn) + " layers)",
      (uint64_t)d.n_gdn * ((uint64_t)d.NV * d.DK * d.DV + (uint64_t)(d.KC - 1) * d.C) * 4);
  const uint64_t M = d.maxM;
  uint64_t scratch = 0;
  scratch += M * d.E * 4 * 2;                              // x, xn
  scratch += M * d.Kmax + M * (d.Kmax / 32) * 8;           // q8 activations: xq, da, sa
  scratch += M * ((uint64_t)d.w_a() + d.w_b() + d.w_c() + d.w_o() + d.w_q() + d.F) * 4;
  scratch += M * d.NV * 4 * 2;                             // beta, g
  scratch += M * (uint64_t)d.H * d.nsplit * (d.D + 2) * 4; // attention partials
  scratch += (uint64_t)d.V * 4;                            // logits
  scratch += 4096;                                         // argmax partials, tokens, pos
  add("activations, logits, scratch", scratch);
  p.decode_bytes_fixed = mats + ab + small + sb + (uint64_t)d.E / 4 + (uint64_t)d.E / 128 * 2 +
                         2ull * d.n_gdn * ((uint64_t)d.NV * d.DK * d.DV + (uint64_t)(d.KC - 1) * d.C) * 4 +
                         2ull * d.V * 4;
  p.decode_bytes_per_ctx = (uint64_t)d.n_attn * 2 * d.HK * d.D * 2;
  add("staging for upload (2 x " + std::to_string(o.staging_mb) + " MiB, freed after load)",
      2ull * o.staging_mb * 1048576, true);
  return p;
}

}  // namespace btf
