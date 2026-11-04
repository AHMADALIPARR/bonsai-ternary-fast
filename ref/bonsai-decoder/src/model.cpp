// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Hyperparameters: prism src/llama-model.cpp load_hparams (QWEN35 case).
// Hadamard metadata: prism src/llama-model.cpp:2091-2255 (validation) and
// 2336-2414 (rotation, sign vector per input width, GDN v permutation).
#include "bd/model.hpp"

#include <sstream>
#include <stdexcept>

#include "bd/types.hpp"

namespace bd {

namespace {
std::runtime_error err(const std::string& s) { return std::runtime_error("model: " + s); }
}  // namespace

std::unique_ptr<Model> Model::load(const std::string& path) {
  return from_gguf(Gguf::open(path, false));
}

std::unique_ptr<Model> Model::from_gguf(std::unique_ptr<Gguf> g) {
  auto m = std::unique_ptr<Model>(new Model());
  m->gguf = std::move(g);
  m->bind();
  return m;
}

Weight Model::w(const std::string& name, bool required) {
  Weight r;
  r.t = gguf->tensor(name);
  if (!r.t) {
    if (required) throw err("missing tensor " + name);
    return r;
  }
  if (!type_info(r.t->type).dequant) throw err("no dequantizer for " + name + " type " + type_name(r.t->type));
  if (hdmd.weight_names.count(name)) {
    r.hx.active = true;
    r.hx.block = hdmd.block;
    const int width = (int)r.t->dim(0);
    if (width % hdmd.block) throw err("hadamard width not a multiple of block: " + name);
    if (hdmd.sign_mode == "explicit") {
      auto it = hdmd.signs.find(width);
      if (it == hdmd.signs.end()) throw err("no hadamard sign vector for width " + std::to_string(width) + " (" + name + ")");
      r.hx.signs = &it->second;
    }
    // llama-model.cpp:2410-2414: ssm_out of a GDN reads tiled v heads, fold used grouped
    if (hdmd.gdn_v_grouped && name.find(".ssm_out.") != std::string::npos) {
      const int n_v = hp.dt_rank, n_k = hp.n_group;
      if (n_k <= 0 || n_v <= 0 || n_v % n_k || width % n_v) throw err("bad GDN head geometry for " + name);
      r.hx.perm_hd = width / n_v;
      r.hx.perm_nk = n_k;
      r.hx.perm_rep = n_v / n_k;
    }
  }
  return r;
}

void Model::bind() {
  const Gguf& g = *gguf;
  arch = g.get_s("general.architecture");
  if (arch != "qwen35") throw err("unsupported architecture '" + arch + "' (this decoder implements qwen35)");
  const std::string p = arch + ".";
  hp.n_layer = (int)g.get_i(p + "block_count");
  hp.n_embd = (int)g.get_i(p + "embedding_length");
  hp.n_ff = (int)g.get_i(p + "feed_forward_length");
  hp.n_head = (int)g.get_i(p + "attention.head_count");
  hp.n_head_kv = (int)g.get_i(p + "attention.head_count_kv");
  hp.head_dim = (int)g.get_i(p + "attention.key_length", hp.n_embd / hp.n_head);
  if (g.get_i(p + "attention.value_length", hp.head_dim) != hp.head_dim) throw err("key_length != value_length");
  hp.n_rot = (int)g.get_i(p + "rope.dimension_count", hp.head_dim);
  hp.rope_base = (float)g.get_f(p + "rope.freq_base", 10000.0);
  if (g.get_f(p + "rope.scaling.factor", 1.0) != 1.0) throw err("rope scaling not implemented");
  hp.rms_eps = (float)g.get_f(p + "attention.layer_norm_rms_epsilon");
  if (const GgufValue* s = g.find(p + "rope.dimension_sections")) {
    for (size_t i = 0; i < 4 && i < s->ai.size(); ++i) hp.sections[i] = (int)s->ai[i];
  }
  hp.conv_kernel = (int)g.get_i(p + "ssm.conv_kernel");
  hp.d_state = (int)g.get_i(p + "ssm.state_size");
  hp.n_group = (int)g.get_i(p + "ssm.group_count");
  hp.dt_rank = (int)g.get_i(p + "ssm.time_step_rank");
  hp.d_inner = (int)g.get_i(p + "ssm.inner_size");
  hp.full_attn_interval = (int)g.get_i(p + "full_attention_interval", 4);
  hp.n_ctx_train = (int)g.get_i(p + "context_length", 4096);
  if (g.get_i(p + "nextn_predict_layers", 0) != 0) {
    // MTP blocks are appended after the main layers and not run in the main pass
    hp.n_layer -= (int)g.get_i(p + "nextn_predict_layers", 0);
  }
  if (hp.n_head % hp.n_head_kv) throw err("n_head % n_head_kv != 0");
  if (hp.dt_rank % hp.n_group) throw err("ssm v heads % k heads != 0");
  if (hp.n_rot > hp.head_dim || hp.n_rot % 2) throw err("bad n_rot");

  // ---- prism Hadamard metadata (llama-model.cpp:2091-2255)
  hdmd.version = (int)g.get_i("prism.hadamard.version", 0);
  hdmd.tied_output = g.get_b("prism.hadamard.tied_output", false);
  if (hdmd.version) {
    if (hdmd.version != 1 && hdmd.version != 2) throw err("unsupported prism.hadamard.version");
    if ((hdmd.version == 2) != hdmd.tied_output) throw err("hadamard version/tied_output mismatch");
    hdmd.block = (int)g.get_i("prism.hadamard.block_size");
    if (hdmd.block <= 0 || (hdmd.block & (hdmd.block - 1))) throw err("bad hadamard block size");
    if (g.get_s("prism.hadamard.transform") != "normalized-sylvester-walsh-hadamard") throw err("unsupported hadamard transform");
    if (g.get_s("prism.hadamard.axis") != "input-last-dimension") throw err("unsupported hadamard axis");
    hdmd.sign_mode = g.get_s("prism.hadamard.sign_mode");
    if (hdmd.sign_mode != "identity" && hdmd.sign_mode != "explicit") throw err("unsupported sign mode");
    const GgufValue* wn = g.find("prism.hadamard.weight_names");
    if (!wn || wn->as.empty()) throw err("prism.hadamard.weight_names missing/empty");
    for (auto& s : wn->as) {
      if (!hdmd.weight_names.insert(s).second) throw err("duplicate hadamard weight " + s);
    }
    if (hdmd.sign_mode == "explicit") {
      const GgufValue* sw = g.find("prism.hadamard.sign_widths");
      const GgufValue* sv = g.find("prism.hadamard.sign_values");
      if (!sw || !sv || sw->ai.empty()) throw err("explicit sign mode without sign arrays");
      size_t off = 0;
      for (int64_t width : sw->ai) {
        if (width <= 0 || width % hdmd.block || off + width > sv->ai.size()) throw err("bad sign width");
        if (hdmd.signs.count((int)width)) throw err("duplicate sign width");
        auto& vec = hdmd.signs[(int)width];
        vec.resize(width);
        for (int64_t i = 0; i < width; ++i) {
          const int64_t v = sv->ai[off + i];
          if (v != 1 && v != -1) throw err("sign values must be +/-1");
          vec[i] = (float)v;
        }
        off += width;
      }
      if (off != sv->ai.size()) throw err("sign_values length mismatch");
    }
    hdmd.gdn_v_grouped = g.get_b("prism.hadamard.gdn_v_grouped", false);
    if (const GgufValue* inv = g.find("prism.hadamard.inverse_weight_names")) {
      for (auto& s : inv->as) {
        if (s != "token_embd.weight") throw err("unsupported inverse table " + s);
        hdmd.inverse_names.insert(s);
      }
    }
    if (hdmd.tied_output) hdmd.weight_names.insert("token_embd.weight");
  }

  // ---- tensors
  token_embd = w("token_embd.weight");
  if (token_embd.hx.active && !hdmd.tied_output) throw err("token_embd folded without tied output");
  token_embd.hx = HadamardXform{};  // the lookup itself is never transformed
  hp.n_vocab = (int)token_embd.out();
  if (hdmd.inverse_names.count("token_embd.weight")) {
    embd_inverse.active = true;
    embd_inverse.block = hdmd.block;
    if (hdmd.sign_mode == "explicit") {
      auto it = hdmd.signs.find((int)token_embd.in());
      if (it == hdmd.signs.end()) throw err("no sign vector for token_embd width");
      embd_inverse.signs = &it->second;
    }
  }
  output_norm = w("output_norm.weight");
  output = w("output.weight", false);
  if (!output) {
    // tied output: the head is token_embd (with the fold transform under v2)
    output = w("token_embd.weight");
    if (!hdmd.tied_output && embd_inverse.active) throw err("tied hadamard output needs version 2");
  }

  layers.resize(hp.n_layer);
  for (int il = 0; il < hp.n_layer; ++il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    LayerWeights& L = layers[il];
    L.recurrent = hp.is_recurrent(il);
    L.attn_norm = w(b + "attn_norm.weight");
    L.post_norm = w(b + "post_attention_norm.weight");
    L.ffn_gate = w(b + "ffn_gate.weight");
    L.ffn_up = w(b + "ffn_up.weight");
    L.ffn_down = w(b + "ffn_down.weight");
    if (L.recurrent) {
      L.wqkv = w(b + "attn_qkv.weight");
      L.wgate = w(b + "attn_gate.weight");
      L.ssm_alpha = w(b + "ssm_alpha.weight");
      L.ssm_beta = w(b + "ssm_beta.weight");
      L.ssm_a = w(b + "ssm_a");
      L.ssm_dt = w(b + "ssm_dt.bias");
      L.conv1d = w(b + "ssm_conv1d.weight");
      L.ssm_norm = w(b + "ssm_norm.weight");
      L.ssm_out = w(b + "ssm_out.weight");
      if (L.wqkv.out() != hp.gdn_conv_ch()) throw err(b + "attn_qkv width mismatch");
      if (L.wgate.out() != hp.d_inner) throw err(b + "attn_gate width mismatch");
      if (L.conv1d.in() != hp.conv_kernel || L.conv1d.out() != hp.gdn_conv_ch()) throw err(b + "conv1d shape");
      if (L.ssm_out.in() != hp.d_inner) throw err(b + "ssm_out shape");
      if (L.ssm_norm.t->n_elements() != hp.gdn_head_v()) throw err(b + "ssm_norm shape");
    } else {
      L.wq = w(b + "attn_q.weight");
      L.wk = w(b + "attn_k.weight");
      L.wv = w(b + "attn_v.weight");
      L.wo = w(b + "attn_output.weight");
      L.q_norm = w(b + "attn_q_norm.weight");
      L.k_norm = w(b + "attn_k_norm.weight");
      if (L.wq.out() != 2LL * hp.head_dim * hp.n_head) throw err(b + "attn_q must hold q and gate");
      if (L.wk.out() != (int64_t)hp.head_dim * hp.n_head_kv) throw err(b + "attn_k shape");
      if (L.wo.in() != (int64_t)hp.head_dim * hp.n_head) throw err(b + "attn_output shape");
    }
  }
  // every folded name must be bound to a weight we apply it to
  for (auto& n : hdmd.weight_names) {
    if (!gguf->tensor(n)) throw err("hadamard weight not in file: " + n);
  }
}

std::string Model::describe() const {
  std::ostringstream o;
  int nrec = 0;
  for (auto& L : layers) nrec += L.recurrent;
  o << "arch " << arch << ", " << hp.n_layer << " layers (" << nrec << " gated-delta-net, " << hp.n_layer - nrec
    << " full attention), embd " << hp.n_embd << ", ffn " << hp.n_ff << ", heads " << hp.n_head << "/" << hp.n_head_kv
    << " x " << hp.head_dim << ", rope " << hp.n_rot << " dims base " << hp.rope_base << ", ssm k-heads " << hp.n_group
    << " v-heads " << hp.dt_rank << " state " << hp.d_state << ", vocab " << hp.n_vocab;
  if (hdmd.version)
    o << "\nprism hadamard v" << hdmd.version << ": block " << hdmd.block << ", " << hdmd.weight_names.size()
      << " folded weights, sign mode " << hdmd.sign_mode << ", inverse on token_embd "
      << (embd_inverse.active ? "yes" : "no") << ", gdn_v_grouped " << (hdmd.gdn_v_grouped ? "yes" : "no");
  return o.str();
}

}  // namespace bd
