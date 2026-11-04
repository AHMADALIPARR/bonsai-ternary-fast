// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
#include "bd/synth.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

#include "bd/gguf.hpp"
#include "bd/tokenizer.hpp"
#include "bd/types.hpp"

namespace bd {

namespace {

const char* kMerges[][2] = {
    {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {"Ġ", "t"}, {"Ġ", "a"}, {"Ġ", "h"}, {"Ġh", "i"},
    {"i", "n"}, {"Ġt", "he"}, {"e", "r"}, {"o", "n"}, {"Ġ", "w"}, {"o", "r"}, {"Ġw", "or"}, {"Ġwor", "l"},
    {"Ġworl", "d"}, {"Ċ", "Ċ"}, {"Ġ", "Ġ"}, {"r", "e"},
};
const char* kSpecials[] = {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"};

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t s) : g(s) {}
  float uni(float a, float b) { return std::uniform_real_distribution<float>(a, b)(g); }
  float nrm(float s) { return std::normal_distribution<float>(0.f, s)(g); }
  int tern() { return (int)(g() % 3) - 1; }
};

std::vector<uint8_t> encode(uint32_t type, const std::vector<float>& v) {
  const TypeInfo ti = type_info(type);
  std::vector<uint8_t> out(row_bytes(type, (int64_t)v.size()));
  quant_row(type, v.data(), out.data(), (int64_t)v.size());
  (void)ti;
  return out;
}

}  // namespace

std::vector<std::string> synth_vocab_tokens() {
  std::vector<std::string> t;
  for (int b = 0; b < 256; ++b) t.push_back(gpt2_byte_to_unicode((uint8_t)b));
  for (auto& m : kMerges) t.push_back(std::string(m[0]) + m[1]);
  for (auto* s : kSpecials) t.push_back(s);
  return t;
}

std::vector<uint8_t> make_synth_qwen35(const SynthOptions& o) {
  Rng r(o.seed);
  GgufWriter w;
  const int E = o.n_embd, F = o.n_ff, H = o.n_head, HK = o.n_head_kv, D = o.head_dim;
  const int d_inner = o.dt_rank * o.d_state;  // head_v == head_k == d_state
  const int conv_ch = 2 * o.d_state * o.n_group + d_inner;
  const std::vector<std::string> vocab = synth_vocab_tokens();
  const int V = (int)vocab.size();

  w.kv_str("general.architecture", "qwen35");
  w.kv_str("general.name", "bonsai-decoder synthetic qwen35");
  w.kv_u32("qwen35.block_count", (uint32_t)o.n_layer);
  w.kv_u32("qwen35.context_length", 4096);
  w.kv_u32("qwen35.embedding_length", (uint32_t)E);
  w.kv_u32("qwen35.feed_forward_length", (uint32_t)F);
  w.kv_u32("qwen35.attention.head_count", (uint32_t)H);
  w.kv_u32("qwen35.attention.head_count_kv", (uint32_t)HK);
  w.kv_u32("qwen35.attention.key_length", (uint32_t)D);
  w.kv_u32("qwen35.attention.value_length", (uint32_t)D);
  w.kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
  w.kv_u32("qwen35.rope.dimension_count", (uint32_t)o.n_rot);
  w.kv_arr_i32("qwen35.rope.dimension_sections", {3, 3, 2, 0});
  w.kv_f32("qwen35.rope.freq_base", 10000000.0f);
  w.kv_u32("qwen35.ssm.conv_kernel", (uint32_t)o.conv_kernel);
  w.kv_u32("qwen35.ssm.state_size", (uint32_t)o.d_state);
  w.kv_u32("qwen35.ssm.group_count", (uint32_t)o.n_group);
  w.kv_u32("qwen35.ssm.time_step_rank", (uint32_t)o.dt_rank);
  w.kv_u32("qwen35.ssm.inner_size", (uint32_t)d_inner);
  w.kv_u32("qwen35.full_attention_interval", 4);

  // tokenizer
  w.kv_str("tokenizer.ggml.model", "gpt2");
  w.kv_str("tokenizer.ggml.pre", "qwen35");
  w.kv_arr_str("tokenizer.ggml.tokens", vocab);
  std::vector<int32_t> types(V, 1);
  for (size_t i = 0; i < sizeof(kSpecials) / sizeof(kSpecials[0]); ++i) types[V - 1 - i] = 3;
  w.kv_arr_i32("tokenizer.ggml.token_type", types);
  std::vector<std::string> merges;
  for (auto& m : kMerges) merges.push_back(std::string(m[0]) + " " + m[1]);
  w.kv_arr_str("tokenizer.ggml.merges", merges);
  const int id_eot = 256 + (int)(sizeof(kMerges) / sizeof(kMerges[0]));
  w.kv_u32("tokenizer.ggml.eos_token_id", (uint32_t)(id_eot + 2));  // <|im_end|>
  w.kv_u32("tokenizer.ggml.bos_token_id", (uint32_t)id_eot);
  w.kv_u32("tokenizer.ggml.padding_token_id", (uint32_t)id_eot);
  w.kv_bool("tokenizer.ggml.add_bos_token", false);

  std::vector<std::string> folded;
  auto mat = [&](const std::string& name, int in, int out, uint32_t type, bool fold, float scale) {
    std::vector<float> v((size_t)in * out);
    const bool ternary = type == T_PTQ1_0 || type == T_PQ2_0 || type == T_Q2_0;
    for (int ro = 0; ro < out; ++ro) {
      const float s = scale / std::sqrt((float)in) * r.uni(0.6f, 1.4f);
      for (int i = 0; i < in; ++i) v[(size_t)ro * in + i] = ternary ? s * (float)r.tern() : r.nrm(s);
    }
    w.tensor(name, {in, out}, type, encode(type, v));
    if (fold) folded.push_back(name);
  };
  auto vecf = [&](const std::string& name, std::vector<int64_t> ne, std::vector<float> v) {
    w.tensor(name, ne, T_F32, encode(T_F32, v));
  };
  auto norm_w = [&](int n, float mean) {
    std::vector<float> v(n);
    for (auto& x : v) x = mean + r.uni(-0.3f, 0.3f);
    return v;
  };
  const bool hd = o.hadamard;

  mat("token_embd.weight", E, V, o.embd_type, false, 1.0f * std::sqrt((float)E));
  vecf("output_norm.weight", {E}, norm_w(E, 1.9f));
  if (!o.tied) mat("output.weight", E, V, o.mat_type, hd, 1.0f);
  for (int il = 0; il < o.n_layer; ++il) {
    const std::string b = "blk." + std::to_string(il) + ".";
    const bool rec = ((il + 1) % 4) != 0;
    vecf(b + "attn_norm.weight", {E}, norm_w(E, 1.0f));
    vecf(b + "post_attention_norm.weight", {E}, norm_w(E, 0.8f));
    mat(b + "ffn_gate.weight", E, F, o.mat_type, hd, 1.0f);
    mat(b + "ffn_up.weight", E, F, o.mat_type, hd, 1.0f);
    mat(b + "ffn_down.weight", F, E, o.mat_type, hd, 0.5f);
    if (rec) {
      mat(b + "attn_qkv.weight", E, conv_ch, o.mat_type, hd, 1.0f);
      mat(b + "attn_gate.weight", E, d_inner, o.mat_type, hd, 1.0f);
      mat(b + "ssm_alpha.weight", E, o.dt_rank, o.small_mat_type, false, 1.0f);
      mat(b + "ssm_beta.weight", E, o.dt_rank, o.small_mat_type, false, 1.0f);
      std::vector<float> a(o.dt_rank), dt(o.dt_rank);
      for (auto& x : a) x = -std::exp(r.uni(-5.5f, -1.0f));  // stored as -exp(A_log)
      for (auto& x : dt) x = r.uni(-4.0f, 4.0f);
      vecf(b + "ssm_a", {o.dt_rank}, a);
      vecf(b + "ssm_dt.bias", {o.dt_rank}, dt);
      std::vector<float> cw((size_t)o.conv_kernel * conv_ch);
      for (auto& x : cw) x = r.uni(-0.6f, 0.6f);
      vecf(b + "ssm_conv1d.weight", {o.conv_kernel, conv_ch}, cw);
      vecf(b + "ssm_norm.weight", {o.d_state}, norm_w(o.d_state, 0.85f));
      mat(b + "ssm_out.weight", d_inner, E, o.mat_type, hd, 0.5f);
    } else {
      mat(b + "attn_q.weight", E, 2 * H * D, o.mat_type, hd, 1.0f);
      mat(b + "attn_k.weight", E, HK * D, o.mat_type, hd, 1.0f);
      mat(b + "attn_v.weight", E, HK * D, o.mat_type, hd, 1.0f);
      mat(b + "attn_output.weight", H * D, E, o.mat_type, hd, 0.5f);
      vecf(b + "attn_q_norm.weight", {D}, norm_w(D, 1.2f));
      vecf(b + "attn_k_norm.weight", {D}, norm_w(D, 1.2f));
    }
  }

  if (hd) {
    w.kv_u32("prism.hadamard.version", o.tied ? 2 : 1);
    w.kv_u32("prism.hadamard.block_size", (uint32_t)o.hadamard_block);
    w.kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    w.kv_str("prism.hadamard.axis", "input-last-dimension");
    w.kv_str("prism.hadamard.sign_mode", o.explicit_signs ? "explicit" : "identity");
    w.kv_arr_str("prism.hadamard.weight_names", folded);
    if (o.explicit_signs) {
      std::vector<int32_t> widths, vals;
      for (int width : {E, F, d_inner, H * D}) {
        bool seen = false;
        for (int x : widths) seen |= x == width;
        if (seen) continue;
        widths.push_back(width);
        for (int i = 0; i < width; ++i) vals.push_back((r.g() & 1) ? 1 : -1);
      }
      w.kv_arr_i32("prism.hadamard.sign_widths", widths);
      w.kv_arr_i32("prism.hadamard.sign_values", vals);
    }
    w.kv_arr_str("prism.hadamard.inverse_weight_names", {"token_embd.weight"});
    w.kv_bool("prism.hadamard.gdn_v_grouped", o.gdn_v_grouped);
    w.kv_bool("prism.hadamard.tied_output", o.tied);
  }
  return w.finish(32);
}

}  // namespace bd
