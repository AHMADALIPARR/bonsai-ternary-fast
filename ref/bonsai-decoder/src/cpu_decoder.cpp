// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// fp32 CPU reference decoder for qwen35. One token per step. Every step below
// cites the prism source it mirrors; docs/graph-conventions.md has the list.
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "bd/cpu_ops.hpp"
#include "bd/decoder.hpp"
#include "bd/types.hpp"

namespace bd {

namespace {

float round_f16(float x) { return fp16_to_f32(f32_to_fp16(x)); }

struct LayerState {
  // full attention: K/V cache [n_ctx][n_head_kv*head_dim]
  std::vector<float> k, v;
  // linear attention: conv state [(K-1)][channels] (oldest first), delta-net
  // state per v head stored s[h][j*dk + i] = S[i][j] (ggml layout)
  std::vector<float> conv, ssm;
  // small weights in fp32
  std::vector<float> attn_norm, post_norm, q_norm, k_norm, ssm_a, ssm_dt, conv_w, ssm_norm;
};

class CpuDecoder final : public Decoder {
 public:
  CpuDecoder(const Model& m, const DecoderOptions& o) : m_(m), hp_(m.hp), o_(o), pool_(o.threads) {
    if (o_.n_ctx <= 0) throw std::runtime_error("n_ctx must be > 0");
    L_.resize(hp_.n_layer);
    for (int il = 0; il < hp_.n_layer; ++il) {
      const LayerWeights& W = m_.layers[il];
      LayerState& S = L_[il];
      S.attn_norm = to_f32(W.attn_norm);
      S.post_norm = to_f32(W.post_norm);
      if (W.recurrent) {
        S.ssm_a = to_f32(W.ssm_a);
        S.ssm_dt = to_f32(W.ssm_dt);
        S.conv_w = to_f32(W.conv1d);
        S.ssm_norm = to_f32(W.ssm_norm);
        S.conv.assign((size_t)(hp_.conv_kernel - 1) * hp_.gdn_conv_ch(), 0.f);
        S.ssm.assign((size_t)hp_.dt_rank * hp_.gdn_head_k() * hp_.gdn_head_v(), 0.f);
      } else {
        S.q_norm = to_f32(W.q_norm);
        S.k_norm = to_f32(W.k_norm);
        const size_t kv = (size_t)hp_.n_head_kv * hp_.head_dim;
        S.k.assign((size_t)o_.n_ctx * kv, 0.f);
        S.v.assign((size_t)o_.n_ctx * kv, 0.f);
      }
    }
    out_norm_ = to_f32(m_.output_norm);
  }

  std::string name() const override { return "cpu-fp32"; }
  int pos() const override { return pos_; }

  void reset() override {
    pos_ = 0;
    for (auto& S : L_) {
      std::fill(S.conv.begin(), S.conv.end(), 0.f);
      std::fill(S.ssm.begin(), S.ssm.end(), 0.f);
    }
  }

  void forward(int token, float* logits) override {
    if (token < 0 || token >= hp_.n_vocab) throw std::runtime_error("token id out of range");
    if (pos_ >= o_.n_ctx) throw std::runtime_error("context full");
    const int E = hp_.n_embd;
    std::vector<float> x(E), xn(E), out(E), h(E);

    // llama-graph.cpp:2676-2692 build_embd_rows: get_rows, then h = s * (H z)
    row_f32(m_.token_embd, token, x.data());
    hadamard_inverse(m_.embd_inverse, x.data(), E);
    rec("model.input_embed", -1, {E}, x.data());

    for (int il = 0; il < hp_.n_layer; ++il) {
      const LayerWeights& W = m_.layers[il];
      LayerState& S = L_[il];
      // qwen35.cpp:206 attn_norm (RMS, weight already holds the +1, see docs)
      rmsnorm(x.data(), S.attn_norm.data(), E, hp_.rms_eps, xn.data());
      rec("attn_norm", il, {E}, xn.data());
      if (W.recurrent)
        linear_attn(il, xn.data(), out.data());
      else
        full_attn(il, xn.data(), out.data());
      for (int i = 0; i < E; ++i) x[i] += out[i];  // qwen35.cpp:226
      rec("attn_residual", il, {E}, x.data());
      rmsnorm(x.data(), S.post_norm.data(), E, hp_.rms_eps, h.data());  // qwen35.cpp:233
      rec("attn_post_norm", il, {E}, h.data());
      ffn(il, h.data(), out.data());
      rec("ffn_out", il, {E}, out.data());
      for (int i = 0; i < E; ++i) x[i] += out[i];  // qwen35.cpp:241
      rec("l_out", il, {E}, x.data());
    }
    rmsnorm(x.data(), out_norm_.data(), E, hp_.rms_eps, xn.data());
    rec("result_norm", -1, {E}, xn.data());
    matvec(pool_, m_.output, xn.data(), logits, o_.ggml_act);  // qwen35.cpp:309, folded head
    rec("result_output", -1, {hp_.n_vocab}, logits);
    ++pos_;
  }

 private:
  void rec(const char* nm, int il, std::vector<int64_t> ne, const float* d) {
    if (!trace_) return;
    std::string n = il >= 0 ? std::string(nm) + "-" + std::to_string(il) : std::string(nm);
    if (trace_->wants(n)) trace_->record(n, pos_, ne, d);
  }

  // llama-graph.cpp build_ffn, LLM_FFN_SILU + LLM_FFN_PAR: down(silu(gate(x)) * up(x))
  void ffn(int il, const float* x, float* y) {
    const LayerWeights& W = m_.layers[il];
    const int F = hp_.n_ff;
    std::vector<float> g(F), u(F);
    matvec(pool_, W.ffn_gate, x, g.data(), o_.ggml_act);
    rec("ffn_gate", il, {F}, g.data());
    matvec(pool_, W.ffn_up, x, u.data(), o_.ggml_act);
    rec("ffn_up", il, {F}, u.data());
    for (int i = 0; i < F; ++i) g[i] = silu(g[i]) * u[i];
    rec("ffn_swiglu", il, {F}, g.data());
    matvec(pool_, W.ffn_down, g.data(), y, o_.ggml_act);
  }

  // qwen35.cpp:343-422
  void full_attn(int il, const float* x, float* y) {
    const LayerWeights& W = m_.layers[il];
    LayerState& S = L_[il];
    const int D = hp_.head_dim, H = hp_.n_head, HK = hp_.n_head_kv, R = hp_.n_rot;
    std::vector<float> qf((size_t)2 * D * H), k((size_t)D * HK), v((size_t)D * HK);
    matvec(pool_, W.wq, x, qf.data(), o_.ggml_act);
    matvec(pool_, W.wk, x, k.data(), o_.ggml_act);
    matvec(pool_, W.wv, x, v.data(), o_.ggml_act);
    rec("Qcur_full", il, {2 * D * H}, qf.data());
    rec("Vcur", il, {D * HK}, v.data());
    rec("Kcur", il, {D * HK}, k.data());
    // per head: q = Qfull[h*2D + 0..D), gate = Qfull[h*2D + D..2D)  (view_3d, nb1 = 2D)
    std::vector<float> q((size_t)D * H), gate((size_t)D * H);
    for (int hh = 0; hh < H; ++hh) {
      std::memcpy(&q[(size_t)hh * D], &qf[(size_t)hh * 2 * D], D * sizeof(float));
      std::memcpy(&gate[(size_t)hh * D], &qf[(size_t)hh * 2 * D + D], D * sizeof(float));
    }
    for (int hh = 0; hh < H; ++hh) rmsnorm(&q[(size_t)hh * D], S.q_norm.data(), D, hp_.rms_eps, &q[(size_t)hh * D]);
    rec("Qcur_normed", il, {D, H}, q.data());
    for (int hh = 0; hh < HK; ++hh) rmsnorm(&k[(size_t)hh * D], S.k_norm.data(), D, hp_.rms_eps, &k[(size_t)hh * D]);
    rec("Kcur_normed", il, {D, HK}, k.data());
    rec("gate_reshaped", il, {D * H}, gate.data());
    // IMROPE (rope type 40) with t=h=w=pos: NEOX pairs (i, i+R/2) on the first
    // R dims, theta_i = pos * base^(-2i/R) accumulated as in ggml_mrope_cache_init
    // (ggml-cpu/ops.cpp:6338-6408, rotate_pairs 6410, dispatch 6557-6559).
    std::vector<float> cs(R);
    {
      const float theta_scale = std::pow(hp_.rope_base, -2.0f / R);
      float theta = (float)pos_;
      for (int i0 = 0; i0 < R; i0 += 2) {
        cs[i0] = std::cos(theta);
        cs[i0 + 1] = std::sin(theta);
        theta *= theta_scale;
      }
    }
    auto rope = [&](float* hv) {
      for (int i0 = 0; i0 < R; i0 += 2) {
        const int ic = i0 / 2;
        const float x0 = hv[ic], x1 = hv[ic + R / 2];
        hv[ic] = x0 * cs[i0] - x1 * cs[i0 + 1];
        hv[ic + R / 2] = x0 * cs[i0 + 1] + x1 * cs[i0];
      }
    };
    for (int hh = 0; hh < H; ++hh) rope(&q[(size_t)hh * D]);
    for (int hh = 0; hh < HK; ++hh) rope(&k[(size_t)hh * D]);
    rec("Qcur", il, {D, H}, q.data());
    rec("k_cache_in", il, {D, HK}, k.data());
    const size_t kvw = (size_t)D * HK;
    float* kc = &S.k[(size_t)pos_ * kvw];
    float* vc = &S.v[(size_t)pos_ * kvw];
    for (size_t i = 0; i < kvw; ++i) {
      kc[i] = o_.kv_f16 ? round_f16(k[i]) : k[i];
      vc[i] = o_.kv_f16 ? round_f16(v[i]) : v[i];
    }
    // ggml-cpu converts Q to the f16 vec_dot type of an F16 K cache
    if (o_.ggml_act && o_.kv_f16)
      for (float& qv : q) qv = round_f16(qv);
    // softmax(q k^T * 1/sqrt(D)) v, causal, grouped GQA: q head h -> kv head h / (H/HK)
    const float scale = 1.0f / std::sqrt((float)D);
    const int T = pos_ + 1, grp = H / HK;
    std::vector<float> att((size_t)D * H);
    pool_.parallel_for(H, [&](int64_t b, int64_t e) {
      std::vector<float> s(T);
      for (int64_t hh = b; hh < e; ++hh) {
        const int g = (int)hh / grp;
        const float* qh = &q[(size_t)hh * D];
        float mx = -INFINITY;
        for (int t = 0; t < T; ++t) {
          const float* kt = &S.k[(size_t)t * kvw + (size_t)g * D];
          double d = 0;
          for (int i = 0; i < D; ++i) d += (double)qh[i] * kt[i];
          s[t] = (float)d * scale;
          mx = std::fmax(mx, s[t]);
        }
        double sum = 0;
        for (int t = 0; t < T; ++t) {
          s[t] = std::exp(s[t] - mx);
          sum += s[t];
        }
        float* o = &att[(size_t)hh * D];
        for (int i = 0; i < D; ++i) o[i] = 0.f;
        for (int t = 0; t < T; ++t) {
          const float p = (float)(s[t] / sum);
          const float* vt = &S.v[(size_t)t * kvw + (size_t)g * D];
          for (int i = 0; i < D; ++i) o[i] += p * vt[i];
        }
      }
    });
    rec("attn_pregate", il, {D * H}, att.data());
    for (size_t i = 0; i < att.size(); ++i) gate[i] = sigmoid(gate[i]);
    rec("gate_sigmoid", il, {D * H}, gate.data());
    for (size_t i = 0; i < att.size(); ++i) att[i] *= gate[i];
    rec("attn_gated", il, {D * H}, att.data());
    matvec(pool_, W.wo, att.data(), y, o_.ggml_act);
    rec("attn_output", il, {hp_.n_embd}, y);
  }

  // qwen35.cpp:424-617, delta-net-base.cpp, ggml-cpu/ops.cpp gated_delta_net
  void linear_attn(int il, const float* x, float* y) {
    const LayerWeights& W = m_.layers[il];
    LayerState& S = L_[il];
    const int C = hp_.gdn_conv_ch(), DK = hp_.gdn_head_k(), DV = hp_.gdn_head_v();
    const int NK = hp_.n_group, NV = hp_.dt_rank, K = hp_.conv_kernel, QK = hp_.gdn_qk_dim();
    std::vector<float> qkv(C), z((size_t)NV * DV), beta(NV), alpha(NV);
    matvec(pool_, W.wqkv, x, qkv.data(), o_.ggml_act);
    rec("linear_attn_qkv_mixed", il, {C}, qkv.data());
    matvec(pool_, W.wgate, x, z.data(), o_.ggml_act);
    rec("z", il, {NV * DV}, z.data());
    matvec(pool_, W.ssm_beta, x, beta.data(), o_.ggml_act);
    rec("beta", il, {1, NV}, beta.data());
    for (auto& b : beta) b = sigmoid(b);
    rec("beta_sigmoid", il, {1, NV}, beta.data());
    matvec(pool_, W.ssm_alpha, x, alpha.data(), o_.ggml_act);
    rec("alpha", il, {NV}, alpha.data());
    std::vector<float> g(NV);
    for (int h = 0; h < NV; ++h) g[h] = softplus(alpha[h] + S.ssm_dt[h]);
    rec("a_softplus", il, {NV}, g.data());
    for (int h = 0; h < NV; ++h) g[h] *= S.ssm_a[h];  // ssm_a = -exp(A_log) (converter)
    rec("gate", il, {NV}, g.data());

    // ssm_conv (ops.cpp:10233-10290): out[c] = sum_i xpad[c, i] * w[c*K + i],
    // xpad = [K-1 previous inputs, current]; no bias; state keeps last K-1 inputs
    std::vector<float> conv(C);
    for (int c = 0; c < C; ++c) {
      float s = 0.f;
      for (int i = 0; i < K - 1; ++i) s += S.conv[(size_t)i * C + c] * S.conv_w[(size_t)c * K + i];
      s += qkv[c] * S.conv_w[(size_t)c * K + K - 1];
      conv[c] = s;
    }
    for (int i = 0; i + 1 < K - 1; ++i) std::memcpy(&S.conv[(size_t)i * C], &S.conv[(size_t)(i + 1) * C], C * sizeof(float));
    if (K > 1) std::memcpy(&S.conv[(size_t)(K - 2) * C], qkv.data(), C * sizeof(float));
    rec("conv_output_raw", il, {C}, conv.data());
    for (auto& c : conv) c = silu(c);
    rec("conv_output_silu", il, {C}, conv.data());

    // q = conv[0, QK), k = conv[QK, 2QK), v = conv[2QK, C); l2 norm per head (eps = rms eps)
    std::vector<float> qk(2 * QK);
    for (int hh = 0; hh < 2 * NK; ++hh) l2norm(&conv[(size_t)hh * DK], DK, hp_.rms_eps, &qk[(size_t)hh * DK]);
    rec("qk_conv_l2", il, {DK, 2 * NK}, qk.data());
    const float* q = qk.data();
    const float* k = qk.data() + QK;
    const float* v = conv.data() + 2 * QK;
    rec("q_conv_predelta", il, {DK, NK}, q);
    rec("k_conv_predelta", il, {DK, NK}, k);
    rec("v_conv_predelta", il, {DV, NV}, v);

    // gated delta rule, one step, per v head h with k/q head h % NK (tiled)
    std::vector<float> o((size_t)NV * DV);
    const float qscale = 1.0f / std::sqrt((float)DK);
    pool_.parallel_for(NV, [&](int64_t b, int64_t e) {
      std::vector<float> delta(DV);
      for (int64_t h = b; h < e; ++h) {
        const int kh = (int)h % NK;
        const float* qh = q + (size_t)kh * DK;
        const float* kk = k + (size_t)kh * DK;
        const float* vh = v + (size_t)h * DV;
        float* st = &S.ssm[(size_t)h * DK * DV];  // st[j*DK + i] = S[i][j]
        const float decay = std::exp(g[h]);
        for (int j = 0; j < DV; ++j) {
          float* col = st + (size_t)j * DK;
          float kv = 0.f;
          for (int i = 0; i < DK; ++i) {
            col[i] *= decay;
            kv += col[i] * kk[i];
          }
          delta[j] = (vh[j] - kv) * beta[h];
        }
        float* oh = &o[(size_t)h * DV];
        for (int j = 0; j < DV; ++j) {
          float* col = st + (size_t)j * DK;
          float acc = 0.f;
          for (int i = 0; i < DK; ++i) {
            col[i] += kk[i] * delta[j];
            acc += col[i] * qh[i];
          }
          oh[j] = acc * qscale;
        }
      }
    });
    rec("attn_output", il, {DV, NV}, o.data());
    // build_norm_gated (qwen35.cpp:332-341): rms(o per head) * ssm_norm * silu(z)
    for (int h = 0; h < NV; ++h) {
      float* oh = &o[(size_t)h * DV];
      rmsnorm(oh, S.ssm_norm.data(), DV, hp_.rms_eps, oh);
      for (int j = 0; j < DV; ++j) oh[j] *= silu(z[(size_t)h * DV + j]);
    }
    rec("final_output", il, {NV * DV}, o.data());
    matvec(pool_, W.ssm_out, o.data(), y, o_.ggml_act);  // gdn v permutation happens inside (folded)
    rec("linear_attn_out", il, {hp_.n_embd}, y);
  }

  const Model& m_;
  const HParams& hp_;
  DecoderOptions o_;
  ThreadPool pool_;
  std::vector<LayerState> L_;
  std::vector<float> out_norm_;
  int pos_ = 0;
};

}  // namespace

std::unique_ptr<Decoder> make_cpu_decoder(const Model& m, const DecoderOptions& o) {
  return std::unique_ptr<Decoder>(new CpuDecoder(m, o));
}

}  // namespace bd
