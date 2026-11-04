// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CPU emulator of the CUDA decode path. Same packed T2 weights, same fused
// prep (RMSNorm -> perm -> signs -> WHT -> Q8), same integer dot (codes . q8
// minus sum(q8)), same batched prompt order (store K/V of the whole chunk,
// then causal attention per row). It exists so the algorithm can be checked
// against the bonsai-decoder reference on a machine without a GPU.
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "bd/cpu_ops.hpp"
#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "btf/packed.hpp"
#include "dims.hpp"

namespace btf {

namespace {

float round_f16(float x) { return bd::fp16_to_f32(bd::f32_to_fp16(x)); }
float round_bf16(float x) { return bd::bf16_to_f32(bd::f32_to_bf16(x)); }

struct Q8 {
  std::vector<int8_t> q;
  std::vector<float> d;
  std::vector<int> s;
  int K = 0;
  void resize(int M, int k) {
    K = k;
    q.resize((size_t)M * k);
    d.resize((size_t)M * k / 32);
    s.resize((size_t)M * k / 32);
  }
};

struct ELayer {
  PackedMatrix gate, up, down;
  PackedMatrix qkv, zg, out;     // GDN
  PackedMatrix wq, wk, wv, wo;   // attention
  std::vector<float> attn_norm, post_norm, q_norm, k_norm, ssm_a, ssm_dt, conv_w, ssm_norm, alpha_w, beta_w;
  bool ab_bf16 = false;
  std::vector<uint16_t> kc, vc;  // fp16 KV cache
  std::vector<float> conv, ssm;
};

class EmuDecoder final : public FastDecoder {
 public:
  EmuDecoder(const bd::Model& m, const FastOptions& o) : m_(m), o_(o), d_(m, o), pool_(o.threads) {
    check_supported(m, o);
    L_.resize(d_.L);
    for (int il = 0; il < d_.L; ++il) {
      const bd::LayerWeights& W = m.layers[il];
      ELayer& S = L_[il];
      S.gate = repack(W.ffn_gate, &pool_);
      S.up = repack(W.ffn_up, &pool_);
      S.down = repack(W.ffn_down, &pool_);
      S.attn_norm = bd::to_f32(W.attn_norm);
      S.post_norm = bd::to_f32(W.post_norm);
      if (W.recurrent) {
        S.qkv = repack(W.wqkv, &pool_);
        S.zg = repack(W.wgate, &pool_);
        S.out = repack(W.ssm_out, &pool_);
        S.ssm_a = bd::to_f32(W.ssm_a);
        S.ssm_dt = bd::to_f32(W.ssm_dt);
        S.conv_w = bd::to_f32(W.conv1d);
        S.ssm_norm = bd::to_f32(W.ssm_norm);
        S.alpha_w = bd::to_f32(W.ssm_alpha);
        S.beta_w = bd::to_f32(W.ssm_beta);
        S.ab_bf16 = W.ssm_alpha.t->type == bd::T_BF16;
        S.conv.assign((size_t)(d_.KC - 1) * d_.C, 0.f);
        S.ssm.assign((size_t)d_.NV * d_.DK * d_.DV, 0.f);
      } else {
        S.wq = repack(W.wq, &pool_);
        S.wk = repack(W.wk, &pool_);
        S.wv = repack(W.wv, &pool_);
        S.wo = repack(W.wo, &pool_);
        S.q_norm = bd::to_f32(W.q_norm);
        S.k_norm = bd::to_f32(W.k_norm);
        S.kc.assign((size_t)d_.n_ctx * d_.HK * d_.D, 0);
        S.vc.assign((size_t)d_.n_ctx * d_.HK * d_.D, 0);
      }
    }
    embd_ = repack(m.token_embd, &pool_);
    output_ = repack(m.output, &pool_);
    out_norm_ = bd::to_f32(m.output_norm);
  }

  std::string name() const override { return "btf-cpu-emulator"; }
  int pos() const override { return pos_; }
  void reset() override {
    pos_ = 0;
    for (auto& S : L_) {
      std::fill(S.conv.begin(), S.conv.end(), 0.f);
      std::fill(S.ssm.begin(), S.ssm.end(), 0.f);
    }
  }

  int step(int32_t token, float* logits) override {
    const auto t0 = std::chrono::steady_clock::now();
    const int r = chunk(&token, 1, logits);
    stats_.steps += 1;
    stats_.wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
  }
  StepStats stats() const override { return stats_; }
  void reset_stats() override { stats_ = StepStats{}; }

  int prefill(const int32_t* toks, int n, float* logits) override {
    int next = -1;
    for (int i = 0; i < n; i += d_.maxM) {
      const int M = std::min(d_.maxM, n - i);
      next = chunk(toks + i, M, i + M == n ? logits : nullptr);
    }
    return next;
  }

 private:
  // ---- fused prep: [rmsnorm * w] -> perm -> signs -> blockwise WHT -> Q8_0
  void prep(const float* x, int M, int n, const float* normw, const bd::HadamardXform& hx, Q8& a,
            float* xn_out = nullptr) {
    a.resize(M, n);
    std::vector<float> xn(n), t(n);
    for (int m = 0; m < M; ++m) {
      const float* xr = x + (size_t)m * n;
      if (normw) {
        bd::rmsnorm(xr, normw, n, d_.eps, xn.data());
      } else {
        std::memcpy(xn.data(), xr, n * sizeof(float));
      }
      if (xn_out) std::memcpy(xn_out + (size_t)m * n, xn.data(), n * sizeof(float));
      bd::hadamard_input(hx, xn.data(), n, t.data());
      for (int b = 0; b < n / 32; ++b) {
        const size_t o = (size_t)m * (n / 32) + b;
        a.s[o] = q8_block_host(&t[b * 32], &a.q[(size_t)m * n + b * 32], &a.d[o]);
      }
    }
  }

  // ---- ternary GEMV/GEMM on packed codes; add: y += W x
  void gemv(const PackedMatrix& P, const Q8& a, int M, float* y, int ldy, bool add) {
    if (a.K != P.K) throw std::runtime_error("emu gemv: K mismatch");
    const int64_t nsub = P.K / 32;
    pool_.parallel_for(P.N, [&](int64_t b, int64_t e) {
      for (int64_t r = b; r < e; ++r) {
        const uint8_t* cr = P.codes.data() + r * (P.K / 4);
        const uint16_t* sr = P.scales.data() + r * (P.K / 128);
        for (int m = 0; m < M; ++m) {
          const uint32_t* aw = (const uint32_t*)(a.q.data() + (size_t)m * P.K);
          float acc = 0.f;
          for (int64_t s = 0; s < nsub; ++s) {
            uint32_t w0, w1;
            std::memcpy(&w0, cr + s * 8, 4);
            std::memcpy(&w1, cr + s * 8 + 4, 4);
            const int idot = sub_dot_host(w0, w1, aw + s * 8) - a.s[(size_t)m * nsub + s];
            acc += bd::fp16_to_f32(sr[s / 4]) * a.d[(size_t)m * nsub + s] * (float)idot;
          }
          float& out = y[(size_t)m * ldy + r];
          out = add ? out + acc : acc;
        }
      }
    });
  }

  int chunk(const int32_t* toks, int M, float* logits) {
    if (pos_ + M > d_.n_ctx) throw std::runtime_error("context full");
    const int E = d_.E;
    std::vector<float> x((size_t)M * E);
    // embedding: decode packed row, inverse transform (WHT then signs)
    for (int m = 0; m < M; ++m) {
      const int tok = toks[m];
      if (tok < 0 || tok >= d_.V) throw std::runtime_error("token id out of range");
      float* xr = &x[(size_t)m * E];
      const uint8_t* cr = embd_.codes.data() + (size_t)tok * (E / 4);
      const uint16_t* sr = embd_.scales.data() + (size_t)tok * (E / 128);
      for (int i = 0; i < E; ++i) xr[i] = (float)(packed_code(cr, i) - 1) * bd::fp16_to_f32(sr[i / 128]);
      bd::hadamard_inverse(m_.embd_inverse, xr, E);
    }
    Q8 a;
    std::vector<float> xn((size_t)M * E);
    for (int il = 0; il < d_.L; ++il) {
      ELayer& S = L_[il];
      const bool rec = m_.layers[il].recurrent;
      if (rec) {
        prep(x.data(), M, E, S.attn_norm.data(), S.qkv.hx, a, xn.data());
        gdn(S, a, xn.data(), M, x.data());
      } else {
        prep(x.data(), M, E, S.attn_norm.data(), S.wq.hx, a);
        attn(S, a, M, x.data());
      }
      prep(x.data(), M, E, S.post_norm.data(), S.gate.hx, a);
      std::vector<float> g((size_t)M * d_.F), u((size_t)M * d_.F);
      gemv(S.gate, a, M, g.data(), d_.F, false);
      gemv(S.up, a, M, u.data(), d_.F, false);
      for (size_t i = 0; i < g.size(); ++i) g[i] = bd::silu(g[i]) * u[i];
      prep(g.data(), M, d_.F, nullptr, S.down.hx, a);
      gemv(S.down, a, M, x.data(), E, true);
    }
    prep(&x[(size_t)(M - 1) * E], 1, E, out_norm_.data(), output_.hx, a);
    std::vector<float> lg(d_.V);
    gemv(output_, a, 1, lg.data(), d_.V, false);
    int best = 0;
    for (int i = 1; i < d_.V; ++i)
      if (lg[i] > lg[best]) best = i;
    if (logits) std::memcpy(logits, lg.data(), lg.size() * sizeof(float));
    pos_ += M;
    return best;
  }

  void gdn(ELayer& S, const Q8& a, const float* xn, int M, float* x) {
    const int C = d_.C, DK = d_.DK, DV = d_.DV, NK = d_.NK, NV = d_.NV, QK = d_.QK, DI = d_.DI, E = d_.E;
    std::vector<float> qkv((size_t)M * C), z((size_t)M * DI), beta((size_t)M * NV), g((size_t)M * NV);
    gemv(S.qkv, a, M, qkv.data(), C, false);
    gemv(S.zg, a, M, z.data(), DI, false);
    // alpha / beta: plain dot on (bf16 rounded) xn
    for (int m = 0; m < M; ++m) {
      std::vector<float> xr(xn + (size_t)m * E, xn + (size_t)(m + 1) * E);
      if (S.ab_bf16)
        for (float& v : xr) v = round_bf16(v);
      for (int h = 0; h < NV; ++h) {
        float sa = 0.f, sb = 0.f;
        for (int i = 0; i < E; ++i) {
          sa += S.alpha_w[(size_t)h * E + i] * xr[i];
          sb += S.beta_w[(size_t)h * E + i] * xr[i];
        }
        beta[(size_t)m * NV + h] = bd::sigmoid(sb);
        g[(size_t)m * NV + h] = bd::softplus(sa + S.ssm_dt[h]) * S.ssm_a[h];
      }
    }
    // conv (K = 4) + silu, sequential over the chunk
    std::vector<float> cv((size_t)M * C);
    const int K = d_.KC;
    for (int c = 0; c < C; ++c) {
      for (int m = 0; m < M; ++m) {
        float s = 0.f;
        for (int i = 0; i < K - 1; ++i) s += S.conv[(size_t)i * C + c] * S.conv_w[(size_t)c * K + i];
        const float xin = qkv[(size_t)m * C + c];
        s += xin * S.conv_w[(size_t)c * K + K - 1];
        for (int i = 0; i + 1 < K - 1; ++i) S.conv[(size_t)i * C + c] = S.conv[(size_t)(i + 1) * C + c];
        S.conv[(size_t)(K - 2) * C + c] = xin;
        cv[(size_t)m * C + c] = bd::silu(s);
      }
    }
    // delta rule + gated norm
    std::vector<float> o((size_t)M * DI);
    const float qscale = 1.0f / std::sqrt((float)DK);
    pool_.parallel_for(NV, [&](int64_t hb, int64_t he) {
      std::vector<float> q(DK), k(DK), delta(DV);
      for (int64_t h = hb; h < he; ++h) {
        const int kh = (int)h % NK;
        float* st = &S.ssm[(size_t)h * DK * DV];
        for (int m = 0; m < M; ++m) {
          const float* cr = &cv[(size_t)m * C];
          bd::l2norm(cr + (size_t)kh * DK, DK, d_.eps, q.data());
          bd::l2norm(cr + QK + (size_t)kh * DK, DK, d_.eps, k.data());
          const float* v = cr + 2 * QK + (size_t)h * DV;
          const float decay = std::exp(g[(size_t)m * NV + h]);
          const float bt = beta[(size_t)m * NV + h];
          float* oh = &o[(size_t)m * DI + (size_t)h * DV];
          for (int j = 0; j < DV; ++j) {
            float* col = st + (size_t)j * DK;
            float kv = 0.f;
            for (int i = 0; i < DK; ++i) {
              col[i] *= decay;
              kv += col[i] * k[i];
            }
            const float dl = (v[j] - kv) * bt;
            float acc = 0.f;
            for (int i = 0; i < DK; ++i) {
              col[i] += k[i] * dl;
              acc += col[i] * q[i];
            }
            oh[j] = acc * qscale;
          }
          bd::rmsnorm(oh, S.ssm_norm.data(), DV, d_.eps, oh);
          for (int j = 0; j < DV; ++j) oh[j] *= bd::silu(z[(size_t)m * DI + (size_t)h * DV + j]);
        }
      }
    });
    Q8 b;
    prep(o.data(), M, DI, nullptr, S.out.hx, b);
    gemv(S.out, b, M, x, E, true);
  }

  void attn(ELayer& S, const Q8& a, int M, float* x) {
    const int D = d_.D, H = d_.H, HK = d_.HK, R = d_.R, E = d_.E;
    const int kvw = HK * D;
    std::vector<float> qf((size_t)M * 2 * H * D), k((size_t)M * kvw), v((size_t)M * kvw);
    gemv(S.wq, a, M, qf.data(), 2 * H * D, false);
    gemv(S.wk, a, M, k.data(), kvw, false);
    gemv(S.wv, a, M, v.data(), kvw, false);
    const float theta_scale = std::pow(d_.rope_base, -2.0f / R);
    std::vector<float> q((size_t)M * H * D);
    auto rope = [&](float* hv, int p) {
      float theta = (float)p;
      for (int ic = 0; ic < R / 2; ++ic) {
        const float c = std::cos(theta), s = std::sin(theta);
        const float x0 = hv[ic], x1 = hv[ic + R / 2];
        hv[ic] = x0 * c - x1 * s;
        hv[ic + R / 2] = x0 * s + x1 * c;
        theta *= theta_scale;
      }
    };
    for (int m = 0; m < M; ++m) {
      const int p = pos_ + m;
      for (int h = 0; h < H; ++h) {
        float* qh = &q[((size_t)m * H + h) * D];
        bd::rmsnorm(&qf[(size_t)m * 2 * H * D + (size_t)h * 2 * D], S.q_norm.data(), D, d_.eps, qh);
        rope(qh, p);
        if (o_.round_q_f16)
          for (int i = 0; i < D; ++i) qh[i] = round_f16(qh[i]);
      }
      for (int h = 0; h < HK; ++h) {
        float* kh = &k[(size_t)m * kvw + (size_t)h * D];
        bd::rmsnorm(kh, S.k_norm.data(), D, d_.eps, kh);
        rope(kh, p);
      }
      for (int i = 0; i < kvw; ++i) {
        S.kc[(size_t)p * kvw + i] = bd::f32_to_fp16(k[(size_t)m * kvw + i]);
        S.vc[(size_t)p * kvw + i] = bd::f32_to_fp16(v[(size_t)m * kvw + i]);
      }
    }
    std::vector<float> o((size_t)M * H * D);
    const float scale = 1.0f / std::sqrt((float)D);
    const int grp = H / HK;
    for (int m = 0; m < M; ++m) {
      const int T = pos_ + m + 1;
      pool_.parallel_for(H, [&](int64_t hb, int64_t he) {
        std::vector<float> s(T);
        for (int64_t h = hb; h < he; ++h) {
          const int g = (int)h / grp;
          const float* qh = &q[((size_t)m * H + h) * D];
          float mx = -INFINITY;
          for (int t = 0; t < T; ++t) {
            const uint16_t* kt = &S.kc[(size_t)t * kvw + (size_t)g * D];
            float dd = 0.f;
            for (int i = 0; i < D; ++i) dd += qh[i] * bd::fp16_to_f32(kt[i]);
            s[t] = dd * scale;
            mx = std::fmax(mx, s[t]);
          }
          float sum = 0.f;
          for (int t = 0; t < T; ++t) {
            s[t] = std::exp(s[t] - mx);
            sum += s[t];
          }
          float* oh = &o[((size_t)m * H + h) * D];
          for (int i = 0; i < D; ++i) oh[i] = 0.f;
          for (int t = 0; t < T; ++t) {
            const uint16_t* vt = &S.vc[(size_t)t * kvw + (size_t)g * D];
            for (int i = 0; i < D; ++i) oh[i] += s[t] * bd::fp16_to_f32(vt[i]);
          }
          const float* gate = &qf[(size_t)m * 2 * H * D + (size_t)h * 2 * D + D];
          for (int i = 0; i < D; ++i) oh[i] = oh[i] / sum * bd::sigmoid(gate[i]);
        }
      });
    }
    Q8 b;
    prep(o.data(), M, H * D, nullptr, S.wo.hx, b);
    gemv(S.wo, b, M, x, E, true);
  }

  const bd::Model& m_;
  FastOptions o_;
  Dims d_;
  bd::ThreadPool pool_;
  std::vector<ELayer> L_;
  PackedMatrix embd_, output_;
  std::vector<float> out_norm_;
  int pos_ = 0;
  StepStats stats_;
};

}  // namespace

std::unique_ptr<FastDecoder> make_emu_decoder(const bd::Model& m, const FastOptions& o) {
  return std::unique_ptr<FastDecoder>(new EmuDecoder(m, o));
}

}  // namespace btf
