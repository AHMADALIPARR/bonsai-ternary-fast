// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// The CPU emulator of the fast path (packed T2 weights, fused prep, integer
// dot, batched prompt) against the bonsai-decoder CPU reference with
// ggml-style activation rounding and an fp16 KV cache, on synthetic models:
// logits of a 9-token prompt and 24 greedy tokens (teacher-forced on the
// reference's tokens), then the batched prompt path (chunks of 8 and 3)
// against the one-token path.
//
// Tolerances. Both paths do the same arithmetic up to summation order, so
// the first token agrees to ~1e-7 (wider models can already see a flip). Then a Q8 activation that lands within
// float rounding of a .5 boundary can round the other way; the GDN state and
// KV cache then carry that difference forward. The noise scale of that effect
// is the gap between the reference with and without activation rounding (the
// Q8/f16 rounding itself), measured here on the same tokens: the fast path
// must stay within the same order. (With the GEMV summed in the reference's
// float order instead of exact integers, all steps of the default config
// match bit for bit, so the drift is rounding flips, not a semantic error.)
#include <cstdlib>

#include "bd/decoder.hpp"
#include "btf/fast.hpp"
#include "test_common.hpp"

static const double kFirstTol = 2e-3;   // first token: summation order (+ at most a flip)
static const double kNoiseFrac = 2.0;   // fast vs ref  <=  2 x  (ref vs unrounded ref)
static const double kAbsTol = 2e-2;     // and never more than this

int main() {
  for (const auto& cfg : btft::configs()) {
    auto model = btft::make_model(cfg.so);
    bd::DecoderOptions ro;
    ro.n_ctx = 64;
    ro.kv_f16 = true;
    ro.ggml_act = true;
    auto ref = bd::make_cpu_decoder(*model, ro);
    bd::DecoderOptions uo;
    uo.n_ctx = 64;
    auto unr = bd::make_cpu_decoder(*model, uo);  // no activation rounding, f32 KV
    btf::FastOptions fo;
    fo.n_ctx = 64;
    auto emu = btf::make_emu_decoder(*model, fo);
    const int nv = model->hp.n_vocab;
    std::vector<float> lr(nv), le(nv), lu(nv);
    double worst = 0, noise = 0, first = -1;
    int steps = 0;
    auto run = [&](int t) {
      ref->forward(t, lr.data());
      emu->forward(t, le.data());
      unr->forward(t, lu.data());
      const double r = btft::rel_rms(le.data(), lr.data(), nv), n = btft::rel_rms(lu.data(), lr.data(), nv);
      if (std::getenv("BTF_VERBOSE")) std::printf("    step %d fast %.2e noise %.2e\n", steps, r, n);
      if (first < 0) first = r;
      worst = std::max(worst, r);
      noise += n;
      ++steps;
    };
    std::vector<int> seq = btft::prompt();
    for (int t : seq) run(t);
    const std::vector<float> last_prompt_logits = le;
    int tr = btft::argmax(lr), same = 0;
    std::vector<int> gen;
    for (int i = 0; i < 24; ++i) {
      if (btft::argmax(le) == tr) ++same;
      gen.push_back(tr);
      run(tr);
      tr = btft::argmax(lr);
    }
    noise /= steps;
    std::printf("%-20s fast vs ref: first %.1e, worst %.2e | rounding noise (mean) %.2e | greedy %d/24\n",
                cfg.label, first, worst, noise, same);
    BD_CHECK(first <= kFirstTol);
    BD_CHECK(worst <= kNoiseFrac * noise || worst <= 1e-5);
    BD_CHECK(worst <= kAbsTol);
    BD_CHECK(same >= 23);

    // batched prompt == one token at a time (same integer math per row)
    for (int mb : {8, 3}) {
      btf::FastOptions fb = fo;
      fb.max_batch = mb;
      auto emb = btf::make_emu_decoder(*model, fb);
      std::vector<float> lb(nv);
      const int next = emb->prefill(seq.data(), (int)seq.size(), lb.data());
      const double d = btft::rel_rms(lb.data(), last_prompt_logits.data(), nv);
      BD_CHECK(next == btft::argmax(lb));
      BD_CHECK(d <= 1e-6);
      // continue decoding the same tokens
      emb->step(gen[0], lb.data());
      BD_CHECK(emb->pos() == (int)seq.size() + 1);
      std::printf("%-20s   prefill batch %d: last-prompt logits rel rms %.2e vs per-token\n", "", mb, d);
    }
  }
  return 0;
}
