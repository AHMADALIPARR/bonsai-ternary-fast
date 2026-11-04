// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// The CUDA decoder on synthetic models against (a) the bonsai-decoder CPU
// reference, with the same noise-baseline criteria as test_emu_vs_cpu, and
// (b) the CPU emulator of the same algorithm. Also: CUDA graph replay ==
// eager launches (bit exact), batched prompt == one token at a time, reset.
// Exits 77 (skipped) when no CUDA device is present.
#include <cstdlib>
#include <cstring>

#include "bd/decoder.hpp"
#include "btf/fast.hpp"
#include "test_common.hpp"

static const double kFirstTol = 2e-3;
static const double kNoiseFrac = 2.0;
static const double kAbsTol = 2e-2;

int main() {
  std::string why;
  if (!btf::gpu_available(&why, 0)) {
    std::printf("test_gpu_decoder: skipped (%s)\n", why.c_str());
    return 77;
  }
  for (const auto& cfg : btft::configs()) {
    auto model = btft::make_model(cfg.so);
    bd::DecoderOptions ro;
    ro.n_ctx = 64;
    ro.kv_f16 = true;
    ro.ggml_act = true;
    auto ref = bd::make_cpu_decoder(*model, ro);
    bd::DecoderOptions uo;
    uo.n_ctx = 64;
    auto unr = bd::make_cpu_decoder(*model, uo);
    btf::FastOptions fo;
    fo.n_ctx = 64;
    fo.drop_page_cache = false;
    auto emu = btf::make_emu_decoder(*model, fo);
    auto gpu = btf::make_gpu_decoder(*model, fo, &why);
    if (!gpu) {
      std::fprintf(stderr, "make_gpu_decoder: %s\n", why.c_str());
      return 1;
    }
    btf::FastOptions fe = fo;
    fe.cuda_graph = false;
    auto eager = btf::make_gpu_decoder(*model, fe, &why);
    BD_CHECK(eager != nullptr);
    const int nv = model->hp.n_vocab;
    std::vector<float> lr(nv), lg(nv), lu(nv), le(nv), lx(nv);
    double worst = 0, noise = 0, first = -1, vs_emu = 0, vs_eager = 0;
    int steps = 0;
    auto run = [&](int t) {
      ref->forward(t, lr.data());
      unr->forward(t, lu.data());
      gpu->forward(t, lg.data());
      emu->forward(t, le.data());
      eager->forward(t, lx.data());
      const double r = btft::rel_rms(lg.data(), lr.data(), nv), n = btft::rel_rms(lu.data(), lr.data(), nv);
      if (std::getenv("BTF_VERBOSE")) std::printf("    step %d gpu %.2e noise %.2e\n", steps, r, n);
      if (first < 0) first = r;
      worst = std::max(worst, r);
      noise += n;
      vs_emu = std::max(vs_emu, btft::rel_rms(lg.data(), le.data(), nv));
      if (std::memcmp(lg.data(), lx.data(), nv * sizeof(float)) != 0) vs_eager = 1;
      ++steps;
    };
    std::vector<int> seq = btft::prompt();
    for (int t : seq) run(t);
    const std::vector<float> last_prompt_logits = lg;
    int tr = btft::argmax(lr), same = 0;
    std::vector<int> gen;
    for (int i = 0; i < 24; ++i) {
      if (btft::argmax(lg) == tr) ++same;
      gen.push_back(tr);
      run(tr);
      tr = btft::argmax(lr);
    }
    noise /= steps;
    std::printf("%-20s gpu vs ref: first %.1e, worst %.2e | noise %.2e | greedy %d/24 | vs emu %.2e | graph==eager %s\n",
                cfg.label, first, worst, noise, same, vs_emu, vs_eager == 0 ? "yes" : "NO");
    BD_CHECK(first <= kFirstTol);
    BD_CHECK(worst <= kNoiseFrac * noise || worst <= 1e-5);
    BD_CHECK(worst <= kAbsTol);
    BD_CHECK(same >= 23);
    BD_CHECK(vs_emu <= kAbsTol);
    BD_CHECK(vs_eager == 0);
    BD_CHECK(gpu->pos() == steps);

    // step() without logits returns the same greedy ids
    gpu->reset();
    BD_CHECK(gpu->pos() == 0);
    gpu->reset_stats();
    int nxt = -1;
    for (int t : seq) nxt = gpu->step(t, nullptr);
    BD_CHECK(nxt == btft::argmax(last_prompt_logits));
    // weights are resident: decode steps copy no weight bytes host-to-device
    const btf::StepStats st = gpu->stats();
    BD_CHECK(st.steps == seq.size());
    BD_CHECK(st.weight_xfer_bytes == 0);
    BD_CHECK(st.gpu_s > 0.0 && st.gpu_s <= st.wall_s);

    // batched prompt == one token at a time; reset gives the same again
    for (int mb : {8, 3}) {
      btf::FastOptions fb = fo;
      fb.max_batch = mb;
      auto gb = btf::make_gpu_decoder(*model, fb, &why);
      BD_CHECK(gb != nullptr);
      std::vector<float> lb(nv), lb2(nv);
      for (int rep = 0; rep < 2; ++rep) {
        if (rep) gb->reset();
        const int next = gb->prefill(seq.data(), (int)seq.size(), rep ? lb2.data() : lb.data());
        BD_CHECK(next == btft::argmax(rep ? lb2 : lb));
        BD_CHECK(gb->pos() == (int)seq.size());
      }
      const double d = btft::rel_rms(lb.data(), last_prompt_logits.data(), nv);
      BD_CHECK(d <= 1e-5);
      BD_CHECK(std::memcmp(lb.data(), lb2.data(), nv * sizeof(float)) == 0);
      gb->step(gen[0], lb.data());
      BD_CHECK(gb->pos() == (int)seq.size() + 1);
      std::printf("%-20s   prefill batch %d: rel rms %.2e vs per-token, reset repeat exact\n", "", mb, d);
    }
    std::printf("%-20s   %s, %.1f MiB device\n", "", gpu->info().c_str(), gpu->device_bytes() / 1048576.0);
  }
  return 0;
}
