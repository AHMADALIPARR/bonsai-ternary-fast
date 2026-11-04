// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// btf_bench: load time, prompt tok/s (batched prefill), decode tok/s (greedy,
// CUDA graph replay) and GPU memory of the fast path. Exits 77 without a GPU
// unless --emu (CPU emulator, slow, needs ~7.3 GB RAM on the real model).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd/model.hpp"
#include "btf/fast.hpp"
#include "cli.hpp"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static void usage() {
  std::printf(
      "usage: btf_bench -m MODEL.gguf [options]\n"
      "  -p, --prompt-tokens N  prompt length (default 512)\n"
      "  -n, --gen N            decode steps (default 128)\n"
      "  -r, --reps N           repetitions, best is reported (default 3)\n"
      "  -c, --ctx N            KV cache tokens (default 4096)\n"
      "      --max-batch N      prompt rows per launch (default 8)\n"
      "      --no-graph         launch kernels eagerly for every decode step\n"
      "      --embd-device      keep token_embd in device memory\n"
      "      --device N         CUDA device\n"
      "      --emu              CPU emulator instead of the GPU\n"
      "  -t, --threads N        CPU threads (emulator / host repack)\n");
}

int main(int argc, char** argv) {
  std::string path;
  btf::FastOptions o;
  int np = 512, ng = 128, reps = 3;
  bool emu = false;
  try {
    cli::Args a{argc, argv};
    for (; a.more(); ++a.i) {
      const std::string s = a.cur();
      if (s == "-m" || s == "--model") path = a.value();
      else if (s == "-p" || s == "--prompt-tokens") np = std::atoi(a.value().c_str());
      else if (s == "-n" || s == "--gen") ng = std::atoi(a.value().c_str());
      else if (s == "-r" || s == "--reps") reps = std::atoi(a.value().c_str());
      else if (s == "-c" || s == "--ctx") o.n_ctx = std::atoi(a.value().c_str());
      else if (s == "--max-batch") o.max_batch = std::atoi(a.value().c_str());
      else if (s == "--no-graph") o.cuda_graph = false;
      else if (s == "--embd-device") o.embd_on_device = true;
      else if (s == "--device") o.device = std::atoi(a.value().c_str());
      else if (s == "--emu") emu = true;
      else if (s == "-t" || s == "--threads") o.threads = std::atoi(a.value().c_str());
      else if (s == "-h" || s == "--help") { usage(); return 0; }
      else throw std::runtime_error("unknown option " + s);
    }
    if (path.empty()) { usage(); return 2; }
    if (np + ng > o.n_ctx) throw std::runtime_error("prompt + gen exceeds --ctx");
    std::string why;
    if (!emu && !btf::gpu_available(&why, o.device)) {
      std::printf("btf_bench: no GPU (%s), skipped\n", why.c_str());
      return 77;
    }
    auto m = bd::Model::load(path);
    o.gguf_path = path;
    const btf::MemoryPlan plan = btf::plan_memory(*m, o);
    std::printf("plan: %.3f GB device, %.3f GB pinned host\n", plan.device_bytes() / 1e9, plan.host_bytes() / 1e9);
    auto t0 = clk::now();
    std::unique_ptr<btf::FastDecoder> dec = emu ? btf::make_emu_decoder(*m, o) : btf::make_gpu_decoder(*m, o, &why);
    if (!dec) throw std::runtime_error("GPU decoder: " + why);
    const double load_s = since(t0);
    std::printf("%s\nload + repack: %.2f s, device memory in use by the decoder: %.3f GB\n", dec->info().c_str(),
                load_s, dec->device_bytes() / 1e9);
    std::vector<int32_t> prompt(np);
    for (int i = 0; i < np; ++i) prompt[i] = 1000 + (int32_t)((i * 7919u) % 100000u);
    double best_p = 1e30, best_g = 1e30;
    int last = 0;
    btf::StepStats best_st;
    for (int r = 0; r < reps; ++r) {
      dec->reset();
      t0 = clk::now();
      int tok = dec->prefill(prompt.data(), np, nullptr);
      const double tp = since(t0);
      dec->reset_stats();
      t0 = clk::now();
      for (int i = 0; i < ng; ++i) tok = dec->step(tok, nullptr);
      const double tg = since(t0);
      last = tok;
      best_p = std::min(best_p, tp);
      if (tg < best_g) best_st = dec->stats();
      best_g = std::min(best_g, tg);
      std::printf("rep %d: prompt %d tok %.3f s (%.1f tok/s), decode %d tok %.3f s (%.2f tok/s)\n", r, np, tp,
                  np / tp, ng, tg, ng / tg);
    }
    std::printf("best: prompt %.1f tok/s, decode %.2f tok/s (%.2f ms/token), last token %d\n", np / best_p,
                ng / best_g, 1e3 * best_g / ng, last);

    // Per-token breakdown of the best decode rep, in the same terms as an
    // adapter that re-sends weights every token (weight transfer + compute).
    const double sp = best_g / ng;  // s/token
    const uint64_t n = best_st.steps ? best_st.steps : 1;
    const uint64_t traffic = plan.decode_bytes_fixed + plan.decode_bytes_per_ctx * (uint64_t)(np + ng / 2);
    std::printf("per-token breakdown (decode, best rep, %d tokens after a %d-token prompt):\n", ng, np);
    std::printf("  weight transfer host->device   %12.6f s/token  (%llu B/token)\n", best_st.weight_xfer_s / n,
                (unsigned long long)(best_st.weight_xfer_bytes / n));
    if (best_st.gpu_s > 0) {
      std::printf("  GPU compute (CUDA events)      %12.6f s/token\n", best_st.gpu_s / n);
      std::printf("  host overhead (wall - GPU)     %12.6f s/token\n", (best_st.wall_s - best_st.gpu_s) / n);
    } else {
      std::printf("  compute (CPU emulator, wall)   %12.6f s/token\n", best_st.wall_s / n);
    }
    std::printf("  total                          %12.6f s/token  = %.3f tok/s\n", sp, 1.0 / sp);
    std::printf("  token_embd row read zero-copy  %12llu B/token\n", (unsigned long long)(best_st.mapped_read_bytes / n));
    std::printf("  minimum memory traffic         %12.3f GB/token (at %d tokens of context)\n", traffic / 1e9,
                np + ng / 2);
    if (best_st.gpu_s > 0)
      std::printf("  effective bandwidth            %12.1f GB/s  (traffic / GPU time)\n",
                  traffic / 1e9 / (best_st.gpu_s / n));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "btf_bench: %s\n", e.what());
    return 2;
  }
}
