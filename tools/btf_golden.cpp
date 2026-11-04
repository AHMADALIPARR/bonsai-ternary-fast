// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// btf_golden: greedy tokens of the fast path against the bonsai-decoder CPU
// reference (ggml activation rounding + fp16 KV, the settings that match
// prism llama.cpp) on the real model, for one chat turn.
//
// The reference run takes minutes on a CPU, so it can be saved once
// (--save-ref FILE) and replayed (--ref-tokens FILE). Exit codes: 0 all tokens
// match, 1 a difference, 2 usage/load error, 77 no GPU (without --emu).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd/decoder.hpp"
#include "bd/model.hpp"
#include "bd/tokenizer.hpp"
#include "btf/fast.hpp"
#include "cli.hpp"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static void usage() {
  std::printf(
      "usage: btf_golden -m MODEL.gguf [options]\n"
      "%s"
      "  -n N                 max generated tokens (default 64)\n"
      "      --ref-tokens F   reference tokens from F (written by --save-ref)\n"
      "      --save-ref F     run the CPU reference and save its tokens to F\n"
      "      --ref-only       only run the reference (with --save-ref)\n"
      "      --emu            check the CPU emulator instead of the GPU\n"
      "      --no-graph       GPU: no CUDA graph\n"
      "  -c, --ctx N          KV cache tokens (default 4096)\n"
      "      --device N       CUDA device\n"
      "  -t, --threads N      CPU threads\n",
      cli::PromptOpts::help());
}

static std::string join(const std::vector<int32_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
  return s;
}
static std::vector<int32_t> parse_ids(const std::string& s) {
  std::istringstream ss(s);
  std::vector<int32_t> v;
  long x;
  while (ss >> x) v.push_back((int32_t)x);
  return v;
}

static void top5(const char* who, const std::vector<float>& l, const bd::Tokenizer& tok) {
  std::vector<int> idx(l.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  std::printf("  %s top5:", who);
  for (int i = 0; i < 5; ++i)
    std::printf(" %d '%s' %.4f;", idx[i], cli::printable(tok.piece(idx[i])).c_str(), l[idx[i]]);
  std::printf("\n");
}

int main(int argc, char** argv) {
  std::string path, ref_file, save_file;
  cli::PromptOpts po;
  po.text = "Hello!";
  btf::FastOptions o;
  int n_gen = 64, threads = 0;
  bool emu = false, ref_only = false;
  try {
    cli::Args a{argc, argv};
    for (; a.more(); ++a.i) {
      const std::string s = a.cur();
      if (po.parse(a)) continue;
      if (s == "-m" || s == "--model") path = a.value();
      else if (s == "-n") n_gen = std::atoi(a.value().c_str());
      else if (s == "--ref-tokens") ref_file = a.value();
      else if (s == "--save-ref") save_file = a.value();
      else if (s == "--ref-only") ref_only = true;
      else if (s == "--emu") emu = true;
      else if (s == "--no-graph") o.cuda_graph = false;
      else if (s == "-c" || s == "--ctx") o.n_ctx = std::atoi(a.value().c_str());
      else if (s == "--device") o.device = std::atoi(a.value().c_str());
      else if (s == "-t" || s == "--threads") threads = std::atoi(a.value().c_str());
      else if (s == "-h" || s == "--help") { usage(); return 0; }
      else throw std::runtime_error("unknown option " + s);
    }
    if (path.empty()) { usage(); return 2; }
    o.threads = threads;
    std::string why;
    if (!ref_only && !emu && !btf::gpu_available(&why, o.device)) {
      std::printf("btf_golden: no GPU (%s), skipped\n", why.c_str());
      return 77;
    }
    auto m = bd::Model::load(path);
    o.gguf_path = path;
    bd::Tokenizer tok(*m->gguf);
    std::vector<int32_t> prompt = po.ids(tok), ref;
    std::printf("prompt: %zu tokens: %s\n", prompt.size(), cli::printable(po.rendered()).c_str());
    const int nv = m->hp.n_vocab;

    if (!ref_file.empty()) {
      std::ifstream f(ref_file);
      if (!f) throw std::runtime_error("cannot open " + ref_file);
      std::string line, pl, gl;
      while (std::getline(f, line)) {
        if (line.rfind("prompt:", 0) == 0) pl = line.substr(7);
        if (line.rfind("tokens:", 0) == 0) gl = line.substr(7);
      }
      if (parse_ids(pl) != prompt) throw std::runtime_error(ref_file + ": prompt ids differ from this prompt");
      ref = parse_ids(gl);
      std::printf("reference: %zu tokens from %s\n", ref.size(), ref_file.c_str());
    } else {
      bd::DecoderOptions ro;
      ro.n_ctx = o.n_ctx;
      ro.threads = threads;
      ro.kv_f16 = true;
      ro.ggml_act = true;
      auto cpu = bd::make_cpu_decoder(*m, ro);
      std::vector<float> l(nv);
      auto t0 = clk::now();
      for (int32_t t : prompt) cpu->forward(t, l.data());
      int next = (int)(std::max_element(l.begin(), l.end()) - l.begin());
      while ((int)ref.size() < n_gen) {
        ref.push_back(next);
        if (tok.is_eog(next)) break;
        cpu->forward(next, l.data());
        next = (int)(std::max_element(l.begin(), l.end()) - l.begin());
      }
      std::printf("reference (bonsai-decoder CPU, ggml-act, kv-f16): %zu tokens in %.1f s\n", ref.size(), since(t0));
      if (!save_file.empty()) {
        std::ofstream f(save_file);
        f << "# btf_golden reference: bonsai-decoder CPU (ggml_act, kv_f16), greedy\n";
        f << "# text: " << cli::printable(po.rendered()) << "\n";
        f << "prompt:" << join(prompt) << "\n";
        f << "tokens:" << join(ref) << "\n";
        std::printf("saved %s\n", save_file.c_str());
      }
    }
    std::printf("reference text: %s\n", cli::printable(tok.decode(ref)).c_str());
    if (ref_only) return 0;

    auto t0 = clk::now();
    std::unique_ptr<btf::FastDecoder> dec = emu ? btf::make_emu_decoder(*m, o) : btf::make_gpu_decoder(*m, o, &why);
    if (!dec) throw std::runtime_error("decoder: " + why);
    std::printf("%s loaded in %.2f s\n", dec->info().c_str(), since(t0));
    std::vector<float> l(nv);
    t0 = clk::now();
    int next = dec->prefill(prompt.data(), (int)prompt.size(), l.data());
    std::vector<int32_t> got;
    int first_diff = -1;
    for (size_t i = 0; i < ref.size(); ++i) {
      got.push_back(next);
      if (next != ref[i]) {
        first_diff = (int)i;
        std::printf("DIFF at generated token %zu: fast %d '%s', reference %d '%s'\n", i, next,
                    cli::printable(tok.piece(next)).c_str(), ref[i], cli::printable(tok.piece(ref[i])).c_str());
        top5("fast", l, tok);
        break;
      }
      if (i + 1 < ref.size()) next = dec->step(ref[i], l.data());
    }
    const double secs = since(t0);
    std::printf("fast text: %s\n", cli::printable(tok.decode(got)).c_str());
    const size_t same = first_diff < 0 ? ref.size() : (size_t)first_diff;
    std::printf("greedy tokens: %zu/%zu match (%.2f s for %zu prompt + %zu tokens)\n", same, ref.size(), secs,
                prompt.size(), got.size());
    return first_diff < 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "btf_golden: %s\n", e.what());
    return 2;
  }
}
