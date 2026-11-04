// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Small shared helpers for the command line tools (argument parsing, prompt
// construction, decoder creation).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd/decoder.hpp"
#include "bd/model.hpp"
#include "bd/tokenizer.hpp"

namespace cli {

struct Args {
  int argc;
  char** argv;
  int i = 1;
  bool more() const { return i < argc; }
  std::string cur() const { return argv[i]; }
  std::string value() {
    if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
    return argv[++i];
  }
};

inline std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Prompt options shared by bonsai_run and bonsai_trace.
struct PromptOpts {
  std::string text;          // -p / -f
  std::string system;        // --system
  std::string tokens;        // --tokens "1,2,3" (overrides text)
  bool chat = true;          // --raw disables the chat template
  bool thinking = true;      // --no-think
  bool parse_special = true; // --no-special
  bool parse(Args& a) {
    const std::string s = a.cur();
    if (s == "-p" || s == "--prompt") text = a.value();
    else if (s == "-f" || s == "--file") text = read_file(a.value());
    else if (s == "--system") system = a.value();
    else if (s == "--tokens") tokens = a.value();
    else if (s == "--raw") chat = false;
    else if (s == "--chat") chat = true;
    else if (s == "--no-think") thinking = false;
    else if (s == "--think") thinking = true;
    else if (s == "--no-special") parse_special = false;
    else return false;
    return true;
  }
  std::string rendered() const { return chat ? bd::chat_prompt_qwen35(text, system, thinking) : text; }
  std::vector<int32_t> ids(const bd::Tokenizer& tok) const {
    std::vector<int32_t> v;
    if (!tokens.empty()) {
      std::string t = tokens;
      for (char& c : t)
        if (c == ',') c = ' ';
      std::istringstream ss(t);
      long x;
      while (ss >> x) v.push_back((int32_t)x);
      return v;
    }
    if (tok.add_bos() && tok.bos() >= 0) v.push_back(tok.bos());
    auto e = tok.encode(rendered(), parse_special);
    v.insert(v.end(), e.begin(), e.end());
    return v;
  }
  static const char* help() {
    return "  -p, --prompt TEXT     prompt text\n"
           "  -f, --file PATH       prompt from file\n"
           "      --system TEXT     system message (chat mode)\n"
           "      --raw             no chat template (text is tokenized as is)\n"
           "      --no-think        chat template with enable_thinking=false\n"
           "      --tokens 1,2,3    feed these token ids instead of a prompt\n"
           "      --no-special      do not parse <|special|> tokens in the text\n";
  }
};

struct DecOpts {
  bool gpu = false;
  bd::DecoderOptions o;
  bool parse(Args& a) {
    const std::string s = a.cur();
    if (s == "--gpu") gpu = true;
    else if (s == "--cpu") gpu = false;
    else if (s == "-t" || s == "--threads") o.threads = std::atoi(a.value().c_str());
    else if (s == "-c" || s == "--ctx") o.n_ctx = std::atoi(a.value().c_str());
    else if (s == "--kv-f16") o.kv_f16 = true;
    else if (s == "--ggml-act") o.ggml_act = true;
    else if (s == "--device") o.device = std::atoi(a.value().c_str());
    else return false;
    return true;
  }
  std::unique_ptr<bd::Decoder> make(const bd::Model& m) const {
    if (!gpu) return bd::make_cpu_decoder(m, o);
    std::string why;
    auto d = bd::make_cuda_decoder(m, o, &why);
    if (!d) throw std::runtime_error("CUDA decoder unavailable: " + why);
    return d;
  }
  static const char* help() {
    return "      --gpu             CUDA decoder (default: CPU fp32 reference)\n"
           "  -t, --threads N       CPU threads (default: all)\n"
           "  -c, --ctx N           KV cache size in tokens (default 4096)\n"
           "      --kv-f16          round K/V through fp16 like llama.cpp's F16 cache\n"
           "      --ggml-act        CPU: round matmul inputs like ggml-cpu (Q8_0 for PTQ1_0,\n"
           "                        bf16 for BF16) for op-by-op comparison with llama.cpp\n";
  }
};

inline std::string printable(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c == '\n') o += "\\n";
    else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; }
    else o += (char)c;
  }
  return o;
}

}  // namespace cli
