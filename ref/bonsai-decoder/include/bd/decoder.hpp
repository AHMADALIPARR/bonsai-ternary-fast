// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Decoder interface shared by the CPU fp32 reference and the CUDA decoder.
// Tokens are fed one at a time (positions 0, 1, 2, ...); the KV cache (full
// attention layers) and the conv + delta-net state (linear layers) persist
// until reset().
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "bd/model.hpp"
#include "bd/trace.hpp"

namespace bd {

struct DecoderOptions {
  int n_ctx = 4096;      // KV cache capacity (tokens)
  int threads = 0;       // CPU threads, 0 = all
  bool kv_f16 = false;   // round K/V through fp16 like llama.cpp's default F16 cache
  bool ggml_act = false; // CPU only: round matmul inputs like ggml-cpu (Q8_0 for PTQ1_0, bf16 for BF16)
  int device = 0;        // CUDA device
};

class Decoder {
 public:
  virtual ~Decoder() = default;
  virtual void reset() = 0;
  // Runs token at position pos (must equal the number of tokens fed since
  // reset) and writes n_vocab logits.
  virtual void forward(int token, float* logits) = 0;
  virtual int pos() const = 0;
  virtual std::string name() const = 0;
  void set_trace(TraceSink* t) { trace_ = t; }

 protected:
  TraceSink* trace_ = nullptr;
};

std::unique_ptr<Decoder> make_cpu_decoder(const Model& m, const DecoderOptions& o);
// Returns nullptr (and sets why) when built without CUDA or no device exists.
std::unique_ptr<Decoder> make_cuda_decoder(const Model& m, const DecoderOptions& o, std::string* why);
bool cuda_available(std::string* why);

}  // namespace bd
