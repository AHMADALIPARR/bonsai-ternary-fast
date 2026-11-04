// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Public API of the fast ternary decode path: options, memory plan, host
// repack, the CPU emulator and the CUDA decoder.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bd/cpu_ops.hpp"
#include "bd/decoder.hpp"
#include "bd/model.hpp"

namespace btf {

struct FastOptions {
  int n_ctx = 4096;          // KV cache capacity (tokens)
  int device = 0;            // CUDA device
  int max_batch = 8;         // prompt rows per kernel launch (1..8)
  bool cuda_graph = true;    // capture one decode step as a CUDA graph and replay it
  bool round_q_f16 = true;   // round q to fp16 before q.k (ggml's F16 KV vec-dot), like the reference with --ggml-act --kv-f16
  bool embd_on_device = false;  // keep token_embd (packed) in device memory instead of mapped host memory
  bool drop_page_cache = true;  // posix_fadvise(DONTNEED) on gguf_path once its tensors are uploaded
  std::string gguf_path;        // file the model was loaded from (only used for drop_page_cache)
  int staging_mb = 64;       // pinned staging buffer size per slot (2 slots)
  int threads = 0;           // CPU emulator / host repack threads (0 = all)
  int attn_splits = 16;      // flash-decoding splits of the KV range
  bool verbose = false;
};

// One packed matrix on the host (emulator, tests).
struct PackedMatrix {
  int64_t N = 0, K = 0;
  std::vector<uint8_t> codes;    // [N][K/4]
  std::vector<uint16_t> scales;  // [N][K/128] fp16 bits
  bd::HadamardXform hx;
};

// Repacks rows of a PTQ1_0 tensor: src points to row 0, K cols per row.
void repack_rows(const uint8_t* src, int64_t rows, int64_t K, uint8_t* codes, uint16_t* scales);
PackedMatrix repack(const bd::Weight& w, bd::ThreadPool* pool = nullptr);

// Size of the packed form of a [N][K] matrix.
inline uint64_t packed_codes_bytes(int64_t N, int64_t K) { return (uint64_t)N * (K / 4); }
inline uint64_t packed_scales_bytes(int64_t N, int64_t K) { return (uint64_t)N * (K / 128) * 2; }

// Throws std::runtime_error if the model is outside what the fast path
// supports (non-PTQ1_0 big matrices, odd shapes, ...).
void check_supported(const bd::Model& m, const FastOptions& o);

struct PlanItem {
  std::string what;
  uint64_t bytes = 0;
  bool host = false;  // pinned mapped host memory instead of device memory
};
struct MemoryPlan {
  std::vector<PlanItem> items;
  uint64_t device_bytes() const;
  uint64_t host_bytes() const;
  std::string table() const;  // human readable
  // Minimum memory traffic of one decode step at batch 1, every byte touched
  // once: all resident weights + the token_embd row read, GDN/conv state read
  // and written, logits written and read by argmax. Add decode_bytes_per_ctx
  // per token already in the KV cache (K and V of every attention layer).
  uint64_t decode_bytes_fixed = 0;
  uint64_t decode_bytes_per_ctx = 0;
};
// Exact allocation sizes the GPU decoder will make (from tensor shapes only).
MemoryPlan plan_memory(const bd::Model& m, const FastOptions& o);

// What decode steps cost, accumulated by step() (see btf_bench).
struct StepStats {
  uint64_t steps = 0;              // step() calls counted
  double wall_s = 0;               // host wall time inside step()
  double gpu_s = 0;                // GPU time between CUDA events around each step (0: not measured)
  uint64_t weight_xfer_bytes = 0;  // host-to-device weight copies made during those steps
  double weight_xfer_s = 0;        // wall time of those copies
  uint64_t mapped_read_bytes = 0;  // token_embd bytes the GPU read zero-copy from mapped host memory
};

// Decoder with a batched prompt path. forward() is one token.
class FastDecoder : public bd::Decoder {
 public:
  // Runs n tokens (the prompt) in chunks of max_batch; writes the logits of the
  // last one if logits != nullptr. Returns the greedy (argmax) next token.
  virtual int prefill(const int32_t* tokens, int n, float* logits) = 0;
  // One decode step for token; returns the greedy next token. With logits ==
  // nullptr nothing is copied back but the token id (GPU: graph replay).
  virtual int step(int32_t token, float* logits) = 0;
  void forward(int token, float* logits) override { step(token, logits); }
  // Per-token accounting of decode steps since reset_stats() (prefill excluded).
  virtual StepStats stats() const { return {}; }
  virtual void reset_stats() {}
  // Device memory in use by this decoder (0 for the emulator).
  virtual uint64_t device_bytes() const { return 0; }
  virtual std::string info() const { return name(); }
};

// CPU emulation of the GPU algorithm on the same packed layout (validation).
std::unique_ptr<FastDecoder> make_emu_decoder(const bd::Model& m, const FastOptions& o);
// CUDA decoder; nullptr + why when CUDA or a device is missing.
std::unique_ptr<FastDecoder> make_gpu_decoder(const bd::Model& m, const FastOptions& o, std::string* why);
bool gpu_available(std::string* why, int device = 0);

}  // namespace btf
