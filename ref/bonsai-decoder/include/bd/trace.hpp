// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Activation trace: one record per (op, token). File format "BDTRACE1":
//   magic[8] = "BDTRACE1", then records until EOF:
//   u32 name_len, name bytes, i32 pos, u32 n_dims, i64 ne[n_dims], f32 data[prod(ne)]
// Names follow the llama.cpp cb() names with "-<layer>" suffix so that the
// compare tool can line them up with llama-eval-callback output.
#pragma once
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace bd {

class TraceSink {
 public:
  virtual ~TraceSink() = default;
  virtual bool wants(const std::string& name) const { (void)name; return true; }
  virtual void record(const std::string& name, int pos, const std::vector<int64_t>& ne, const float* data) = 0;
};

class TraceFileWriter : public TraceSink {
 public:
  // filters: empty = all; otherwise a name is kept if it starts with any filter
  TraceFileWriter(const std::string& path, std::vector<std::string> filters = {});
  ~TraceFileWriter() override;
  bool wants(const std::string& name) const override;
  void record(const std::string& name, int pos, const std::vector<int64_t>& ne, const float* data) override;
  size_t records() const { return n_; }

 private:
  FILE* f_ = nullptr;
  std::vector<std::string> filters_;
  size_t n_ = 0;
};

// In-memory trace (tests, CPU-vs-GPU compare).
class TraceMemory : public TraceSink {
 public:
  struct Rec {
    std::string name;
    int pos;
    std::vector<int64_t> ne;
    std::vector<float> data;
  };
  void record(const std::string& name, int pos, const std::vector<int64_t>& ne, const float* data) override;
  std::vector<Rec> recs;
};

struct TraceRecord {
  std::string name;
  int pos = 0;
  std::vector<int64_t> ne;
  std::vector<float> data;
};
// Reads a BDTRACE1 file (throws on a malformed file).
std::vector<TraceRecord> read_trace(const std::string& path);

// One tensor parsed from llama-eval-callback (common/debug.cpp) output: the
// printer shows up to 3 leading and 3 trailing indices per dim plus the sum.
struct EvalCbTensor {
  std::string name;
  std::string type;
  int64_t ne[4] = {1, 1, 1, 1};
  std::vector<int64_t> idx;   // flat index of each printed value
  std::vector<float> val;     // printed values (%12.4f)
  std::vector<double> sums;   // one "sum = " per i3
  size_t order = 0;           // position in the log
};
std::vector<EvalCbTensor> read_eval_callback_log(const std::string& path);

// Element-wise comparison summary.
struct DiffStats {
  int64_t n = 0;
  double max_abs = 0, max_rel = 0, ref_max_abs = 0;
  int64_t worst = -1;
  double rms_ref = 0, rms_diff = 0;
  double rel_rms() const { return rms_ref > 0 ? rms_diff / rms_ref : rms_diff; }
};
DiffStats diff(const float* a, const float* ref, int64_t n, double rel_floor = 1e-3);

}  // namespace bd
