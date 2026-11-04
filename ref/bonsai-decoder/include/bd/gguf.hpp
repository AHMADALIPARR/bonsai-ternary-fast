// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// GGUF v2/v3 reader that keeps every metadata value (scalars, strings, arrays)
// and every tensor info. Tensor data is reached through the mmap (or a memory
// buffer) without copying. ne[] is GGUF/ggml order: ne[0] is contiguous.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bd {

enum GgufType : uint32_t {
  GV_U8 = 0, GV_I8 = 1, GV_U16 = 2, GV_I16 = 3, GV_U32 = 4, GV_I32 = 5,
  GV_F32 = 6, GV_BOOL = 7, GV_STRING = 8, GV_ARRAY = 9, GV_U64 = 10,
  GV_I64 = 11, GV_F64 = 12,
};

struct GgufValue {
  uint32_t type = 0;       // GgufType
  uint32_t arr_type = 0;   // element type when type == GV_ARRAY
  // scalars: numeric value in both forms; strings: s
  int64_t i = 0;
  double f = 0.0;
  std::string s;
  // arrays
  std::vector<int64_t> ai;
  std::vector<double> af;
  std::vector<std::string> as;
  size_t arr_len() const;
};

struct GgufTensor {
  std::string name;
  std::vector<int64_t> ne;  // up to 4 dims, ggml order
  uint32_t type = 0;        // ggml type id
  uint64_t offset = 0;      // relative to data section
  uint64_t bytes = 0;
  const uint8_t* data = nullptr;  // nullptr when header-only
  int64_t n_elements() const;
  int64_t dim(int d) const { return d < (int)ne.size() ? ne[d] : 1; }
};

class MappedFile;

class Gguf {
 public:
  // Opens and mmaps a file. header_only: accept a truncated file (header bytes
  // only, e.g. from an HTTP range request); tensor data pointers stay null.
  static std::unique_ptr<Gguf> open(const std::string& path, bool header_only = false);
  // Parses an in-memory image (used by the synthetic-model tests).
  static std::unique_ptr<Gguf> from_memory(std::vector<uint8_t> bytes);
  ~Gguf();

  uint32_t version = 0;
  uint64_t alignment = 32;
  uint64_t data_offset = 0;
  uint64_t file_bytes = 0;
  bool header_only = false;
  std::vector<std::string> kv_order;
  std::map<std::string, GgufValue> kv;
  std::vector<GgufTensor> tensors;
  std::map<std::string, size_t> tensor_index;

  const GgufValue* find(const std::string& key) const;
  bool has(const std::string& key) const { return find(key) != nullptr; }
  int64_t get_i(const std::string& key) const;              // throws if missing
  int64_t get_i(const std::string& key, int64_t def) const;
  double get_f(const std::string& key) const;
  double get_f(const std::string& key, double def) const;
  std::string get_s(const std::string& key, const std::string& def = "") const;
  bool get_b(const std::string& key, bool def) const;
  const GgufTensor* tensor(const std::string& name) const;  // nullptr if absent
  const GgufTensor& need_tensor(const std::string& name) const;

  std::string value_to_string(const GgufValue& v, size_t max_items = 8) const;

 private:
  void parse(const uint8_t* d, size_t n);
  std::unique_ptr<MappedFile> map_;
  std::vector<uint8_t> mem_;
};

// Minimal GGUF writer used by the synthetic-model generator and tests.
class GgufWriter {
 public:
  void kv_u32(const std::string& k, uint32_t v);
  void kv_i32(const std::string& k, int32_t v);
  void kv_f32(const std::string& k, float v);
  void kv_bool(const std::string& k, bool v);
  void kv_str(const std::string& k, const std::string& v);
  void kv_arr_i32(const std::string& k, const std::vector<int32_t>& v);
  void kv_arr_f32(const std::string& k, const std::vector<float>& v);
  void kv_arr_str(const std::string& k, const std::vector<std::string>& v);
  void tensor(const std::string& name, std::vector<int64_t> ne, uint32_t type,
              std::vector<uint8_t> data);
  std::vector<uint8_t> finish(uint32_t alignment = 32) const;

 private:
  std::vector<uint8_t> kvbuf_;
  uint64_t n_kv_ = 0;
  struct T {
    std::string name;
    std::vector<int64_t> ne;
    uint32_t type;
    std::vector<uint8_t> data;
  };
  std::vector<T> t_;
};

}  // namespace bd
