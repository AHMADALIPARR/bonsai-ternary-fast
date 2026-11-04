// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// GGUF parser/writer. Layout (v3): "GGUF" u32 version, u64 n_tensors, u64 n_kv,
// kv pairs, tensor infos, pad to general.alignment, data.
#include "bd/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "bd/mapped_file.hpp"
#include "bd/types.hpp"

namespace bd {

size_t GgufValue::arr_len() const {
  if (type != GV_ARRAY) return 0;
  if (arr_type == GV_STRING) return as.size();
  if (arr_type == GV_F32 || arr_type == GV_F64) return af.size();
  return ai.size();
}

int64_t GgufTensor::n_elements() const {
  int64_t n = 1;
  for (int64_t x : ne) n *= x;
  return n;
}

namespace {
struct Rd {
  const uint8_t* d;
  size_t n, p = 0;
  void need(size_t k) const {
    if (k > n || p > n - k) throw std::runtime_error("gguf: truncated at byte " + std::to_string(p));
  }
  template <class T> T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, d + p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  std::string str() {
    uint64_t len = get<uint64_t>();
    if (len > (1ull << 30)) throw std::runtime_error("gguf: string too long");
    need(len);
    std::string s((const char*)d + p, len);
    p += len;
    return s;
  }
};

void read_scalar(Rd& r, uint32_t t, int64_t* i, double* f, std::string* s) {
  switch (t) {
    case GV_U8: *i = r.get<uint8_t>(); *f = (double)*i; break;
    case GV_I8: *i = r.get<int8_t>(); *f = (double)*i; break;
    case GV_U16: *i = r.get<uint16_t>(); *f = (double)*i; break;
    case GV_I16: *i = r.get<int16_t>(); *f = (double)*i; break;
    case GV_U32: *i = r.get<uint32_t>(); *f = (double)*i; break;
    case GV_I32: *i = r.get<int32_t>(); *f = (double)*i; break;
    case GV_F32: *f = r.get<float>(); *i = (int64_t)*f; break;
    case GV_BOOL: *i = r.get<uint8_t>() ? 1 : 0; *f = (double)*i; break;
    case GV_U64: *i = (int64_t)r.get<uint64_t>(); *f = (double)*i; break;
    case GV_I64: *i = r.get<int64_t>(); *f = (double)*i; break;
    case GV_F64: *f = r.get<double>(); *i = (int64_t)*f; break;
    case GV_STRING: *s = r.str(); break;
    default: throw std::runtime_error("gguf: bad value type " + std::to_string(t));
  }
}
}  // namespace

std::unique_ptr<Gguf> Gguf::open(const std::string& path, bool header_only) {
  std::unique_ptr<Gguf> g(new Gguf());
  g->map_.reset(new MappedFile(path));
  g->header_only = header_only;
  g->parse(g->map_->data(), g->map_->size());
  return g;
}

std::unique_ptr<Gguf> Gguf::from_memory(std::vector<uint8_t> bytes) {
  std::unique_ptr<Gguf> g(new Gguf());
  g->mem_ = std::move(bytes);
  g->parse(g->mem_.data(), g->mem_.size());
  return g;
}

Gguf::~Gguf() = default;

void Gguf::parse(const uint8_t* d, size_t n) {
  Rd r{d, n};
  r.need(4);
  if (std::memcmp(d, "GGUF", 4) != 0) throw std::runtime_error("gguf: bad magic");
  r.p = 4;
  version = r.get<uint32_t>();
  if (version != 2 && version != 3) throw std::runtime_error("gguf: unsupported version");
  const uint64_t nt = r.get<uint64_t>(), nkv = r.get<uint64_t>();
  if (nt > (1u << 24) || nkv > (1u << 24)) throw std::runtime_error("gguf: implausible counts");
  file_bytes = n;
  for (uint64_t k = 0; k < nkv; ++k) {
    std::string key = r.str();
    GgufValue v;
    v.type = r.get<uint32_t>();
    if (v.type == GV_ARRAY) {
      v.arr_type = r.get<uint32_t>();
      const uint64_t cnt = r.get<uint64_t>();
      if (cnt > (1ull << 32)) throw std::runtime_error("gguf: array too large");
      for (uint64_t j = 0; j < cnt; ++j) {
        int64_t i = 0; double f = 0; std::string s;
        read_scalar(r, v.arr_type, &i, &f, &s);
        if (v.arr_type == GV_STRING) v.as.push_back(std::move(s));
        else if (v.arr_type == GV_F32 || v.arr_type == GV_F64) v.af.push_back(f);
        else v.ai.push_back(i);
      }
    } else {
      read_scalar(r, v.type, &v.i, &v.f, &v.s);
    }
    if (key == "general.alignment") {
      if (v.i <= 0 || (v.i & (v.i - 1))) throw std::runtime_error("gguf: bad alignment");
      alignment = (uint64_t)v.i;
    }
    kv_order.push_back(key);
    kv[key] = std::move(v);
  }
  tensors.resize(nt);
  for (uint64_t k = 0; k < nt; ++k) {
    GgufTensor& t = tensors[k];
    t.name = r.str();
    const uint32_t nd = r.get<uint32_t>();
    if (nd == 0 || nd > 4) throw std::runtime_error("gguf: bad n_dims for " + t.name);
    for (uint32_t j = 0; j < nd; ++j) t.ne.push_back((int64_t)r.get<uint64_t>());
    t.type = r.get<uint32_t>();
    t.offset = r.get<uint64_t>();
    if (t.offset % alignment) throw std::runtime_error("gguf: misaligned " + t.name);
    TypeInfo ti = type_info(t.type);
    if (!ti.block) throw std::runtime_error("gguf: unknown ggml type " + std::to_string(t.type) + " for " + t.name);
    if (t.ne[0] % ti.block) throw std::runtime_error("gguf: row not whole blocks: " + t.name);
    t.bytes = (uint64_t)(t.n_elements() / ti.block) * (uint64_t)ti.bytes;
    tensor_index[t.name] = k;
  }
  data_offset = (r.p + alignment - 1) / alignment * alignment;
  for (GgufTensor& t : tensors) {
    const uint64_t end = data_offset + t.offset + t.bytes;
    if (end <= n) t.data = d + data_offset + t.offset;
    else if (!header_only) throw std::runtime_error("gguf: tensor past end of file: " + t.name);
  }
}

const GgufValue* Gguf::find(const std::string& key) const {
  auto it = kv.find(key);
  return it == kv.end() ? nullptr : &it->second;
}
int64_t Gguf::get_i(const std::string& key) const {
  const GgufValue* v = find(key);
  if (!v) throw std::runtime_error("gguf: missing key " + key);
  return v->i;
}
int64_t Gguf::get_i(const std::string& key, int64_t def) const {
  const GgufValue* v = find(key);
  return v ? v->i : def;
}
double Gguf::get_f(const std::string& key) const {
  const GgufValue* v = find(key);
  if (!v) throw std::runtime_error("gguf: missing key " + key);
  return v->f;
}
double Gguf::get_f(const std::string& key, double def) const {
  const GgufValue* v = find(key);
  return v ? v->f : def;
}
std::string Gguf::get_s(const std::string& key, const std::string& def) const {
  const GgufValue* v = find(key);
  return v && v->type == GV_STRING ? v->s : def;
}
bool Gguf::get_b(const std::string& key, bool def) const {
  const GgufValue* v = find(key);
  return v ? v->i != 0 : def;
}
const GgufTensor* Gguf::tensor(const std::string& name) const {
  auto it = tensor_index.find(name);
  return it == tensor_index.end() ? nullptr : &tensors[it->second];
}
const GgufTensor& Gguf::need_tensor(const std::string& name) const {
  const GgufTensor* t = tensor(name);
  if (!t) throw std::runtime_error("missing tensor " + name);
  return *t;
}

std::string Gguf::value_to_string(const GgufValue& v, size_t max_items) const {
  std::ostringstream o;
  auto sc = [&](uint32_t t, int64_t i, double f, const std::string& s) {
    if (t == GV_STRING) {
      std::string e = s.size() > 120 ? s.substr(0, 120) + "...(" + std::to_string(s.size()) + " bytes)" : s;
      for (char& c : e) if (c == '\n') c = ' ';
      o << '"' << e << '"';
    } else if (t == GV_F32 || t == GV_F64) {
      o << f;
    } else if (t == GV_BOOL) {
      o << (i ? "true" : "false");
    } else {
      o << i;
    }
  };
  if (v.type != GV_ARRAY) {
    sc(v.type, v.i, v.f, v.s);
    return o.str();
  }
  const size_t n = v.arr_len();
  o << "[" << n << "]{";
  for (size_t j = 0; j < n && j < max_items; ++j) {
    if (j) o << ", ";
    if (v.arr_type == GV_STRING) sc(GV_STRING, 0, 0, v.as[j]);
    else if (v.arr_type == GV_F32 || v.arr_type == GV_F64) sc(GV_F32, 0, v.af[j], "");
    else sc(v.arr_type, v.ai[j], 0, "");
  }
  if (n > max_items) o << ", ...";
  o << "}";
  return o.str();
}

// ------------------------------------------------------------------ writer --
namespace {
template <class T> void put(std::vector<uint8_t>& b, T v) {
  const uint8_t* p = (const uint8_t*)&v;
  b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
  put<uint64_t>(b, s.size());
  b.insert(b.end(), s.begin(), s.end());
}
}  // namespace

void GgufWriter::kv_u32(const std::string& k, uint32_t v) { put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_U32); put(kvbuf_, v); ++n_kv_; }
void GgufWriter::kv_i32(const std::string& k, int32_t v) { put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_I32); put(kvbuf_, v); ++n_kv_; }
void GgufWriter::kv_f32(const std::string& k, float v) { put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_F32); put(kvbuf_, v); ++n_kv_; }
void GgufWriter::kv_bool(const std::string& k, bool v) { put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_BOOL); put<uint8_t>(kvbuf_, v); ++n_kv_; }
void GgufWriter::kv_str(const std::string& k, const std::string& v) { put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_STRING); put_str(kvbuf_, v); ++n_kv_; }
void GgufWriter::kv_arr_i32(const std::string& k, const std::vector<int32_t>& v) {
  put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_ARRAY); put<uint32_t>(kvbuf_, GV_I32); put<uint64_t>(kvbuf_, v.size());
  for (int32_t x : v) put(kvbuf_, x);
  ++n_kv_;
}
void GgufWriter::kv_arr_f32(const std::string& k, const std::vector<float>& v) {
  put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_ARRAY); put<uint32_t>(kvbuf_, GV_F32); put<uint64_t>(kvbuf_, v.size());
  for (float x : v) put(kvbuf_, x);
  ++n_kv_;
}
void GgufWriter::kv_arr_str(const std::string& k, const std::vector<std::string>& v) {
  put_str(kvbuf_, k); put<uint32_t>(kvbuf_, GV_ARRAY); put<uint32_t>(kvbuf_, GV_STRING); put<uint64_t>(kvbuf_, v.size());
  for (const std::string& x : v) put_str(kvbuf_, x);
  ++n_kv_;
}
void GgufWriter::tensor(const std::string& name, std::vector<int64_t> ne, uint32_t type, std::vector<uint8_t> data) {
  int64_t n = 1;
  for (int64_t x : ne) n *= x;
  TypeInfo ti = type_info(type);
  if (!ti.block || n % ti.block || data.size() != (size_t)(n / ti.block * ti.bytes))
    throw std::runtime_error("GgufWriter: bad tensor " + name);
  t_.push_back({name, std::move(ne), type, std::move(data)});
}
std::vector<uint8_t> GgufWriter::finish(uint32_t alignment) const {
  std::vector<uint8_t> b;
  b.insert(b.end(), {'G', 'G', 'U', 'F'});
  put<uint32_t>(b, 3);
  put<uint64_t>(b, t_.size());
  put<uint64_t>(b, n_kv_ + 1);
  b.insert(b.end(), kvbuf_.begin(), kvbuf_.end());
  put_str(b, "general.alignment"); put<uint32_t>(b, GV_U32); put<uint32_t>(b, alignment);
  uint64_t off = 0;
  std::vector<uint64_t> offs;
  for (const T& t : t_) {
    put_str(b, t.name);
    put<uint32_t>(b, (uint32_t)t.ne.size());
    for (int64_t x : t.ne) put<uint64_t>(b, (uint64_t)x);
    put<uint32_t>(b, t.type);
    put<uint64_t>(b, off);
    offs.push_back(off);
    off = (off + t.data.size() + alignment - 1) / alignment * alignment;
  }
  b.resize((b.size() + alignment - 1) / alignment * alignment, 0);
  const size_t base = b.size();
  b.resize(base + off, 0);
  for (size_t i = 0; i < t_.size(); ++i) std::memcpy(b.data() + base + offs[i], t_[i].data.data(), t_[i].data.size());
  return b;
}

}  // namespace bd
