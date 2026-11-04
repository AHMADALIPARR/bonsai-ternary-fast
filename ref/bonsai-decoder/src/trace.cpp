// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
#include "bd/trace.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace bd {

TraceFileWriter::TraceFileWriter(const std::string& path, std::vector<std::string> filters)
    : filters_(std::move(filters)) {
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) throw std::runtime_error("trace: cannot create " + path);
  std::fwrite("BDTRACE1", 1, 8, f_);
}

TraceFileWriter::~TraceFileWriter() {
  if (f_) std::fclose(f_);
}

bool TraceFileWriter::wants(const std::string& name) const {
  if (filters_.empty()) return true;
  for (auto& p : filters_)
    if (name.compare(0, p.size(), p) == 0) return true;
  return false;
}

void TraceFileWriter::record(const std::string& name, int pos, const std::vector<int64_t>& ne, const float* data) {
  if (!wants(name)) return;
  const uint32_t nl = (uint32_t)name.size(), nd = (uint32_t)ne.size();
  const int32_t p = pos;
  int64_t n = 1;
  for (auto v : ne) n *= v;
  std::fwrite(&nl, 4, 1, f_);
  std::fwrite(name.data(), 1, nl, f_);
  std::fwrite(&p, 4, 1, f_);
  std::fwrite(&nd, 4, 1, f_);
  std::fwrite(ne.data(), 8, nd, f_);
  std::fwrite(data, 4, n, f_);
  ++n_;
}

void TraceMemory::record(const std::string& name, int pos, const std::vector<int64_t>& ne, const float* data) {
  int64_t n = 1;
  for (auto v : ne) n *= v;
  recs.push_back({name, pos, ne, std::vector<float>(data, data + n)});
}

std::vector<TraceRecord> read_trace(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("trace: cannot open " + path);
  char magic[8];
  if (!f.read(magic, 8) || std::memcmp(magic, "BDTRACE1", 8)) throw std::runtime_error("trace: bad magic in " + path);
  std::vector<TraceRecord> out;
  for (;;) {
    uint32_t nl;
    if (!f.read((char*)&nl, 4)) break;
    if (nl > 4096) throw std::runtime_error("trace: corrupt name length");
    TraceRecord r;
    r.name.resize(nl);
    uint32_t nd;
    int32_t pos;
    if (!f.read(&r.name[0], nl) || !f.read((char*)&pos, 4) || !f.read((char*)&nd, 4) || nd > 8)
      throw std::runtime_error("trace: truncated record");
    r.pos = pos;
    r.ne.resize(nd);
    if (!f.read((char*)r.ne.data(), 8 * nd)) throw std::runtime_error("trace: truncated dims");
    int64_t n = 1;
    for (auto v : r.ne) n *= v;
    if (n < 0 || n > (1LL << 31)) throw std::runtime_error("trace: corrupt dims");
    r.data.resize(n);
    if (!f.read((char*)r.data.data(), 4 * n)) throw std::runtime_error("trace: truncated data");
    out.push_back(std::move(r));
  }
  return out;
}

// ------------------------------------------------------------ eval-callback
// Header printed by common_debug_cb_eval (common/debug.cpp):
//   "common_debug_cb_eval: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n"
// followed by nested [ ... ] rows with "%12.4f" values and "sum = %f".
namespace {
bool parse_header(const std::string& line, EvalCbTensor& t) {
  const std::string tag = "common_debug_cb_eval:";
  size_t p = line.find(tag);
  if (p == std::string::npos) {
    // older builds print "ggml_debug:"
    p = line.find("ggml_debug:");
    if (p == std::string::npos) return false;
    p += 11;
  } else {
    p += tag.size();
  }
  const size_t eq = line.find(" = (", p);
  if (eq == std::string::npos) return false;
  size_t b = p;
  while (b < eq && line[b] == ' ') ++b;
  t.name = line.substr(b, eq - b);
  const size_t tc = line.find(')', eq + 4);
  t.type = line.substr(eq + 4, tc - (eq + 4));
  const size_t ob = line.rfind("= {");
  const size_t cb = line.rfind('}');
  if (ob == std::string::npos || cb == std::string::npos || cb < ob) return false;
  std::string dims = line.substr(ob + 3, cb - ob - 3);
  int d = 0;
  size_t s = 0;
  while (d < 4 && s < dims.size()) {
    size_t c = dims.find(',', s);
    t.ne[d++] = std::stoll(dims.substr(s, c == std::string::npos ? std::string::npos : c - s));
    if (c == std::string::npos) break;
    s = c + 1;
  }
  return true;
}

// Flat indices visited by common_debug_print_tensor with n = 3.
std::vector<int64_t> printed_indices(const int64_t* ne) {
  std::vector<int64_t> v;
  const int64_t n = 3;
  for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    for (int64_t i2 = 0; i2 < ne[2]; i2++) {
      if (i2 == n && ne[2] > 2 * n) i2 = ne[2] - n;
      for (int64_t i1 = 0; i1 < ne[1]; i1++) {
        if (i1 == n && ne[1] > 2 * n) i1 = ne[1] - n;
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
          if (i0 == n && ne[0] > 2 * n) i0 = ne[0] - n;
          v.push_back(i0 + ne[0] * (i1 + ne[1] * (i2 + ne[2] * i3)));
        }
      }
    }
  }
  return v;
}
}  // namespace

std::vector<EvalCbTensor> read_eval_callback_log(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("eval-callback log: cannot open " + path);
  std::vector<EvalCbTensor> out;
  std::string line;
  EvalCbTensor cur;
  bool in = false;
  std::vector<float> vals;
  auto flush = [&] {
    if (!in) return;
    cur.idx = printed_indices(cur.ne);
    if (vals.size() == cur.idx.size()) {
      cur.val = vals;
    } else {
      cur.idx.clear();  // quantized / not printed: header only
    }
    cur.order = out.size();
    out.push_back(cur);
    in = false;
  };
  while (std::getline(f, line)) {
    EvalCbTensor h;
    if (parse_header(line, h)) {
      flush();
      cur = h;
      in = true;
      vals.clear();
      continue;
    }
    if (!in) continue;
    const size_t sp = line.find("sum = ");
    if (sp != std::string::npos) {
      cur.sums.push_back(std::strtod(line.c_str() + sp + 6, nullptr));
      int64_t n3 = cur.ne[3];
      if ((int64_t)cur.sums.size() >= n3) flush();
      continue;
    }
    if (line.find('[') == std::string::npos) continue;
    // value row: numbers separated by commas, "..." markers skipped
    const char* s = line.c_str();
    while (*s) {
      if ((*s >= '0' && *s <= '9') || *s == '-' || *s == '+' || (*s == 'n') || (*s == 'i')) {
        char* end = nullptr;
        const float v = std::strtof(s, &end);
        if (end == s) {
          ++s;
          continue;
        }
        vals.push_back(v);
        s = end;
      } else {
        ++s;
      }
    }
  }
  flush();
  return out;
}

DiffStats diff(const float* a, const float* ref, int64_t n, double rel_floor) {
  DiffStats d;
  d.n = n;
  double sr = 0, sd = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double r = ref[i], e = std::fabs((double)a[i] - r);
    const double rel = e / std::fmax(std::fabs(r), rel_floor);
    if (!(e <= d.max_abs) || std::isnan(e)) {
      if (std::isnan(e) || e > d.max_abs) {
        d.max_abs = std::isnan(e) ? INFINITY : e;
        d.worst = i;
      }
    }
    if (rel > d.max_rel || std::isnan(rel)) d.max_rel = std::isnan(rel) ? INFINITY : rel;
    d.ref_max_abs = std::fmax(d.ref_max_abs, std::fabs(r));
    sr += r * r;
    sd += e * e;
  }
  d.rms_ref = n ? std::sqrt(sr / n) : 0;
  d.rms_diff = n ? std::sqrt(sd / n) : 0;
  return d;
}

}  // namespace bd
