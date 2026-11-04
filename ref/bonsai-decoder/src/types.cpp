// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Type table and CPU dequantizers. Each dequantizer is a port of the matching
// dequantize_row_* in prism ggml/src/ggml-quants.c (branch prism, 62c418c).
#include "bd/types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bd {

TypeInfo type_info(uint32_t t) {
  switch (t) {
    case T_F32: return {"F32", 1, 4, true};
    case T_F16: return {"F16", 1, 2, true};
    case T_BF16: return {"BF16", 1, 2, true};
    case T_I32: return {"I32", 1, 4, false};
    case T_Q4_0: return {"Q4_0", 32, 18, true};
    case T_Q4_1: return {"Q4_1", 32, 20, true};
    case T_Q5_0: return {"Q5_0", 32, 22, true};
    case T_Q5_1: return {"Q5_1", 32, 24, true};
    case T_Q8_0: return {"Q8_0", 32, 34, true};
    case T_Q4_K: return {"Q4_K", 256, 144, true};
    case T_Q5_K: return {"Q5_K", 256, 176, true};
    case T_Q6_K: return {"Q6_K", 256, 210, true};
    case T_TQ1_0: return {"TQ1_0", 256, 54, false};
    case T_TQ2_0: return {"TQ2_0", 256, 66, false};
    case T_Q2_0: return {"Q2_0", 64, 18, true};
    case T_PQ2_0: return {"PQ2_0", 128, 34, true};
    case T_PTQ1_0: return {"PTQ1_0", 128, 28, true};
    default: return {"?", 0, 0, false};
  }
}

uint64_t row_bytes(uint32_t t, int64_t n) {
  TypeInfo i = type_info(t);
  if (!i.block || n % i.block) throw std::runtime_error("row_bytes: bad type/size");
  return (uint64_t)(n / i.block) * (uint64_t)i.bytes;
}

uint16_t f32_to_fp16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
  const uint32_t ax = x & 0x7fffffffu;
  if (ax >= 0x7f800000u) return sign | 0x7c00u | (ax > 0x7f800000u ? 0x200u : 0);
  if (ax >= 0x477ff000u) return sign | 0x7c00u;
  if (ax < 0x38800000u) {
    if (ax < 0x33000000u) return sign;
    const uint32_t e = ax >> 23, m = (ax & 0x7fffffu) | 0x800000u;
    const uint32_t shift = 126 - e;
    uint32_t r = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1))) ++r;
    return sign | (uint16_t)r;
  }
  uint32_t r = ((ax >> 13) - (112u << 10));
  const uint32_t rem = ax & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
  return sign | (uint16_t)r;
}

uint16_t f32_to_bf16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((x >> 16) | 0x40u);
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

namespace {
inline float h16(const uint8_t* p) { return fp16_to_f32((uint16_t)(p[0] | (p[1] << 8))); }

// get_scale_min_k4 (ggml-quants.c)
inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

void deq_ptq1_0(const uint8_t* x, float* y, int64_t k) {
  // ggml-quants.c:2256 dequantize_row_ptq1_0, stages {32,16,8} (ggml-quants.c:2204)
  static const size_t stages[3] = {32, 16, 8};
  const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
  for (int64_t i = 0; i < k / 128; ++i) {
    const uint8_t* b = x + i * 28;
    const float d = h16(b + 26);
    size_t j = 0;
    for (size_t s = 0; s < 3; ++s) {
      const size_t c = stages[s];
      for (; j + c <= 24; j += c)
        for (size_t n = 0; n < 5; ++n)
          for (size_t m = 0; m < c; ++m) {
            uint8_t q = (uint8_t)(b[j + m] * pow3[n]);
            int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
            *y++ = (float)(xi - 1) * d;
          }
    }
    for (size_t n = 0; n < 4; ++n)
      for (size_t h = 0; h < 2; ++h) {
        uint8_t q = (uint8_t)(b[24 + h] * pow3[n]);
        int16_t xi = (int16_t)(((uint16_t)q * 3) >> 8);
        *y++ = (float)(xi - 1) * d;
      }
  }
}

void deq_q2_0(const uint8_t* x, float* y, int64_t k, int qk) {
  // dequantize_row_q2_0 / dequantize_row_pq2_0: d first, 2-bit codes, 00=-1 .. 11=+2
  const int bb = 2 + qk / 4;
  for (int64_t i = 0; i < k / qk; ++i) {
    const uint8_t* b = x + i * bb;
    const float d = h16(b);
    for (int j = 0; j < qk; ++j) {
      const uint8_t q = (b[2 + j / 4] >> ((j % 4) * 2)) & 3;
      y[i * qk + j] = ((int)q - 1) * d;
    }
  }
}
}  // namespace

void dequant_row(uint32_t type, const uint8_t* x, float* y, int64_t k) {
  switch (type) {
    case T_F32: std::memcpy(y, x, (size_t)k * 4); return;
    case T_F16: for (int64_t i = 0; i < k; ++i) y[i] = h16(x + 2 * i); return;
    case T_BF16: for (int64_t i = 0; i < k; ++i) y[i] = bf16_to_f32((uint16_t)(x[2 * i] | (x[2 * i + 1] << 8))); return;
    case T_PTQ1_0: deq_ptq1_0(x, y, k); return;
    case T_PQ2_0: deq_q2_0(x, y, k, 128); return;
    case T_Q2_0: deq_q2_0(x, y, k, 64); return;
    case T_Q8_0:
      for (int64_t i = 0; i < k / 32; ++i) {
        const uint8_t* b = x + i * 34;
        const float d = h16(b);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = (float)(int8_t)b[2 + j] * d;
      }
      return;
    case T_Q4_0:
      for (int64_t i = 0; i < k / 32; ++i) {
        const uint8_t* b = x + i * 18;
        const float d = h16(b);
        for (int j = 0; j < 16; ++j) {
          y[i * 32 + j] = (float)((b[2 + j] & 15) - 8) * d;
          y[i * 32 + j + 16] = (float)((b[2 + j] >> 4) - 8) * d;
        }
      }
      return;
    case T_Q4_1:
      for (int64_t i = 0; i < k / 32; ++i) {
        const uint8_t* b = x + i * 20;
        const float d = h16(b), m = h16(b + 2);
        for (int j = 0; j < 16; ++j) {
          y[i * 32 + j] = (float)(b[4 + j] & 15) * d + m;
          y[i * 32 + j + 16] = (float)(b[4 + j] >> 4) * d + m;
        }
      }
      return;
    case T_Q5_0:
      for (int64_t i = 0; i < k / 32; ++i) {
        const uint8_t* b = x + i * 22;
        const float d = h16(b);
        uint32_t qh;
        std::memcpy(&qh, b + 2, 4);
        for (int j = 0; j < 16; ++j) {
          const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
          const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
          y[i * 32 + j] = (float)(((b[6 + j] & 0x0F) | xh0) - 16) * d;
          y[i * 32 + j + 16] = (float)(((b[6 + j] >> 4) | xh1) - 16) * d;
        }
      }
      return;
    case T_Q5_1:
      for (int64_t i = 0; i < k / 32; ++i) {
        const uint8_t* b = x + i * 24;
        const float d = h16(b), m = h16(b + 2);
        uint32_t qh;
        std::memcpy(&qh, b + 4, 4);
        for (int j = 0; j < 16; ++j) {
          const uint8_t xh0 = ((qh >> (j + 0)) << 4) & 0x10;
          const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
          y[i * 32 + j] = (float)((b[8 + j] & 0x0F) | xh0) * d + m;
          y[i * 32 + j + 16] = (float)((b[8 + j] >> 4) | xh1) * d + m;
        }
      }
      return;
    case T_Q4_K:
      // block_q4_K: d, dmin (fp16), scales[12], qs[128]
      for (int64_t i = 0; i < k / 256; ++i) {
        const uint8_t* b = x + i * 144;
        const float d = h16(b), mn = h16(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* q = b + 16;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
          uint8_t s, m;
          scale_min_k4(is + 0, sc, &s, &m);
          const float d1 = d * s, m1 = mn * m;
          scale_min_k4(is + 1, sc, &s, &m);
          const float d2 = d * s, m2 = mn * m;
          for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
          q += 32;
          is += 2;
        }
      }
      return;
    case T_Q5_K:
      // block_q5_K: d, dmin, scales[12], qh[32], qs[128]
      for (int64_t i = 0; i < k / 256; ++i) {
        const uint8_t* b = x + i * 176;
        const float d = h16(b), mn = h16(b + 2);
        const uint8_t* sc = b + 4;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          uint8_t s, m;
          scale_min_k4(is + 0, sc, &s, &m);
          const float d1 = d * s, m1 = mn * m;
          scale_min_k4(is + 1, sc, &s, &m);
          const float d2 = d * s, m2 = mn * m;
          for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
          ql += 32;
          is += 2;
          u1 <<= 2;
          u2 <<= 2;
        }
      }
      return;
    case T_Q6_K:
      // block_q6_K: ql[128], qh[64], scales[16] (int8), d (fp16)
      for (int64_t i = 0; i < k / 256; ++i) {
        const uint8_t* b = x + i * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t* sc = (const int8_t*)(b + 192);
        const float d = h16(b + 208);
        for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
          }
          y += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
      }
      return;
    default:
      throw std::runtime_error("dequant_row: no dequantizer for " + type_name(type));
  }
}

void quant_row(uint32_t type, const float* x, uint8_t* out, int64_t k) {
  switch (type) {
    case T_F32: std::memcpy(out, x, (size_t)k * 4); return;
    case T_F16:
      for (int64_t i = 0; i < k; ++i) { uint16_t h = f32_to_fp16(x[i]); out[2 * i] = h & 0xff; out[2 * i + 1] = h >> 8; }
      return;
    case T_BF16:
      for (int64_t i = 0; i < k; ++i) { uint16_t h = f32_to_bf16(x[i]); out[2 * i] = h & 0xff; out[2 * i + 1] = h >> 8; }
      return;
    case T_Q8_0:
      // quantize_row_q8_0_ref
      for (int64_t i = 0; i < k / 32; ++i) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[i * 32 + j]));
        const float d = amax / 127.f, id = d ? 1.f / d : 0.f;
        uint16_t h = f32_to_fp16(d);
        uint8_t* b = out + i * 34;
        b[0] = h & 0xff; b[1] = h >> 8;
        for (int j = 0; j < 32; ++j) b[2 + j] = (uint8_t)(int8_t)std::lround(x[i * 32 + j] * id);
      }
      return;
    case T_PTQ1_0: {
      // quantize_row_ptq1_0_ref (ggml-quants.c:2210)
      static const size_t stages[3] = {32, 16, 8};
      for (int64_t i = 0; i < k / 128; ++i) {
        uint8_t* y = out + i * 28;
        float amax = 0.0f;
        for (int j = 0; j < 128; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float d = amax, id = d ? 1.0f / d : 0.0f;
        const uint16_t dh = f32_to_fp16(d);
        y[26] = (uint8_t)(dh & 0xff);
        y[27] = (uint8_t)(dh >> 8);
        size_t j = 0;
        for (size_t s = 0; s < 3; ++s) {
          const size_t c = stages[s];
          for (; j + c <= 24; j += c) {
            for (size_t m = 0; m < c; ++m) {
              uint8_t q = 0;
              for (size_t n = 0; n < 5; ++n) q = (uint8_t)(q * 3 + ((int)std::lround(x[m + n * c] * id) + 1));
              y[j + m] = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
            }
            x += 5 * c;
          }
        }
        for (size_t h = 0; h < 2; ++h) {
          uint8_t q = 0;
          for (size_t m = 0; m < 4; ++m) q = (uint8_t)(q * 3 + ((int)std::lround(x[h + m * 2] * id) + 1));
          q = (uint8_t)(q * 3);
          y[24 + h] = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
        }
        x += 8;
      }
      return;
    }
    default:
      throw std::runtime_error("quant_row: unsupported " + type_name(type));
  }
}

}  // namespace bd
