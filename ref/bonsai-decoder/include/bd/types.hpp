// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// ggml tensor type ids (prism ggml/include/ggml.h enum ggml_type), block
// geometry, fp16/bf16 conversion and CPU row dequantizers.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

#if defined(__CUDACC__)
#define BD_HD __host__ __device__ __forceinline__
#else
#define BD_HD inline
#endif

namespace bd {

enum GgmlType : uint32_t {
  T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8,
  T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_I32 = 26, T_BF16 = 30,
  T_TQ1_0 = 34, T_TQ2_0 = 35, T_Q2_0 = 42, T_PQ2_0 = 142, T_PTQ1_0 = 143,
};

struct TypeInfo {
  const char* name;
  int block;   // elements per block
  int bytes;   // bytes per block
  bool dequant;  // a CPU dequantizer exists in this project
};
TypeInfo type_info(uint32_t t);  // {"?",0,0,false} when unknown
inline std::string type_name(uint32_t t) {
  TypeInfo i = type_info(t);
  return i.block ? i.name : "type_" + std::to_string(t);
}
uint64_t row_bytes(uint32_t t, int64_t n);  // bytes of n elements (n % block == 0)

BD_HD float fp16_to_f32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400u) == 0) { man <<= 1; --exp; }
      man &= 0x3ffu;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
uint16_t f32_to_fp16(float f);  // round to nearest even
BD_HD float bf16_to_f32(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
uint16_t f32_to_bf16(float f);

// PTQ1_0 element e (0..127) of one 28-byte block, as {-1,0,1}. Same order as
// prism dequantize_row_ptq1_0 (ggml-quants.c:2256): qs[0..15] hold elements
// n*16+m, qs[16..23] hold 80+n*8+m, qh[0..1] hold 120+n*2+h.
BD_HD int ptq1_0_trit(const uint8_t* blk, int e) {
  int bi, n;
  if (e < 80) { bi = e & 15; n = e >> 4; }
  else if (e < 120) { const int f = e - 80; bi = 16 + (f & 7); n = f >> 3; }
  else { const int f = e - 120; bi = 24 + (f & 1); n = f >> 1; }
  const uint32_t p3 = n == 0 ? 1u : n == 1 ? 3u : n == 2 ? 9u : n == 3 ? 27u : 81u;
  const uint32_t q = (uint32_t)(uint8_t)(blk[bi] * p3);
  return (int)((q * 3u) >> 8) - 1;
}

// Dequantize n elements (one row or part of one, block aligned) to fp32.
void dequant_row(uint32_t type, const uint8_t* src, float* dst, int64_t n);

// Quantizers used by the synthetic model generator (ports of the prism *_ref
// quantizers for PTQ1_0, Q8_0 and F16/BF16).
void quant_row(uint32_t type, const float* src, uint8_t* dst, int64_t n);

}  // namespace bd
