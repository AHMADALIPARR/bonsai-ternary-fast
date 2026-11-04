// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// GPU-friendly packed ternary layout ("T2") and the helpers shared by the CUDA
// kernels and the CPU emulator. docs/packed-layout.md has the full picture.
//
// A PTQ1_0 matrix [N rows][K cols] (K % 128 == 0) becomes two arrays:
//   codes  : uint8 [N][K/4]   2-bit code = trit + 1 in {0,1,2}
//   scales : fp16  [N][K/128] the PTQ1_0 block scale, unchanged
// Inside a row, sub-block s (cols 32s..32s+31) owns bytes [8s, 8s+8) = two
// little-endian uint32 words. Word w holds cols 32s+16w .. +15 and puts col
// 32s + 16w + 4k + j at bits 8j + 2k (k, j in 0..3). So
//   (word >> 2k) & 0x03030303
// is four int8 codes that line up with the int8x4 activation word holding
// cols 32s+16w+4k .. +3, ready for dp4a.
#pragma once
#include <cstdint>
#include <cstring>

#include "bd/types.hpp"

#if defined(__CUDACC__)
#define BTF_UNROLL _Pragma("unroll")
#else
#define BTF_UNROLL
#endif

namespace btf {

constexpr int kBlock = 128;          // weights per PTQ1_0 block (one fp16 scale)
constexpr int kBlockBytes = 28;      // PTQ1_0 block size in the GGUF
constexpr int kSub = 32;             // weights per Q8 activation block / per lane load
constexpr int kPackedBlockBytes = 32;  // codes of one 128-weight block (8 words)

// Byte offset of col e (0..127) inside a block's 32 code bytes and its bit shift.
BD_HD int code_byte(int e) {
  return (e >> 5) * 8 + ((e >> 4) & 1) * 4 + (e & 3);
}
BD_HD int code_shift(int e) { return ((e >> 2) & 3) * 2; }

// Repack one 28-byte PTQ1_0 block into 8 code words + the raw fp16 scale.
// Walks the source bytes (5 trits per byte) the way prism's dequantizer does.
BD_HD uint32_t trit_code(uint32_t b, uint32_t p3) {  // trit + 1 of digit p3 of byte b
  return ((uint32_t)(uint8_t)(b * p3) * 3u) >> 8;
}
BD_HD uint32_t code_bits(uint32_t c, int e) { return c << ((e & 3) * 8 + ((e >> 2) & 3) * 2); }

// Get(i) returns source byte i (0..27) of the block; pack_block_get lets the
// CUDA repack read the block as 7 uint32 registers.
template <class Get>
BD_HD void pack_block_get(const Get& blk, uint32_t words[8], uint16_t* scale) {
  BTF_UNROLL
  for (int i = 0; i < 8; ++i) words[i] = 0;
  // qs[0..15]: elements n*16 + m, n = 0..4 -> word n, bit slot m
  BTF_UNROLL
  for (int m = 0; m < 16; ++m) {
    const uint32_t b = blk(m);
    words[0] |= code_bits(trit_code(b, 1u), m);
    words[1] |= code_bits(trit_code(b, 3u), m);
    words[2] |= code_bits(trit_code(b, 9u), m);
    words[3] |= code_bits(trit_code(b, 27u), m);
    words[4] |= code_bits(trit_code(b, 81u), m);
  }
  // qs[16..23]: elements 80 + n*8 + m
  BTF_UNROLL
  for (int m = 0; m < 8; ++m) {
    const uint32_t b = blk(16 + m);
    words[5] |= code_bits(trit_code(b, 1u), m);        // 80 + m
    words[5] |= code_bits(trit_code(b, 3u), 8 + m);    // 88 + m
    words[6] |= code_bits(trit_code(b, 9u), m);        // 96 + m
    words[6] |= code_bits(trit_code(b, 27u), 8 + m);   // 104 + m
    words[7] |= code_bits(trit_code(b, 81u), m);       // 112 + m
  }
  // qh[0..1]: elements 120 + n*2 + h, n = 0..3 -> word 7, slots 8 + n*2 + h
  BTF_UNROLL
  for (int h = 0; h < 2; ++h) {
    const uint32_t b = blk(24 + h);
    words[7] |= code_bits(trit_code(b, 1u), 8 + h);
    words[7] |= code_bits(trit_code(b, 3u), 10 + h);
    words[7] |= code_bits(trit_code(b, 9u), 12 + h);
    words[7] |= code_bits(trit_code(b, 27u), 14 + h);
  }
  *scale = (uint16_t)(blk(26) | (blk(27) << 8));
}

struct ByteGet {
  const uint8_t* p;
  BD_HD uint32_t operator()(int i) const { return p[i]; }
};
BD_HD void pack_block(const uint8_t* blk, uint32_t words[8], uint16_t* scale) {
  pack_block_get(ByteGet{blk}, words, scale);
}

// Code (0..2) of col e in a packed row.
BD_HD int packed_code(const uint8_t* row_codes, int64_t e) {
  const uint8_t* blk = row_codes + (e / kBlock) * kPackedBlockBytes;
  const int c = (int)(e % kBlock);
  return (blk[code_byte(c)] >> code_shift(c)) & 3;
}

// Signed int8x4 dot product (dp4a semantics) for the host emulator.
inline int dp4a_host(uint32_t a, uint32_t b, int c) {
  for (int i = 0; i < 4; ++i) c += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
  return c;
}

// One sub-block: sum_i q[i] * code[i] for the 32 cols of sub-block s, given
// the two code words and the 8 activation words (int8x4) of that sub-block.
inline int sub_dot_host(uint32_t w0, uint32_t w1, const uint32_t* aw) {
  int acc = 0;
  for (int k = 0; k < 4; ++k) acc = dp4a_host((w0 >> (2 * k)) & 0x03030303u, aw[k], acc);
  for (int k = 0; k < 4; ++k) acc = dp4a_host((w1 >> (2 * k)) & 0x03030303u, aw[4 + k], acc);
  return acc;
}

// ggml quantize_row_q8_0 for one block of 32: q = nearbyint(x * 127/amax),
// d = fp16(amax / 127). Returns sum(q) (used to turn code dots into trit dots).
inline int q8_block_host(const float* x, int8_t* q, float* d) {
  float amax = 0.f;
  for (int i = 0; i < 32; ++i) {
    const float a = x[i] < 0 ? -x[i] : x[i];
    amax = amax > a ? amax : a;
  }
  *d = bd::fp16_to_f32(bd::f32_to_fp16(amax / 127.f));
  const float id = amax != 0.f ? 127.f / amax : 0.f;
  int s = 0;
  for (int i = 0; i < 32; ++i) {
    const float v = x[i] * id;
    // nearbyint, round half to even (default FP environment)
    float r = __builtin_nearbyintf(v);
    q[i] = (int8_t)(int)r;
    s += (int)r;
  }
  return s;
}

}  // namespace btf
