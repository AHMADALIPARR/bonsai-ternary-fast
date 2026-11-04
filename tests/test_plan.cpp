// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Memory plan: packed sizes add up on a synthetic model, and (when
// BTF_MODEL_GGUF points at the real file) the plan for the real header.
#include <cstdlib>

#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "test_common.hpp"

int main() {
  auto model = btft::make_model(bd::SynthOptions{});
  btf::FastOptions o;
  o.n_ctx = 64;
  const btf::MemoryPlan p = btf::plan_memory(*model, o);
  uint64_t mats = 0, ptq = 0;
  for (const bd::GgufTensor& t : model->gguf->tensors) {
    if (t.type != bd::T_PTQ1_0 || t.name == "token_embd.weight") continue;
    mats += btf::packed_codes_bytes(t.dim(1), t.dim(0)) + btf::packed_scales_bytes(t.dim(1), t.dim(0));
    ptq += t.bytes;
  }
  BD_CHECK(p.items[0].bytes == mats);
  BD_CHECK(mats * 28 == ptq * 34);  // 34 bytes per 128 weights vs 28
  std::printf("synthetic plan:\n%s", p.table().c_str());

  const char* path = std::getenv("BTF_MODEL_GGUF");
  if (!path || !*path) {
    std::printf("BTF_MODEL_GGUF not set: real-model plan skipped\n");
    return 0;
  }
  auto real = bd::Model::load(path);
  btf::FastOptions ro;
  const btf::MemoryPlan rp = btf::plan_memory(*real, ro);
  std::printf("real model plan (n_ctx %d):\n%s", ro.n_ctx, rp.table().c_str());
  BD_CHECK(rp.device_bytes() > 6000000000ull && rp.device_bytes() < 8000000000ull);
  return 0;
}
