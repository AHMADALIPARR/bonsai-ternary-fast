// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Used when the project is configured without CUDA.
#include "btf/fast.hpp"

namespace btf {

bool gpu_available(std::string* why, int) {
  if (why) *why = "built without CUDA (configure with nvcc available and -DBTF_CUDA=ON)";
  return false;
}

std::unique_ptr<FastDecoder> make_gpu_decoder(const bd::Model&, const FastOptions&, std::string* why) {
  gpu_available(why);
  return nullptr;
}

}  // namespace btf
