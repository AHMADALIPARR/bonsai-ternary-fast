// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Used when the project is configured without CUDA.
#include "bd/decoder.hpp"

namespace bd {

bool cuda_available(std::string* why) {
  if (why) *why = "built without CUDA (configure with -DBD_CUDA=ON)";
  return false;
}

std::unique_ptr<Decoder> make_cuda_decoder(const Model&, const DecoderOptions&, std::string* why) {
  cuda_available(why);
  return nullptr;
}

}  // namespace bd
