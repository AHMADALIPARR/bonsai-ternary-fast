// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// Read-only POSIX mmap (adapted from bonsai-weight-loader).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace bd {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);  // throws std::runtime_error
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  int fd_ = -1;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace bd
