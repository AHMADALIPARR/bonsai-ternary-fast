// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// mmap wrapper (adapted from bonsai-weight-loader/src/mapped_file.cpp).
#include "bd/mapped_file.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace bd {

MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("open " + path + ": " + std::strerror(errno));
  struct stat st;
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("fstat " + path + ": " + std::strerror(errno));
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ == 0) {
    ::close(fd_);
    throw std::runtime_error(path + " is empty");
  }
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    ::close(fd_);
    throw std::runtime_error("mmap " + path + ": " + std::strerror(errno));
  }
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}

}  // namespace bd
