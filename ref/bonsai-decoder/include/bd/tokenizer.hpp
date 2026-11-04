// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// GPT-2 byte-level BPE tokenizer built from the GGUF tokenizer.ggml.* keys,
// with the qwen35 pre-tokenizer (port of prism src/unicode.cpp:779-905) and
// special-token partitioning (llama-vocab.cpp tokenizer_st_partition).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bd/gguf.hpp"

namespace bd {

class Tokenizer {
 public:
  explicit Tokenizer(const Gguf& g);
  // parse_special: text like "<|im_start|>" maps to its special id.
  std::vector<int32_t> encode(const std::string& text, bool parse_special = true) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  std::string piece(int32_t id) const;  // raw UTF-8 bytes of one token
  int32_t find(const std::string& token_text) const;  // -1 if absent
  int32_t n_vocab() const { return (int32_t)tokens_.size(); }
  int32_t eos() const { return eos_; }
  int32_t bos() const { return bos_; }
  bool add_bos() const { return add_bos_; }
  bool is_eog(int32_t id) const;
  const std::vector<int32_t>& eog_ids() const { return eog_; }
  const std::string& pre() const { return pre_; }

  // Exposed for tests.
  static std::vector<std::string> pre_split_qwen35(const std::string& utf8);

 private:
  void bpe_word(const std::string& word_bytes_encoded, std::vector<int32_t>& out) const;
  std::vector<std::string> tokens_;
  std::vector<int32_t> types_;
  std::unordered_map<std::string, int32_t> to_id_;
  std::unordered_map<std::string, int32_t> merge_rank_;
  std::vector<int32_t> specials_;  // sorted by text length desc
  std::vector<int32_t> eog_;
  int32_t eos_ = -1, bos_ = -1, eot_ = -1;
  bool add_bos_ = false;
  std::string pre_;
};

// qwen35 chat template (tokenizer.chat_template in the GGUF) for one user turn
// plus generation prompt. thinking=true: default reasoning effort xhigh system
// turn and "<think>\n" opened; thinking=false: enable_thinking=false path
// ("<think>\n\n</think>\n\n", no reasoning instructions). system: optional
// system message content.
// GPT-2 bytes_to_unicode: the UTF-8 text a raw byte maps to in the vocab.
std::string gpt2_byte_to_unicode(uint8_t b);

std::string chat_prompt_qwen35(const std::string& user, const std::string& system = "", bool thinking = true);

}  // namespace bd
