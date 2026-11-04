// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
#include "bd/tokenizer.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace bd {

namespace {

// --- UTF-8 -------------------------------------------------------------------
std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
  std::vector<uint32_t> out;
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = (unsigned char)s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c >> 5) == 6) { cp = c & 0x1f; n = 2; }
    else if ((c >> 4) == 14) { cp = c & 0x0f; n = 3; }
    else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
    else { cp = 0xFFFD; n = 1; }
    if (i + n > s.size()) { out.push_back(0xFFFD); ++i; continue; }
    bool ok = true;
    for (int k = 1; k < n; ++k) {
      const unsigned char d = (unsigned char)s[i + k];
      if ((d >> 6) != 2) { ok = false; break; }
      cp = (cp << 6) | (d & 0x3f);
    }
    if (!ok) { out.push_back(0xFFFD); ++i; continue; }
    out.push_back(cp);
    i += n;
  }
  return out;
}

std::string cpt_to_utf8(uint32_t cp) {
  std::string r;
  if (cp < 0x80) r += (char)cp;
  else if (cp < 0x800) { r += (char)(0xC0 | (cp >> 6)); r += (char)(0x80 | (cp & 0x3f)); }
  else if (cp < 0x10000) { r += (char)(0xE0 | (cp >> 12)); r += (char)(0x80 | ((cp >> 6) & 0x3f)); r += (char)(0x80 | (cp & 0x3f)); }
  else { r += (char)(0xF0 | (cp >> 18)); r += (char)(0x80 | ((cp >> 12) & 0x3f)); r += (char)(0x80 | ((cp >> 6) & 0x3f)); r += (char)(0x80 | (cp & 0x3f)); }
  return r;
}

// --- Unicode categories ---------------------------------------------------------
// Exact for ASCII and Latin-1. Outside that, range tables for the major
// scripts (documented limitation in docs/tokenizer.md): llama.cpp uses the
// full generated tables in src/unicode-data.cpp.
struct Flags {
  bool letter = false, number = false, mark = false, space = false, any = false;
};

bool in(uint32_t c, uint32_t a, uint32_t b) { return c >= a && c <= b; }

Flags flags_of(uint32_t c) {
  Flags f;
  if (c == 0xFFFFFFFFu) return f;
  // whitespace (unicode White_Space)
  if (in(c, 0x09, 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || in(c, 0x2000, 0x200A) ||
      c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000) {
    f.space = f.any = true;
    return f;
  }
  if (c < 0x80) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) f.letter = true;
    else if (c >= '0' && c <= '9') f.number = true;
    f.any = c >= 0x20 && c != 0x7F ? true : (f.letter || f.number);  // controls: Cc -> flag set in llama (is_control)
    if (c < 0x20 || c == 0x7F) f.any = true;  // llama marks Cc as control (non-zero flags)
    return f;
  }
  if (c < 0xA0) { f.any = true; return f; }  // C1 controls
  if (c < 0x100) {
    if (c == 0xAA || c == 0xB5 || c == 0xBA || (in(c, 0xC0, 0xFF) && c != 0xD7 && c != 0xF7)) f.letter = true;
    else if (c == 0xB2 || c == 0xB3 || c == 0xB9 || in(c, 0xBC, 0xBE)) f.number = true;
    f.any = true;
    return f;
  }
  f.any = true;
  // combining marks
  if (in(c, 0x0300, 0x036F) || in(c, 0x0483, 0x0489) || in(c, 0x0591, 0x05BD) || in(c, 0x0610, 0x061A) ||
      in(c, 0x064B, 0x065F) || c == 0x0670 || in(c, 0x06D6, 0x06DC) || in(c, 0x0900, 0x0903) || in(c, 0x093A, 0x094F) ||
      in(c, 0x0951, 0x0957) || in(c, 0x0E31, 0x0E31) || in(c, 0x0E34, 0x0E3A) || in(c, 0x0E47, 0x0E4E) ||
      in(c, 0x1AB0, 0x1AFF) || in(c, 0x1DC0, 0x1DFF) || in(c, 0x20D0, 0x20FF) || in(c, 0x302A, 0x302F) ||
      in(c, 0x3099, 0x309A) || in(c, 0xFE00, 0xFE0F) || in(c, 0xFE20, 0xFE2F) || in(c, 0xE0100, 0xE01EF)) {
    f.mark = true;
    return f;
  }
  // numbers
  if (in(c, 0x0660, 0x0669) || in(c, 0x06F0, 0x06F9) || in(c, 0x0966, 0x096F) || in(c, 0x0E50, 0x0E59) ||
      in(c, 0x2070, 0x2079) || in(c, 0x2080, 0x2089) || in(c, 0x2150, 0x2189) || in(c, 0x2460, 0x249B) ||
      in(c, 0x3007, 0x3007) || in(c, 0x3021, 0x3029) || in(c, 0xFF10, 0xFF19)) {
    f.number = true;
    return f;
  }
  // punctuation / symbols blocks that are not letters
  if (in(c, 0x2000, 0x2BFF) || in(c, 0x3000, 0x3006) || in(c, 0x3008, 0x3020) || in(c, 0x3030, 0x303F) ||
      in(c, 0xFE30, 0xFE4F) || in(c, 0xFF00, 0xFF0F) || in(c, 0xFF1A, 0xFF20) || in(c, 0xFF3B, 0xFF40) ||
      in(c, 0xFF5B, 0xFF65) || in(c, 0x1F000, 0x1FAFF) || in(c, 0xE000, 0xF8FF) || in(c, 0xD800, 0xDFFF) ||
      c == 0x037E || c == 0x0387 || in(c, 0x055A, 0x055F) || c == 0x0589 || in(c, 0x05BE, 0x05C7) ||
      in(c, 0x060C, 0x060D) || c == 0x061B || in(c, 0x061E, 0x061F) || in(c, 0x066A, 0x066D) || c == 0x06D4 ||
      in(c, 0x0964, 0x0965) || c == 0x0E3F || in(c, 0x3200, 0x33FF)) {
    return f;
  }
  // everything else in the BMP/SMP letter-bearing scripts counts as a letter
  if (in(c, 0x0100, 0x02FF) || in(c, 0x0370, 0x1FFF) || in(c, 0x2C00, 0x2DFF) || in(c, 0x3040, 0x31FF) ||
      in(c, 0x3400, 0x4DBF) || in(c, 0x4E00, 0x9FFF) || in(c, 0xA000, 0xD7FF) || in(c, 0xF900, 0xFAFF) ||
      in(c, 0xFB00, 0xFDFF) || in(c, 0xFE70, 0xFEFF) || in(c, 0xFF21, 0xFF3A) || in(c, 0xFF41, 0xFF5A) ||
      in(c, 0xFF66, 0xFFDC) || in(c, 0x10000, 0x1EFFF) || in(c, 0x20000, 0x3FFFF)) {
    f.letter = true;
    return f;
  }
  return f;
}

uint32_t to_lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// --- GPT-2 bytes_to_unicode -------------------------------------------------------
struct ByteMap {
  std::string enc[256];
  std::unordered_map<uint32_t, uint8_t> dec;
  ByteMap() {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      uint32_t cp;
      if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
        cp = (uint32_t)b;
      else
        cp = 256 + n++;
      enc[b] = cpt_to_utf8(cp);
      dec[cp] = (uint8_t)b;
    }
  }
};
const ByteMap& bytemap() {
  static ByteMap m;
  return m;
}

}  // namespace

// Port of unicode_regex_split_custom_qwen35 (prism src/unicode.cpp:779-905)
std::vector<std::string> Tokenizer::pre_split_qwen35(const std::string& text) {
  const auto cpts = utf8_to_cpts(text);
  const size_t end_all = cpts.size();
  const uint32_t OOR = 0xFFFFFFFFu;
  auto cpt_at = [&](size_t p) -> uint32_t { return p < end_all ? cpts[p] : OOR; };
  auto fl = [&](size_t p) -> Flags { return p < end_all ? flags_of(cpts[p]) : Flags{}; };
  std::vector<std::string> out;
  size_t prev = 0;
  auto add = [&](size_t e) -> size_t {
    size_t len = e - prev;
    if (len > 0) {
      std::string s;
      for (size_t i = prev; i < e; ++i) s += cpt_to_utf8(cpts[i]);
      out.push_back(std::move(s));
    }
    prev = e;
    return len;
  };
  for (size_t pos = 0; pos < end_all;) {
    const uint32_t cpt = cpt_at(pos);
    const Flags f = fl(pos);
    if (cpt == '\'' && pos + 1 < end_all) {
      const uint32_t n1 = to_lower_ascii(cpt_at(pos + 1));
      if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { pos += add(pos + 2); continue; }
      if (pos + 2 < end_all) {
        const uint32_t n2 = to_lower_ascii(cpt_at(pos + 2));
        if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) { pos += add(pos + 3); continue; }
      }
    }
    if (!(cpt == '\r' || cpt == '\n' || f.number)) {
      if (f.letter || f.mark || fl(pos + 1).mark || fl(pos + 1).letter) {
        pos++;
        while (fl(pos).letter || fl(pos).mark) pos++;
        add(pos);
        continue;
      }
    }
    if (f.number) { pos++; add(pos); continue; }
    Flags f2 = cpt == ' ' ? fl(pos + 1) : f;
    if (!(f2.space || f2.letter || f2.mark || f2.number) && f.any) {
      pos += (cpt == ' ');
      while (!(f2.space || f2.letter || f2.mark || f2.number) && f2.any) f2 = fl(++pos);
      uint32_t c2 = cpt_at(pos);
      while (c2 == '\r' || c2 == '\n') c2 = cpt_at(++pos);
      add(pos);
      continue;
    }
    size_t nws = 0, last_rn = 0;
    while (fl(pos + nws).space) {
      const uint32_t c2 = cpt_at(pos + nws);
      if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
      nws++;
    }
    if (last_rn > 0) { pos = last_rn; add(pos); continue; }
    if (nws > 1 && cpt_at(pos + nws) != OOR) { pos += nws - 1; add(pos); continue; }
    if (nws > 0) { pos += nws; add(pos); continue; }
    add(++pos);
  }
  return out;
}

Tokenizer::Tokenizer(const Gguf& g) {
  const std::string model = g.get_s("tokenizer.ggml.model", "");
  if (model != "gpt2") throw std::runtime_error("tokenizer: only gpt2 (byte-level BPE) supported, got '" + model + "'");
  pre_ = g.get_s("tokenizer.ggml.pre", "");
  auto it = g.kv.find("tokenizer.ggml.tokens");
  if (it == g.kv.end()) throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
  tokens_ = it->second.as;
  auto tt = g.kv.find("tokenizer.ggml.token_type");
  types_.assign(tokens_.size(), 1);
  if (tt != g.kv.end())
    for (size_t i = 0; i < tokens_.size() && i < tt->second.ai.size(); ++i) types_[i] = (int32_t)tt->second.ai[i];
  for (size_t i = 0; i < tokens_.size(); ++i) to_id_.emplace(tokens_[i], (int32_t)i);
  auto mg = g.kv.find("tokenizer.ggml.merges");
  if (mg != g.kv.end())
    for (size_t i = 0; i < mg->second.as.size(); ++i) merge_rank_.emplace(mg->second.as[i], (int32_t)i);
  eos_ = (int32_t)g.get_i("tokenizer.ggml.eos_token_id", -1);
  bos_ = (int32_t)g.get_i("tokenizer.ggml.bos_token_id", -1);
  eot_ = (int32_t)g.get_i("tokenizer.ggml.eot_token_id", -1);
  add_bos_ = g.get_b("tokenizer.ggml.add_bos_token", false);
  // special tokens: every non-normal token type (control=3, user_defined=4, unknown=2)
  for (size_t i = 0; i < tokens_.size(); ++i)
    if (types_[i] != 1 && types_[i] != 6 && !tokens_[i].empty()) specials_.push_back((int32_t)i);
  std::stable_sort(specials_.begin(), specials_.end(),
                   [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });
  if (eos_ >= 0) eog_.push_back(eos_);
  if (eot_ >= 0) eog_.push_back(eot_);
  // prism src/llama-vocab.cpp:3237-3320: FIM pad/repo/sep, then tokens matched by
  // text, then eos/eot. Only the names that can occur in a Qwen-style vocab are
  // listed; bonsai_golden checks the whole set against prism.
  for (const char* s : {"<|fim_pad|>", "<|fim_repo|>", "<|repo_name|>", "<|file_sep|>", "<|eot_id|>", "<|eot|>",
                        "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>",
                        "<|end_of_text|>", "<eos>"}) {
    const int32_t id = find(s);
    if (id >= 0 && !is_eog(id)) eog_.push_back(id);
  }
}

int32_t Tokenizer::find(const std::string& t) const {
  auto it = to_id_.find(t);
  return it == to_id_.end() ? -1 : it->second;
}

bool Tokenizer::is_eog(int32_t id) const { return std::find(eog_.begin(), eog_.end(), id) != eog_.end(); }

void Tokenizer::bpe_word(const std::string& w, std::vector<int32_t>& out) const {
  // symbols = utf-8 chars of the byte-encoded word (llama-vocab.cpp:718-728)
  struct Sym { size_t off, n; int prev, next; };
  std::vector<Sym> sy;
  for (size_t off = 0; off < w.size();) {
    const unsigned char c = (unsigned char)w[off];
    size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    n = std::min(n, w.size() - off);
    sy.push_back({off, n, (int)sy.size() - 1, -1});
    off += n;
    if (off < w.size()) sy.back().next = (int)sy.size();
  }
  struct Big { int rank, left, right; std::string text; };
  auto cmp = [](const Big& a, const Big& b) { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); };
  std::priority_queue<Big, std::vector<Big>, decltype(cmp)> q(cmp);
  auto txt = [&](int i) { return w.substr(sy[i].off, sy[i].n); };
  auto add_big = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    const std::string a = txt(l), b = txt(r);
    auto it = merge_rank_.find(a + " " + b);
    if (it == merge_rank_.end()) return;
    q.push({it->second, l, r, a + b});
  };
  for (int i = 1; i < (int)sy.size(); ++i) add_big(i - 1, i);
  while (!q.empty()) {
    Big b = q.top();
    q.pop();
    Sym& L = sy[b.left];
    Sym& R = sy[b.right];
    if (L.n == 0 || R.n == 0) continue;
    if (txt(b.left) + txt(b.right) != b.text) continue;
    L.n += R.n;
    R.n = 0;
    L.next = R.next;
    if (R.next >= 0) sy[R.next].prev = b.left;
    add_big(L.prev, b.left);
    add_big(b.left, L.next);
  }
  for (int i = 0; i != -1 && i < (int)sy.size(); i = sy[i].next) {
    if (sy[i].n == 0) continue;
    const std::string s = txt(i);
    const int32_t id = find(s);
    if (id >= 0) { out.push_back(id); continue; }
    // byte fallback: each byte-encoded char separately (llama-vocab.cpp:790-805)
    for (size_t k = 0; k < s.size();) {
      const unsigned char c = (unsigned char)s[k];
      size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
      const int32_t bid = find(s.substr(k, n));
      if (bid >= 0) out.push_back(bid);
      k += n;
    }
  }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
  std::vector<int32_t> out;
  if (add_bos_ && bos_ >= 0) out.push_back(bos_);
  // partition by special tokens (longest first, leftmost occurrence)
  struct Frag { bool special; int32_t id; std::string s; };
  std::vector<Frag> frags{{false, -1, text}};
  if (parse_special) {
    for (int32_t sid : specials_) {
      const std::string& st = tokens_[sid];
      std::vector<Frag> next;
      for (auto& fr : frags) {
        if (fr.special) { next.push_back(fr); continue; }
        size_t p = 0;
        while (true) {
          const size_t hit = fr.s.find(st, p);
          if (hit == std::string::npos) {
            if (p < fr.s.size()) next.push_back({false, -1, fr.s.substr(p)});
            break;
          }
          if (hit > p) next.push_back({false, -1, fr.s.substr(p, hit - p)});
          next.push_back({true, sid, ""});
          p = hit + st.size();
        }
      }
      frags.swap(next);
    }
  }
  const ByteMap& bm = bytemap();
  for (auto& fr : frags) {
    if (fr.special) { out.push_back(fr.id); continue; }
    for (const std::string& word : pre_split_qwen35(fr.s)) {
      std::string enc;
      for (unsigned char c : word) enc += bm.enc[c];
      bpe_word(enc, out);
    }
  }
  return out;
}

std::string Tokenizer::piece(int32_t id) const {
  if (id < 0 || id >= (int32_t)tokens_.size()) return "";
  const std::string& t = tokens_[id];
  if (types_[id] != 1) return t;  // control / user defined: literal text
  const ByteMap& bm = bytemap();
  std::string r;
  for (uint32_t cp : utf8_to_cpts(t)) {
    auto it = bm.dec.find(cp);
    if (it != bm.dec.end()) r += (char)it->second;
    else r += cpt_to_utf8(cp);
  }
  return r;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string r;
  for (int32_t id : ids) r += piece(id);
  return r;
}

std::string gpt2_byte_to_unicode(uint8_t b) { return bytemap().enc[b]; }

std::string chat_prompt_qwen35(const std::string& user, const std::string& system, bool thinking) {
  // chat_template lines 45-56 (reasoning instructions), 76-87 (system turn),
  // 106 (user turn), 163-170 (generation prompt). Contents are |trim'd.
  static const char* kXhigh =
      "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
      "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
  auto trim = [](const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return std::string();
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
  };
  const std::string ri = thinking ? kXhigh : "";
  const std::string sys = trim(system);
  std::string p;
  if (!sys.empty())
    p += "<|im_start|>system\n" + (ri.empty() ? std::string() : ri + "\n\n") + sys + "<|im_end|>\n";
  else if (!ri.empty())
    p += "<|im_start|>system\n" + ri + "<|im_end|>\n";
  p += "<|im_start|>user\n" + trim(user) + "<|im_end|>\n";
  p += "<|im_start|>assistant\n";
  p += thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
  return p;
}

}  // namespace bd
