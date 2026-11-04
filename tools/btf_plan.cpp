// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// btf_plan: GPU memory plan of the fast path for a GGUF (no GPU needed), and
// optionally a CPU repack of every PTQ1_0 tensor (timed, verified).
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd/cpu_ops.hpp"
#include "bd/model.hpp"
#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "btf/packed.hpp"
#include "cli.hpp"

static void usage() {
  std::printf(
      "usage: btf_plan -m MODEL.gguf [options]\n"
      "  -c, --ctx N         KV cache tokens (default 4096)\n"
      "      --max-batch N   prompt rows per launch (default 8)\n"
      "      --embd-device   keep token_embd in device memory\n"
      "      --repack        repack every PTQ1_0 tensor on the CPU (timed) and verify it\n"
      "      --verify-all    with --repack: check every element (default: first row of each chunk)\n"
      "      --bw GBPS       also print the bandwidth-bound decode ceiling for GBPS GB/s\n"
      "  -t, --threads N     CPU threads for --repack (default all)\n");
}

int main(int argc, char** argv) {
  std::string path;
  btf::FastOptions o;
  bool do_repack = false, verify_all = false;
  double bw_gbs = 0;
  try {
    cli::Args a{argc, argv};
    for (; a.more(); ++a.i) {
      const std::string s = a.cur();
      if (s == "-m" || s == "--model") path = a.value();
      else if (s == "-c" || s == "--ctx") o.n_ctx = std::atoi(a.value().c_str());
      else if (s == "--max-batch") o.max_batch = std::atoi(a.value().c_str());
      else if (s == "--embd-device") o.embd_on_device = true;
      else if (s == "--repack") do_repack = true;
      else if (s == "--verify-all") verify_all = true;
      else if (s == "--bw") bw_gbs = std::atof(a.value().c_str());
      else if (s == "-t" || s == "--threads") o.threads = std::atoi(a.value().c_str());
      else if (s == "-h" || s == "--help") { usage(); return 0; }
      else throw std::runtime_error("unknown option " + s);
    }
    if (path.empty()) { usage(); return 2; }
    auto m = bd::Model::load(path);
    std::printf("%s\n", m->describe().c_str());
    const btf::MemoryPlan p = btf::plan_memory(*m, o);
    std::printf("memory plan (n_ctx %d, max batch %d, token_embd on %s):\n%s", o.n_ctx, o.max_batch,
                o.embd_on_device ? "device" : "mapped host memory", p.table().c_str());
    uint64_t file_ptq = 0;
    for (const auto& t : m->gguf->tensors) file_ptq += t.type == bd::T_PTQ1_0 ? t.bytes : 0;
    std::printf("PTQ1_0 bytes in the file: %llu (packed T2 is 34/28 = 1.214x of that)\n",
                (unsigned long long)file_ptq);
    std::printf("decode traffic per token (batch 1, minimum): %llu B + %llu B x tokens in context\n",
                (unsigned long long)p.decode_bytes_fixed, (unsigned long long)p.decode_bytes_per_ctx);
    if (bw_gbs > 0)
      std::printf("calculated ceiling at %.1f GB/s: %.2f tok/s at an empty context (bandwidth / bytes, not a measurement)\n",
                  bw_gbs, bw_gbs * 1e9 / (double)p.decode_bytes_fixed);
    if (!do_repack) return 0;

    bd::ThreadPool pool(o.threads);
    const int fd = ::open(path.c_str(), O_RDONLY);
    std::vector<uint8_t> codes;
    std::vector<uint16_t> scales;
    double secs = 0;
    uint64_t in_bytes = 0, checked = 0;
    for (const auto& t : m->gguf->tensors) {
      if (t.type != bd::T_PTQ1_0) continue;
      const int64_t K = t.dim(0), N = t.dim(1), nb = K / 128;
      const int64_t chunk = std::max<int64_t>(1, (int64_t)(64 << 20) / (nb * 28));
      for (int64_t r0 = 0; r0 < N; r0 += chunk) {
        const int64_t rows = std::min(chunk, N - r0);
        codes.resize(btf::packed_codes_bytes(rows, K));
        scales.resize(btf::packed_scales_bytes(rows, K) / 2);
        const uint8_t* src = t.data + (uint64_t)r0 * nb * 28;
        // fault the pages in first so the timing is the repack, not the disk
        volatile uint8_t sink = 0;
        for (uint64_t i = 0; i < (uint64_t)rows * nb * 28; i += 4096) sink ^= src[i];
        (void)sink;
        const auto t0 = std::chrono::steady_clock::now();
        pool.parallel_for(rows, [&](int64_t b, int64_t e) {
          btf::repack_rows(src + (uint64_t)b * nb * 28, e - b, K, codes.data() + b * (K / 4), scales.data() + b * nb);
        });
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        in_bytes += (uint64_t)rows * nb * 28;
        const int64_t vrows = verify_all ? rows : 1;
        for (int64_t r = 0; r < vrows; ++r)
          for (int64_t e = 0; e < K; ++e) {
            const int tr = bd::ptq1_0_trit(src + (uint64_t)r * nb * 28 + (e / 128) * 28, (int)(e % 128));
            if (btf::packed_code(codes.data() + r * (K / 4), e) != tr + 1)
              throw std::runtime_error("repack mismatch in " + t.name);
            ++checked;
          }
      }
      if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    if (fd >= 0) ::close(fd);
    std::printf("CPU repack (%d threads): %.3f GB of PTQ1_0 in %.2f s = %.2f GB/s; %llu elements verified\n",
                pool.size(), in_bytes / 1e9, secs, in_bytes / 1e9 / secs, (unsigned long long)checked);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "btf_plan: %s\n", e.what());
    return 2;
  }
}
