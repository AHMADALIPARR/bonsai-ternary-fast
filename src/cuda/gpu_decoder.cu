// SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
// Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
// CUDA decoder: loads every weight once into one device arena in packed T2
// form (repacked on the GPU from pinned staging), keeps token_embd in mapped
// host memory by default, and runs each decode step as one CUDA graph.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bd/cpu_ops.hpp"
#include "bd/types.hpp"
#include "btf/fast.hpp"
#include "dims.hpp"
#include "kernels.cuh"

namespace btf {

namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

struct Mat {
  const bd::Weight* w = nullptr;
  uint8_t* codes = nullptr;
  uint16_t* scales = nullptr;
  int N = 0, K = 0;
  cuda::Hx hx;
};

struct DLayer {
  bool rec = false;
  Mat gate, up, down, qkv, zg, out, wq, wk, wv, wo;
  float *attn_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
  float *ssm_a = nullptr, *ssm_dt = nullptr, *conv_w = nullptr, *ssm_norm = nullptr;
  void *alpha = nullptr, *beta = nullptr;
  bool ab_bf16 = false;
  uint16_t *kc = nullptr, *vc = nullptr;
  float *conv = nullptr, *ssm = nullptr;
};

class GpuDecoder final : public FastDecoder {
 public:
  GpuDecoder(const bd::Model& m, const FastOptions& o) : m_(m), o_(o), d_(m, o), use_graph_(o.cuda_graph) {
    check_supported(m, o);
    ck(cudaSetDevice(o.device), "set device");
    ck(cudaGetDeviceProperties(&prop_, o.device), "device properties");
    cuda::kernels_init(o.device);
    for (int K : {d_.E, d_.F, d_.DI, d_.H * d_.D})
      if (K > 0 && cuda::gemv_max_nb(K) == 0) throw std::runtime_error("GEMV width too large for shared memory");
    ck(cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking), "stream");
    const auto t0 = std::chrono::steady_clock::now();
    layout(nullptr);
    arena_bytes_ = off_;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "mem info");
    if (arena_bytes_ + (64u << 20) > free_b) {
      char buf[256];
      std::snprintf(buf, sizeof buf, "needs %.3f GB of device memory, %.3f GB free of %.3f GB", arena_bytes_ / 1e9,
                    free_b / 1e9, total_b / 1e9);
      throw std::runtime_error(buf);
    }
    ck(cudaMalloc(&arena_, arena_bytes_), "arena alloc");
    layout(static_cast<char*>(arena_));
    ck(cudaEventCreate(&tev_[0]), "timing event");
    ck(cudaEventCreate(&tev_[1]), "timing event");
    {
      const auto tu = std::chrono::steady_clock::now();
      upload();
      wx_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - tu).count();
    }
    load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  ~GpuDecoder() override {
    if (exec_) cudaGraphExecDestroy(exec_);
    if (graph_) cudaGraphDestroy(graph_);
    for (auto e : tev_)
      if (e) cudaEventDestroy(e);
    if (st_) cudaStreamDestroy(st_);
    if (arena_) cudaFree(arena_);
    if (embd_host_) cudaFreeHost(embd_host_);
    if (pin_) cudaFreeHost(pin_);
  }

  std::string name() const override { return "btf-cuda"; }
  int pos() const override { return pos_; }
  uint64_t device_bytes() const override { return arena_bytes_; }
  std::string info() const override {
    std::ostringstream s;
    s << "btf-cuda on " << prop_.name << " (sm_" << prop_.major << prop_.minor << ", " << prop_.multiProcessorCount
      << " SMs" << (prop_.integrated ? ", integrated" : "") << "), arena " << arena_bytes_ / 1e9 << " GB, embd "
      << (o_.embd_on_device ? "device" : "mapped host") << ", graph " << (exec_ ? "on" : use_graph_ ? "pending" : "off")
      << ", load " << load_s_ << " s (" << wx_bytes_ / 1e9 << " GB of weights copied host-to-device once, " << wx_s_
      << " s)";
    return s.str();
  }

  StepStats stats() const override { return stats_; }
  void reset_stats() override { stats_ = StepStats{}; }

  void reset() override {
    for (auto& S : L_) {
      if (S.rec) {
        ck(cudaMemsetAsync(S.conv, 0, (size_t)(d_.KC - 1) * d_.C * 4, st_), "reset conv");
        ck(cudaMemsetAsync(S.ssm, 0, (size_t)d_.NV * d_.DK * d_.DV * 4, st_), "reset ssm");
      }
    }
    ck(cudaMemsetAsync(d_pos_, 0, sizeof(int), st_), "reset pos");
    ck(cudaStreamSynchronize(st_), "reset sync");
    pos_ = 0;
  }

  int step(int32_t token, float* logits) override {
    if (token < 0 || token >= d_.V) throw std::runtime_error("token id out of range");
    if (pos_ + 1 > d_.n_ctx) throw std::runtime_error("context full");
    const auto tw = std::chrono::steady_clock::now();
    const uint64_t wx0 = wx_bytes_;
    const double wxs0 = wx_s_;
    h_tok_[0] = token;
    if (use_graph_ && !exec_) capture();
    ck(cudaEventRecord(tev_[0], st_), "timing");
    if (exec_)
      ck(cudaGraphLaunch(exec_, st_), "graph launch");
    else
      enqueue_step();
    ck(cudaEventRecord(tev_[1], st_), "timing");
    ck(cudaStreamSynchronize(st_), "step");
    float gms = 0.f;
    ck(cudaEventElapsedTime(&gms, tev_[0], tev_[1]), "timing");
    // Weights are uploaded once in the constructor, so these deltas measure any
    // host-to-device weight traffic a step would add (expected: 0).
    stats_.steps += 1;
    stats_.gpu_s += gms * 1e-3;
    stats_.weight_xfer_bytes += wx_bytes_ - wx0;
    stats_.weight_xfer_s += wx_s_ - wxs0;
    if (!o_.embd_on_device) stats_.mapped_read_bytes += (uint64_t)d_.E / 4 + (uint64_t)d_.E / 128 * 2;
    pos_ += 1;
    if (logits) ck(cudaMemcpy(logits, logits_, (size_t)d_.V * 4, cudaMemcpyDeviceToHost), "logits copy");
    stats_.wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    return *h_next_;
  }

  int prefill(const int32_t* toks, int n, float* logits) override {
    if (n <= 0) throw std::runtime_error("empty prompt");
    if (pos_ + n > d_.n_ctx) throw std::runtime_error("context full");
    for (int i = 0; i < n; ++i)
      if (toks[i] < 0 || toks[i] >= d_.V) throw std::runtime_error("token id out of range");
    for (int i = 0; i < n; i += d_.maxM) {
      const int M = std::min(d_.maxM, n - i);
      std::memcpy(h_tok_, toks + i, M * sizeof(int));
      ck(cudaMemcpyAsync(d_tok_, h_tok_, M * sizeof(int), cudaMemcpyHostToDevice, st_), "tokens");
      forward_chunk(M);
      ck(cudaMemcpyAsync(h_next_, d_next_, sizeof(int), cudaMemcpyDeviceToHost, st_), "next");
      cuda::launch_pos_add(d_pos_, M, st_);
      ck(cudaStreamSynchronize(st_), "prefill");
      pos_ += M;
    }
    if (logits) ck(cudaMemcpy(logits, logits_, (size_t)d_.V * 4, cudaMemcpyDeviceToHost), "logits copy");
    return *h_next_;
  }

 private:
  // ---------------------------------------------------------------- arena
  template <class T>
  T* take(size_t count) {
    off_ = (off_ + 255) & ~(size_t)255;
    T* p = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base_) + off_);
    off_ += count * sizeof(T);
    return p;
  }
  float* signs_of(const std::vector<float>* s) {
    auto it = signs_.find(s);
    if (it != signs_.end()) return it->second;
    float* p = take<float>(s->size());
    signs_[s] = p;
    return p;
  }
  cuda::Hx hx_of(const bd::HadamardXform& h) {
    cuda::Hx x;
    if (!h.active) return x;
    x.block = h.block;
    x.perm_hd = h.perm_hd;
    x.perm_nk = h.perm_nk;
    x.perm_rep = h.perm_rep;
    x.signs = h.signs ? signs_of(h.signs) : nullptr;
    return x;
  }
  void mat(Mat& M, const bd::Weight& w, bool on_device = true) {
    M.w = &w;
    M.N = (int)w.out();
    M.K = (int)w.in();
    if (on_device) {
      M.codes = take<uint8_t>(packed_codes_bytes(M.N, M.K));
      M.scales = take<uint16_t>(packed_scales_bytes(M.N, M.K) / 2);
    }
    M.hx = hx_of(w.hx);
  }
  float* vec(const bd::Weight& w) { return take<float>((size_t)w.t->n_elements()); }

  // Pass 0 (base == nullptr) only measures; pass 1 assigns real pointers.
  void layout(char* base) {
    base_ = base;
    off_ = 0;
    signs_.clear();
    L_.assign(d_.L, DLayer());
    const size_t M = d_.maxM;
    for (int il = 0; il < d_.L; ++il) {
      const bd::LayerWeights& W = m_.layers[il];
      DLayer& S = L_[il];
      S.rec = W.recurrent;
      mat(S.gate, W.ffn_gate);
      mat(S.up, W.ffn_up);
      mat(S.down, W.ffn_down);
      S.attn_norm = vec(W.attn_norm);
      S.post_norm = vec(W.post_norm);
      if (S.rec) {
        mat(S.qkv, W.wqkv);
        mat(S.zg, W.wgate);
        mat(S.out, W.ssm_out);
        S.ssm_a = vec(W.ssm_a);
        S.ssm_dt = vec(W.ssm_dt);
        S.conv_w = vec(W.conv1d);
        S.ssm_norm = vec(W.ssm_norm);
        S.ab_bf16 = W.ssm_alpha.t->type == bd::T_BF16;
        S.alpha = take<uint8_t>(W.ssm_alpha.t->bytes);
        S.beta = take<uint8_t>(W.ssm_beta.t->bytes);
        S.conv = take<float>((size_t)(d_.KC - 1) * d_.C);
        S.ssm = take<float>((size_t)d_.NV * d_.DK * d_.DV);
      } else {
        mat(S.wq, W.wq);
        mat(S.wk, W.wk);
        mat(S.wv, W.wv);
        mat(S.wo, W.wo);
        S.q_norm = vec(W.q_norm);
        S.k_norm = vec(W.k_norm);
        S.kc = take<uint16_t>((size_t)d_.n_ctx * d_.HK * d_.D);
        S.vc = take<uint16_t>((size_t)d_.n_ctx * d_.HK * d_.D);
      }
    }
    mat(output_, m_.output);
    out_norm_ = vec(m_.output_norm);
    mat(embd_, m_.token_embd, o_.embd_on_device);
    embd_hx_ = hx_of(m_.embd_inverse);
    // scratch
    x_ = take<float>(M * d_.E);
    xn_ = take<float>(M * d_.E);
    act_.q = take<int8_t>(M * d_.Kmax);
    act_.d = take<float>(M * (d_.Kmax / 32));
    act_.s = take<int>(M * (d_.Kmax / 32));
    A_ = take<float>(M * std::max(1, d_.w_a()));
    B_ = take<float>(M * std::max(1, d_.w_b()));
    C_ = take<float>(M * std::max(1, d_.w_c()));
    O_ = take<float>(M * std::max(1, d_.w_o()));
    Q_ = take<float>(M * std::max(1, d_.w_q()));
    G_ = take<float>(M * d_.F);
    beta_ = take<float>(M * std::max(1, d_.NV));
    g_ = take<float>(M * std::max(1, d_.NV));
    part_ = take<float>(M * d_.H * d_.nsplit * (d_.D + 2));
    logits_ = take<float>(d_.V);
    argpart_ = take<uint8_t>(cuda::argmax_part_bytes(d_.V));
    d_tok_ = take<int>(16);
    d_pos_ = take<int>(1);
    d_next_ = take<int>(1);
  }

  // ---------------------------------------------------------------- upload
  void upload_f32(float* dst, const bd::Weight& w) {
    const std::vector<float> v = bd::to_f32(w);
    ck(cudaMemcpy(dst, v.data(), v.size() * 4, cudaMemcpyHostToDevice), "upload f32");
    wx_bytes_ += v.size() * 4;
  }

  void upload_mat(const Mat& M) {
    const uint8_t* src = M.w->t->data;
    const int64_t nb_row = M.K / 128;
    const int64_t row_bytes = nb_row * 28;
    const int64_t rows_per = std::max<int64_t>(1, (int64_t)stage_bytes_ / row_bytes);
    for (int64_t r0 = 0; r0 < M.N; r0 += rows_per) {
      const int64_t rows = std::min<int64_t>(rows_per, M.N - r0);
      const int slot = slot_;
      slot_ ^= 1;
      ck(cudaEventSynchronize(ev_[slot]), "staging wait");
      std::memcpy(stage_[slot], src + r0 * row_bytes, rows * row_bytes);
      cuda::launch_repack(stage_dev_[slot], rows * nb_row, M.codes + r0 * (M.K / 4), M.scales + r0 * nb_row, st_);
      ck(cudaEventRecord(ev_[slot], st_), "staging record");
      uploaded_ += rows * row_bytes;
      wx_bytes_ += rows * row_bytes;
    }
  }

  void upload() {
    stage_bytes_ = (size_t)std::max(1, o_.staging_mb) << 20;
    for (int i = 0; i < 2; ++i) {
      ck(cudaHostAlloc(&stage_[i], stage_bytes_, cudaHostAllocMapped), "staging alloc");
      ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&stage_dev_[i]), stage_[i], 0), "staging map");
      ck(cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming), "event");
      ck(cudaEventRecord(ev_[i], st_), "event");
    }
    for (auto& kv : signs_) {
      ck(cudaMemcpy(kv.second, kv.first->data(), kv.first->size() * 4, cudaMemcpyHostToDevice), "signs");
      wx_bytes_ += kv.first->size() * 4;
    }
    for (int il = 0; il < d_.L; ++il) {
      const bd::LayerWeights& W = m_.layers[il];
      DLayer& S = L_[il];
      upload_mat(S.gate);
      upload_mat(S.up);
      upload_mat(S.down);
      upload_f32(S.attn_norm, W.attn_norm);
      upload_f32(S.post_norm, W.post_norm);
      if (S.rec) {
        upload_mat(S.qkv);
        upload_mat(S.zg);
        upload_mat(S.out);
        upload_f32(S.ssm_a, W.ssm_a);
        upload_f32(S.ssm_dt, W.ssm_dt);
        upload_f32(S.conv_w, W.conv1d);
        upload_f32(S.ssm_norm, W.ssm_norm);
        ck(cudaMemcpy(S.alpha, W.ssm_alpha.t->data, W.ssm_alpha.t->bytes, cudaMemcpyHostToDevice), "alpha");
        ck(cudaMemcpy(S.beta, W.ssm_beta.t->data, W.ssm_beta.t->bytes, cudaMemcpyHostToDevice), "beta");
        wx_bytes_ += W.ssm_alpha.t->bytes + W.ssm_beta.t->bytes;
      } else {
        upload_mat(S.wq);
        upload_mat(S.wk);
        upload_mat(S.wv);
        upload_mat(S.wo);
        upload_f32(S.q_norm, W.q_norm);
        upload_f32(S.k_norm, W.k_norm);
      }
    }
    upload_mat(output_);
    upload_f32(out_norm_, m_.output_norm);
    if (o_.embd_on_device) {
      upload_mat(embd_);
    } else {
      // packed on the CPU straight into mapped pinned memory; the embed kernel
      // reads one row (E/4 + E/64 bytes) per token over the bus.
      const size_t cb = packed_codes_bytes(embd_.N, embd_.K), sb = packed_scales_bytes(embd_.N, embd_.K);
      ck(cudaHostAlloc(&embd_host_, cb + sb, cudaHostAllocMapped), "embd host alloc");
      uint8_t* hc = static_cast<uint8_t*>(embd_host_);
      uint16_t* hs = reinterpret_cast<uint16_t*>(hc + cb);
      const int64_t nb = embd_.K / 128;
      bd::ThreadPool pool(o_.threads);
      const uint8_t* src = embd_.w->t->data;
      pool.parallel_for(embd_.N, [&](int64_t b, int64_t e) {
        repack_rows(src + b * nb * 28, e - b, embd_.K, hc + b * (embd_.K / 4), hs + b * nb);
      });
      uint8_t* dev = nullptr;
      ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dev), embd_host_, 0), "embd map");
      embd_.codes = dev;
      embd_.scales = reinterpret_cast<uint16_t*>(dev + cb);
      host_bytes_ = cb + sb;
    }
    ck(cudaStreamSynchronize(st_), "upload");
    for (int i = 0; i < 2; ++i) {
      cudaEventDestroy(ev_[i]);
      cudaFreeHost(stage_[i]);
      stage_[i] = nullptr;
    }
    // zero states / caches / pos
    for (auto& S : L_) {
      if (S.rec) {
        ck(cudaMemset(S.conv, 0, (size_t)(d_.KC - 1) * d_.C * 4), "zero conv");
        ck(cudaMemset(S.ssm, 0, (size_t)d_.NV * d_.DK * d_.DV * 4), "zero ssm");
      } else {
        ck(cudaMemset(S.kc, 0, (size_t)d_.n_ctx * d_.HK * d_.D * 2), "zero kc");
        ck(cudaMemset(S.vc, 0, (size_t)d_.n_ctx * d_.HK * d_.D * 2), "zero vc");
      }
    }
    ck(cudaMemset(d_pos_, 0, sizeof(int)), "zero pos");
    ck(cudaHostAlloc(&pin_, 64 * sizeof(int), cudaHostAllocDefault), "pinned tokens");
    h_tok_ = static_cast<int*>(pin_);
    h_next_ = h_tok_ + 32;
    ck(cudaDeviceSynchronize(), "load");
    if (o_.drop_page_cache && !o_.gguf_path.empty()) {
      const int fd = ::open(o_.gguf_path.c_str(), O_RDONLY);
      if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        ::close(fd);
      }
    }
  }

  // ---------------------------------------------------------------- forward
  void gemv1(const Mat& W, float* y, int ldy, int M, cuda::GemvMode mode) {
    const cuda::GemvSeg s{W.codes, W.scales, W.N, y, ldy};
    cuda::launch_gemv(&s, 1, W.K, M, act_, mode, st_);
  }

  void forward_chunk(int M) {
    const int E = d_.E;
    cuda::launch_embed(embd_.codes, embd_.scales, d_tok_, M, E, embd_hx_, x_, st_);
    for (int il = 0; il < d_.L; ++il) {
      DLayer& S = L_[il];
      if (S.rec)
        gdn_layer(S, M);
      else
        attn_layer(S, M);
      // FFN
      cuda::launch_prep(x_, E, M, E, S.post_norm, d_.eps, S.gate.hx, nullptr, 0, act_, st_);
      const cuda::GemvSeg gu[2] = {{S.gate.codes, S.gate.scales, S.gate.N, G_, d_.F},
                                   {S.up.codes, S.up.scales, S.up.N, G_, d_.F}};
      cuda::launch_gemv(gu, 2, E, M, act_, cuda::kSwiglu, st_);
      cuda::launch_prep(G_, d_.F, M, d_.F, nullptr, d_.eps, S.down.hx, nullptr, 0, act_, st_);
      gemv1(S.down, x_, E, M, cuda::kAdd);
    }
    cuda::launch_prep(x_ + (size_t)(M - 1) * E, E, 1, E, out_norm_, d_.eps, output_.hx, nullptr, 0, act_, st_);
    gemv1(output_, logits_, d_.V, 1, cuda::kStore);
    cuda::launch_argmax(logits_, d_.V, argpart_, d_next_, st_);
  }

  void gdn_layer(DLayer& S, int M) {
    const int E = d_.E;
    cuda::launch_prep(x_, E, M, E, S.attn_norm, d_.eps, S.qkv.hx, xn_, E, act_, st_);
    const cuda::GemvSeg segs[2] = {{S.qkv.codes, S.qkv.scales, S.qkv.N, A_, d_.w_a()},
                                   {S.zg.codes, S.zg.scales, S.zg.N, B_, d_.w_b()}};
    cuda::launch_gemv(segs, 2, E, M, act_, cuda::kStore, st_);
    cuda::launch_ab_gates(S.alpha, S.beta, S.ab_bf16, xn_, E, M, E, d_.NV, S.ssm_dt, S.ssm_a, beta_, g_, st_);
    cuda::launch_conv_silu(A_, d_.w_a(), M, d_.C, S.conv_w, S.conv, C_, d_.w_c(), st_);
    cuda::launch_gdn(C_, d_.w_c(), B_, d_.w_b(), g_, beta_, M, d_.NV, d_.NK, d_.DK, d_.DV, S.ssm_norm, d_.eps, S.ssm,
                     O_, d_.w_o(), st_);
    cuda::launch_prep(O_, d_.w_o(), M, d_.DI, nullptr, d_.eps, S.out.hx, nullptr, 0, act_, st_);
    gemv1(S.out, x_, E, M, cuda::kAdd);
  }

  void attn_layer(DLayer& S, int M) {
    const int E = d_.E, HD = d_.H * d_.D;
    cuda::launch_prep(x_, E, M, E, S.attn_norm, d_.eps, S.wq.hx, nullptr, 0, act_, st_);
    const cuda::GemvSeg segs[3] = {{S.wq.codes, S.wq.scales, S.wq.N, A_, d_.w_a()},
                                   {S.wk.codes, S.wk.scales, S.wk.N, B_, d_.w_b()},
                                   {S.wv.codes, S.wv.scales, S.wv.N, C_, d_.w_c()}};
    cuda::launch_gemv(segs, 3, E, M, act_, cuda::kStore, st_);
    cuda::launch_attn_prep(A_, d_.w_a(), B_, d_.w_b(), C_, d_.w_c(), M, d_.H, d_.HK, d_.D, d_.R, d_.rope_base, d_.eps,
                           S.q_norm, S.k_norm, o_.round_q_f16, d_pos_, Q_, S.kc, S.vc, st_);
    cuda::launch_attention(Q_, S.kc, S.vc, d_pos_, M, d_.H, d_.HK, d_.D, d_.nsplit, d_.chunk_keys(), part_, st_);
    cuda::launch_attn_combine(part_, A_, d_.w_a(), M, d_.H, d_.D, d_.nsplit, O_, d_.w_o(), st_);
    cuda::launch_prep(O_, d_.w_o(), M, HD, nullptr, d_.eps, S.wo.hx, nullptr, 0, act_, st_);
    gemv1(S.wo, x_, E, M, cuda::kAdd);
  }

  void enqueue_step() {
    ck(cudaMemcpyAsync(d_tok_, h_tok_, sizeof(int), cudaMemcpyHostToDevice, st_), "token");
    forward_chunk(1);
    ck(cudaMemcpyAsync(h_next_, d_next_, sizeof(int), cudaMemcpyDeviceToHost, st_), "next");
    cuda::launch_pos_add(d_pos_, 1, st_);
  }

  void capture() {
    bool ok = false;
    if (cudaStreamBeginCapture(st_, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
      try {
        enqueue_step();
        ok = true;
      } catch (const std::exception&) {
        ok = false;
      }
      cudaGraph_t g = nullptr;
      const cudaError_t e = cudaStreamEndCapture(st_, &g);
      if (ok && e == cudaSuccess && g) {
        if (cudaGraphInstantiate(&exec_, g, 0) == cudaSuccess) {
          graph_ = g;
        } else {
          exec_ = nullptr;
          cudaGraphDestroy(g);
        }
      } else if (g) {
        cudaGraphDestroy(g);
      }
    }
    if (!exec_) {
      use_graph_ = false;
      cudaGetLastError();
      if (o_.verbose) std::fprintf(stderr, "btf-cuda: graph capture failed, running eagerly\n");
    }
  }

  const bd::Model& m_;
  FastOptions o_;
  Dims d_;
  cudaDeviceProp prop_{};
  cudaStream_t st_ = nullptr;
  void* arena_ = nullptr;
  size_t arena_bytes_ = 0;
  char* base_ = nullptr;
  size_t off_ = 0;
  std::map<const std::vector<float>*, float*> signs_;
  std::vector<DLayer> L_;
  Mat output_, embd_;
  cuda::Hx embd_hx_;
  float* out_norm_ = nullptr;
  float *x_ = nullptr, *xn_ = nullptr, *A_ = nullptr, *B_ = nullptr, *C_ = nullptr, *O_ = nullptr, *Q_ = nullptr,
        *G_ = nullptr, *beta_ = nullptr, *g_ = nullptr, *part_ = nullptr, *logits_ = nullptr;
  uint8_t* argpart_ = nullptr;
  cuda::Q8Act act_;
  int *d_tok_ = nullptr, *d_pos_ = nullptr, *d_next_ = nullptr;
  void* pin_ = nullptr;
  int *h_tok_ = nullptr, *h_next_ = nullptr;
  void* embd_host_ = nullptr;
  uint64_t host_bytes_ = 0;
  // upload staging
  size_t stage_bytes_ = 0;
  uint8_t* stage_[2] = {nullptr, nullptr};
  uint8_t* stage_dev_[2] = {nullptr, nullptr};
  cudaEvent_t ev_[2] = {nullptr, nullptr};
  int slot_ = 0;
  uint64_t uploaded_ = 0;
  // graph
  bool use_graph_ = true;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  int pos_ = 0;
  double load_s_ = 0;
  // per-step accounting
  cudaEvent_t tev_[2] = {nullptr, nullptr};
  StepStats stats_;
  uint64_t wx_bytes_ = 0;  // all host-to-device weight bytes copied so far
  double wx_s_ = 0;        // wall time spent in weight upload so far

};

}  // namespace

bool gpu_available(std::string* why, int device) {
  int n = 0;
  const cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess || n == 0) {
    if (why) *why = e != cudaSuccess ? std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e) : "no CUDA device";
    cudaGetLastError();
    return false;
  }
  if (device < 0 || device >= n) {
    if (why) *why = "device " + std::to_string(device) + " not present (" + std::to_string(n) + " devices)";
    return false;
  }
  cudaDeviceProp p{};
  if (cudaGetDeviceProperties(&p, device) != cudaSuccess || p.major < 8) {
    if (why) *why = "compute capability " + std::to_string(p.major) + "." + std::to_string(p.minor) + " < 8.0";
    return false;
  }
  return true;
}

std::unique_ptr<FastDecoder> make_gpu_decoder(const bd::Model& m, const FastOptions& o, std::string* why) {
  if (!gpu_available(why, o.device)) return nullptr;
  auto* d = new GpuDecoder(m, o);
  return std::unique_ptr<FastDecoder>(d);
}

}  // namespace btf
