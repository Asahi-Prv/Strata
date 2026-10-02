// src/kernels/sycl/native_gr_postops.cpp - SYCL port of src/kernels/cuda/native_gr_postops.cu.
//
// Adapted from llama.cpp 3cf03257 (ggml-cuda/dsv4-hc.cu, scale.cu, unary.cu): contiguous single-token Gated-Ripple
// post-operations on F32 [n_embd, hc].  Three one-thread-per-element kernels, each a plain elementwise map with no
// reduction and no warp ops, so the port is a flat `parallel_for(range<1>(count))` mirroring the CUDA one block of
// THREADS per chunk.  The CUDA intrinsics map to their IEEE / SYCL equivalents: __fmaf_rn -> sycl::fma (single
// rounding FMA), __fmul_rn/__fadd_rn -> plain * + (default round-to-nearest), expf -> sycl::exp, and the pinned
// SCALE(1/hc) + +0-bias scale_zero_bias -> sycl::fma(scale, x, 0.0f) kept verbatim so signed zero is preserved.
#include "strata/kernels/native_gr_postops.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

namespace {

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}

void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}

}  // namespace

void native_gr_down_silu(float* lo, int hc_lr, int hc, void* stream) {
    if (!lo || !stream) throw std::invalid_argument("native_gr_down_silu requires lo and stream");
    check_shape(hc_lr, hc);
    check_pointer(lo);
    const float scale = 1.0f / float(hc);
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) hc_lr), [=](sycl::id<1> idx) {
                // x = scale*lo[i] (pinned +0 bias), then SiLU(x) = x/(1 + exp(-x)).
                const float x = sycl::fma(scale, lo[idx[0]], 0.0f);
                lo[idx[0]] = x / (1.0f + sycl::exp(-x));
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gr_down_silu launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_gr_pre_gated(const float* xn, float* gate, float* mixed, int n_embd, int hc, bool fused_layer, void* stream) {
    if (!xn || !gate || !mixed || !stream) throw std::invalid_argument("native_gr_pre_gated requires pointers and stream");
    check_shape(n_embd, hc);
    check_pointer(xn);
    check_pointer(gate);
    check_pointer(mixed);
    const float scale = 1.0f / float(hc);
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    const bool Fused = fused_layer;
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> idx) {
                const size_t d = idx[0];
                if (d >= (size_t) n_embd) return;
                float sum = 0.0f;
                for (int c = 0; c < hc; ++c) {
                    const size_t i = (size_t) c * n_embd + d;
                    const float x = xn[i];
                    const float w = 1.0f / (1.0f + sycl::exp(-gate[i]));  // sigmoid(gate[i])
                    const float product = x * w;  // __fmul_rn: default round-to-nearest
                    gate[i] = product;  // pin the rounded product
                    if (Fused) sum = sycl::fma(x, w, sum);  // __fmaf_rn, ordered from stream zero
                    else sum = (c == 0) ? product : sum + product;  // __fadd_rn, ordered from stream zero
                }
                if (Fused) mixed[d] = scale * sum; else mixed[d] = sycl::fma(scale, sum, 0.0f);
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gr_pre_gated launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_gr_post(const float* residual, const float* block_out, const float* inject, float* output, int n_embd,
                    int hc, void* stream) {
    if (!residual || !block_out || !inject || !output || !stream)
        throw std::invalid_argument("native_gr_post requires pointers and stream");
    check_shape(n_embd, hc);
    check_pointer(residual);
    check_pointer(block_out);
    check_pointer(inject);
    check_pointer(output);
    const float scale = 1.0f / float(hc);
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_embd * hc), [=](sycl::id<1> idx) {
                const int i = (int) idx[0];
                const int c = i / n_embd, d = i % n_embd;
                // weight = 2 * sigmoid(inject[c] / hc); output = block_out[d]*weight + residual[i] (exact alias ok).
                const float inner = sycl::fma(scale, inject[c], 0.0f);
                const float sig = 1.0f / (1.0f + sycl::exp(-inner));
                const float weight = sycl::fma(2.0f, sig, 0.0f);
                output[i] = sycl::fma(block_out[d], weight, residual[i]);
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gr_post launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
