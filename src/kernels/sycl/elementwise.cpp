// src/kernels/sycl/elementwise.cpp - SYCLomatic port of the simplest glue kernels in src/kernels/cuda/elementwise.cu.
//
// These are the plain elementwise members of the P2.S5 glue set: `add_inplace` (dst += src),
// `scale_inplace` (x *= s) and `silu_inplace` (x / (1 + exp(-x))).  They are one work-item per element, a flat
// range<1>, so this is the same shape as the CUDA `<<<grid_for(n), THREADS>>>` launch with the total work
// `n` spread across `n` work-items instead of across grid*THREADS with a bounds check.  `silu_inplace` keeps
// the source's double-precision intermediate (matching `ref/gdn.py`) rather than letting the compiler do an
// f32 exp.  `gdn_gate` (a softplus) and `f32_to_bf16_bulk` (the bf16 conversion) are the same one-per-element
// shape.
#include "strata/kernels/elementwise.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.  `exp(89)`
// overflows f32 and `exp(20)` is already ~5e8 where `log1p` loses precision, so above 20 the result is `x`.
inline float softplus_dev(float x) { return x > 20.0f ? x : std::log1pf(std::expf(x)); }
}  // namespace

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] += src[i]; });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "add_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] *= s; });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "scale_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
                const double v = (double) x[i];
                x[i] = (float) (v / (1.0 + std::exp(-v)));
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "silu_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
                const int64_t h = (int64_t) i % h_v;
                gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_gate launch: %s\n", e.what());
        std::exit(1);
    }
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = bf16_from_f32(x[i]); });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "f32_to_bf16_bulk launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
