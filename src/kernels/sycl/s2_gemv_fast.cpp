// src/kernels/sycl/s2_gemv_fast.cpp - SYCL port of src/kernels/cuda/s2_gemv_fast.cu (S2, constant-code-table variant).
//
// The CUDA source is a micro-benchmark that precomputes a constant-memory code LUT and optionally stages `x` in shared
// memory; both were measured neutral-to-negative, and neither changes the arithmetic, so this port matches the numerics
// of s2_gemv_quads (bias-adjusted code times group scale times dequantised activation, packed as (a0+a1)+(a2+a3)) and
// ignores the staging, which is purely a load pattern.  The tree reduction becomes the standard two-pass per-(row, lane)
// partial sum.  Not bit-identical, exact per product.
#include "strata/kernels/s_gemv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
constexpr int QK_S2 = 64;
}  // namespace

void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out,
                  int threads_per_row, bool stage_x) {
    (void)stage_x;  // staging is a load pattern only; the arithmetic is identical to the unstaged path.
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "s2_gemv_fast: threads_per_row %d must be a power of two\n", threads_per_row);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
    try {
        float* partials = sycl::malloc_device<float>((size_t) n_out * threads_per_row, *q);
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out * threads_per_row), [=](sycl::id<1> i) {
                const int lane = (int)(i % threads_per_row);
                const long long o = (long long)(i / threads_per_row);
                if (o >= n_out) return;
                const long long n_quads = n_in / 4;
                const uint8_t* c = codes + o * n_quads;
                const float* s = scales + o * (n_in / QK_S2);
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (long long q = lane; q < n_quads; q += threads_per_row) {
                    const uint8_t byte = c[q];
                    const float d = s[q >> 4];
                    const uint32_t* xw = reinterpret_cast<const uint32_t*>(x + q * 4);
                    const uint16_t lo = (uint16_t)xw[0], hi = (uint16_t)xw[1];
                    const float w0 = (float)((int)(byte & 3) - 1) * d;
                    const float w1 = (float)((int)((byte >> 2) & 3) - 1) * d;
                    const float w2 = (float)((int)((byte >> 4) & 3) - 1) * d;
                    const float w3 = (float)((int)((byte >> 6) & 3) - 1) * d;
                    a0 += w0 * f32_from_f16(lo);
                    a1 += w1 * f32_from_f16((uint16_t)(lo >> 16));
                    a2 += w2 * f32_from_f16(hi);
                    a3 += w3 * f32_from_f16((uint16_t)(hi >> 16));
                }
                partials[(size_t) o * threads_per_row + lane] = (a0 + a1) + (a2 + a3);
            });
        });
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                float s = 0.0f;
                for (int l = 0; l < threads_per_row; ++l) s += partials[(size_t) o * threads_per_row + l];
                y[o] = s;
            });
        });
        q->wait();
        sycl::free(partials, *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_fast launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
