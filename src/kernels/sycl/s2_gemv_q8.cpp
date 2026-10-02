// src/kernels/sycl/s2_gemv_q8.cpp - SYCL port of src/kernels/cuda/s2_gemv_q8.cu (S2 GEMV consuming Q8_0 activations).
//
// Same decode as s2_gemv_quads but the activation is (n_in/32) ggml Q8_0 blocks, 34 bytes each - fp16 scale `d`
// then 32 int8.  A quad of four elements lies inside one 32-block, so one block scale `dx` serves all four, and the
// product is `(code-1)*group_scale * (int8_activation * dx)`.  One block per row, tree reduction -> two-pass per
// (row, lane) partials summed by pass two.  Not bit-identical (reduction order), but exact per product.
#include "strata/kernels/s2_gemv_q8.hpp"

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
constexpr int QK8_0 = 32;
}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out,
                int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        float* partials = sycl::malloc_device<float>((size_t) n_out * threads_per_row, qref);
        qref.submit([&](sycl::handler& h) {
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
                    const long long ablk = (q * 4) / QK8_0;
                    const uint8_t* blk = act + ablk * 34;
                    const uint16_t dbits = (uint16_t)(blk[0] | (blk[1] << 8));
                    const float dx = f32_from_f16(dbits);
                    const int off = (int)((q * 4) % QK8_0);
                    const float w0 = (float)((int)(byte & 3) - 1) * d;
                    const float w1 = (float)((int)((byte >> 2) & 3) - 1) * d;
                    const float w2 = (float)((int)((byte >> 4) & 3) - 1) * d;
                    const float w3 = (float)((int)((byte >> 6) & 3) - 1) * d;
                    a0 += w0 * ((float)blk[2 + off + 0] * dx);
                    a1 += w1 * ((float)blk[2 + off + 1] * dx);
                    a2 += w2 * ((float)blk[2 + off + 2] * dx);
                    a3 += w3 * ((float)blk[2 + off + 3] * dx);
                }
                partials[(size_t) o * threads_per_row + lane] = (a0 + a1) + (a2 + a3);
            });
        });
        qref.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                float s = 0.0f;
                for (int l = 0; l < threads_per_row; ++l) s += partials[(size_t) o * threads_per_row + l];
                y[o] = s;
            });
        });
        if (stream == nullptr) qref.wait();
        sycl::free(partials, qref);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_q8 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
