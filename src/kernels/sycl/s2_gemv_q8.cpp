// src/kernels/sycl/s2_gemv_q8.cpp - SYCL port of src/kernels/cuda/s2_gemv_q8.cu (S2 GEMV consuming Q8_0 activations).
//
// Same decode as s2_gemv_quads but the activation is (n_in/32) ggml Q8_0 blocks, 34 bytes each - fp16 scale `d`
// then 32 int8.  A quad of four elements lies inside one 32-block, so one block scale `dx` serves all four, and the
// product is `(code-1)*group_scale * (int8_activation * dx)`.  One work-group per row: lanes stride the quads and
// the lane partials are reduced through a local-accessor buffer with a group barrier (icpx 2026.1 queues are
// out-of-order, so the earlier two-submit partials buffer raced).  Lane 0 sums in lane order, so it matches the
// two-pass form bit-for-bit.
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
        qref.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) threads_per_row), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_out * (size_t) threads_per_row, (size_t) threads_per_row),
                           [=](sycl::nd_item<1> item) {
                const int lane = (int) item.get_local_id(0);
                const long long o = (long long) item.get_group(0);
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
                    a0 += w0 * ((float)(int8_t)blk[2 + off + 0] * dx);
                    a1 += w1 * ((float)(int8_t)blk[2 + off + 1] * dx);
                    a2 += w2 * ((float)(int8_t)blk[2 + off + 2] * dx);
                    a3 += w3 * ((float)(int8_t)blk[2 + off + 3] * dx);
                }
                partials[lane] = (a0 + a1) + (a2 + a3);
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float ss = 0.0f;
                    for (int l = 0; l < threads_per_row; ++l) ss += partials[l];
                    y[o] = ss;
                }
            });
        });
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv_q8 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
