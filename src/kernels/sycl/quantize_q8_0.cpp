// src/kernels/sycl/quantize_q8_0.cpp - SYCLomatic port of src/kernels/cuda/quantize_act.cu's quantize_q8_0 (P2.S2).
//
// The inverse of `dequant_q8_0`: each 32 FP32 elements packed into ggml's block_q8_0 (`{ fp16 d ; int8 qs[32] }`).
// One work-item per block, a flat range<1> (see dequant_q8_0.cpp for why an nd_range would be rejected on the Arc).
//
// The three subtleties from the source are transcribed, not re-derived:
//   1. `d32 = amax / 127.0f` in FP32; the integers divide by that f32 `d32`, NOT the fp16-rounded value stored.
//   2. The divide is done in DOUBLE and rounded with round-half-to-even (`rint`), matching `blk.astype(f64)/...`
//      in the reference - an f32 divide followed by `rintf` differs from the reference on .5 boundaries.
//   3. The stored scale is the fp16 (`d16`), which is what a reader multiplies by.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

namespace strata::kernels {

namespace {
constexpr int QK8_0 = 32;   // 32 elements per block_q8_0, 34 bytes of storage
}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> b) {
                const float* xb = x + b * QK8_0;
                uint8_t* out = blocks + b * 34;

                float amax = 0.0f;
                for (int i = 0; i < QK8_0; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
                if (amax == 0.0f) {
                    // ggml leaves the block zeroed; writing the fp16 zero explicitly keeps the layout stable.
                    const uint16_t zb = f16_from_f32(0.0f);
                    out[0] = (uint8_t) (zb & 0xFF);
                    out[1] = (uint8_t) (zb >> 8);
                    for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
                    return;
                }
                const float d32 = amax / 127.0f;
                const uint16_t d16bits = f16_from_f32(d32);
                out[0] = (uint8_t) (d16bits & 0xFF);
                out[1] = (uint8_t) (d16bits >> 8);

                for (int i = 0; i < QK8_0; ++i) {
                    double q = std::rint((double) xb[i] / (double) d32);   // double divide, round-half-to-even
                    if (q > 127.0) q = 127.0;
                    if (q < -128.0) q = -128.0;
                    out[2 + i] = (uint8_t) (int8_t) q;
                }
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_0 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
