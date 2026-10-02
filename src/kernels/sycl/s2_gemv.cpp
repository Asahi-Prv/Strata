// src/kernels/sycl/s2_gemv.cpp - SYCLomatic port of src/kernels/cuda/s2_gemv.cu's `s2_gemv`, the S2 GEMV (P2.S2).
//
// One thread per output row: dequantize on the fly (2-bit codes packed 4 per byte, one FP32 scale per 64 elements),
// FP32 accumulation inside the row, no shared memory and no warp ops - the plainest shape in the file, so the port
// is one work-item per row over the total `n_out`, exactly the CUDA `<<<blocks,128>>>` with the rows spread across
// `n_out` work-items.  It is the S2 (CODE_BITS==2) instance of the `s_gemv` kernel above, written here standalone
// because the CUDA source is a standalone file, not the S2/S4/S8 dispatcher.
//
// The fp16 activation is read with `f32_from_f16` (the exact bit converter from f16_bits.hpp, device-callable)
// rather than `__half2float(__ushort_as_half(...))`, which round 193 found producing wrong bits.  The decode is
// `(code - 1)` in the INTEGER domain, then the group scale, then the activation - the same order the CPU reference
// uses, so the two differ only by floating-point contraction.
#include "strata/kernels/s2_gemv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

constexpr int QK = 64;

void s2_gemv_impl(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out) {
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            // One work-item per output row; a flat range<1> rather than an nd_range, because the Arc runtime
            // rejects an nd_range whose global size is not an exact multiple of the local one (see s_gemv.cpp).
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                const long long nb = n_in / QK;
                const uint8_t* c = codes + (size_t) o * nb * (QK / 4);
                const float* s = scales + (size_t) o * nb;

                float acc = 0.0f;
                for (long long b = 0; b < nb; ++b) {
                    const float d = s[b];
                    const uint8_t* cb = c + (size_t) b * (QK / 4);
                    const uint16_t* xb = x + (size_t) b * QK;
                    for (int j = 0; j < QK; ++j) {
                        const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
                        acc += (float) (code - 1) * d * f32_from_f16(xb[j]);
                    }
                }
                y[o] = acc;
            });
        });
        q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s2_gemv launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    s2_gemv_impl(x, codes, scales, y, n_in, n_out);
}

}  // namespace strata::kernels
