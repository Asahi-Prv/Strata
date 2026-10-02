// src/kernels/sycl/dequant_s2.cpp - SYCLomatic port of src/kernels/cuda/dequant_s2.cu (P2.S2, P3).
//
// The S2 (Q2_0) dequant: the single hottest decode in the engine (31.64 GiB of the artifact is Q2_0).  Still
// naive on purpose - one block per row, no vectors, no shared memory, no unrolling - so this is a clean port:
// the only device work is integer bit-unpack then one multiply, `y[j] = (float)(code - 1) * d`.  The -1 is
// applied to the CODE in the INTEGER domain (not the decoded value), which is exactly what makes it bit-exact,
// and SYCL's float32 multiply matches the CUDA one bit for bit.
//
// Unlike rope.cu this kernel takes NO stream and SYNCHRONIZES itself (it is called on the decode path where the
// result is needed next), so there is always a launch + wait on the default queue.
#include "strata/kernels/dequant_s2.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

constexpr int QK = 64;                     // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;
constexpr int THREADS = 256;

}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());

    try {
        q->submit([&](sycl::handler& h) {
            // One work-item per Q2_0 block, a flat range rather than an nd_range: the Arc runtime rejects an
            // nd_range whose global size is not an exact multiple of the local size, and n_blocks is not
            // (this call site passes 1000).  The CUDA launch's ceil(n/THREADS)-blocks-of-THREADS is one row
            // per block; a flat range keeps that one-work-item-per-row shape exactly.
            h.parallel_for(sycl::range<1>((size_t) n_blocks), [=](sycl::id<1> b) {
                                const float d = scales[b];
                                const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
                                float* y = out + b * QK;
                                for (int j = 0; j < QK; ++j) {
                                    const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
                                    y[j] = (float) (code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
                                }
                            });
        });
        // This kernel syncs itself (no stream to order it), so wait here as the CUDA cudaDeviceSynchronize did.
        q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_s2: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
