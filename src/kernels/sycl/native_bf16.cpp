// src/kernels/sycl/native_bf16.cpp - SYCL port of src/kernels/cuda/native_bf16.cu's FP32-activation MMVF (ggml-cuda/
// mmvf.cu).  The CUDA source is one block per output row: each thread strides the paired elements (two bf16 weights,
// two fp32 activations) applying the two ordered multiply-adds of ggml_cuda_mad, then a warp-XOR butterfly plus a
// block reduce.
//
// An earlier port mirrored that shape as TWO submits over a per-(row, lane) `partials` buffer (pass one writes the
// lane partials, pass two sums them).  icpx 2026.1 queues are OUT-OF-ORDER by default, and this function does not
// wait when `stream` is non-null: with no dependency between the two submits the queue is free to run pass two
// before pass one, and `sycl::free(partials)` right after may hand the scratch back while it is still in flight -
// so the reduction could read unwritten or reused memory (the same failure native_gr_norm and
// native_qsa_rms_norm_weighted had).  The reduce is therefore done by ONE work-item per row walking the pairs
// serially - a single kernel with no scratch buffer and no cross-submit dependency.
//
// The two ordered multiply-adds of ggml_cuda_mad are kept exactly (fma(low, x2i, fma(high, x2i+1, acc)), NOT a pair
// sum followed by one add); only the summation order differs from the CUDA warp-XOR + block reduce.  Every product
// is exact in f32 and only the summation order differs - validated by max_rel on Arc.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        reinterpret_cast<uintptr_t>(x) & 7u || reinterpret_cast<uintptr_t>(w) & 3u || reinterpret_cast<uintptr_t>(y) & 3u)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        qref.submit([&](sycl::handler& h) {
            // One work-item per row walks every pair serially (no lane partials, no scratch buffer, so nothing for
            // an out-of-order queue to race on).  The two ordered fmadds match ggml_cuda_mad exactly.
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> oid) {
                const long long o = (long long) oid;
                const uint16_t* row = w + (size_t) o * n_in;
                const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
                float acc = 0.0f;
                for (int64_t pair = 0; pair < n_in / 2; ++pair) {
                    const uint32_t weight = weights2[pair];
                    const float xi = x[(size_t) pair * 2], xj = x[(size_t) pair * 2 + 1];
                    acc = sycl::fma(f32_from_bf16((uint16_t) weight), xi, acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) (weight >> 16)), xj, acc);
                }
                y[o] = acc;
            });
        });
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv_fp32_mmvf launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
