// src/kernels/sycl/native_bf16.cpp - SYCL port of src/kernels/cuda/native_bf16.cu's FP32-activation MMVF (ggml-cuda/
// mmvf.cu).  One block per output row: each thread strides the paired elements (two bf16 weights, two fp32 activations)
// applying the two ordered multiply-adds of ggml_cuda_mad, then a warp-XOR butterfly plus a block reduce.  icpx 2026.1
// has no group-reduction surface, so the warp/block reduce becomes TWO PASSES over a per-(row, lane) partial buffer:
// pass one has one work-item per lane (i % block_size) striding the paired elements, pass two sums the block_size lane
// partials.  Every product is exact in f32 and only the summation order differs - validated by max_rel on Arc.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace strata::kernels {

namespace {
int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}
}  // namespace

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        reinterpret_cast<uintptr_t>(x) & 7u || reinterpret_cast<uintptr_t>(w) & 3u || reinterpret_cast<uintptr_t>(y) & 3u)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    const int block_size = mmvf_block_size(n_in);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        float* partials = sycl::malloc_device<float>((size_t) n_out * block_size, qref);
        qref.submit([&](sycl::handler& h) {
            // pass 1: one work-item per (row, lane) accumulates the ordered multiply-adds over its strided pairs.
            h.parallel_for(sycl::range<1>((size_t) n_out * block_size), [=](sycl::id<1> i) {
                const int lane = (int)(i % block_size);
                const long long o = (long long)(i / block_size);
                if (o >= n_out) return;
                const uint16_t* row = w + (size_t) o * n_in;
                const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
                float acc = 0.0f;
                for (int pair = lane; pair < n_in / 2; pair += block_size) {
                    const uint32_t weight = weights2[pair];
                    const float xi = x[(size_t) pair * 2], xj = x[(size_t) pair * 2 + 1];
                    acc = sycl::fma(f32_from_bf16((uint16_t)weight), xi, acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t)(weight >> 16)), xj, acc);
                }
                partials[(size_t) o * block_size + lane] = acc;
            });
        });
        qref.submit([&](sycl::handler& h) {
            // pass 2: one work-item per row sums its block_size lane partials.
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                float s = 0.0f;
                for (int l = 0; l < block_size; ++l) s += partials[(size_t) o * block_size + l];
                y[o] = s;
            });
        });
        if (stream == nullptr) qref.wait();
        sycl::free(partials, qref);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv_fp32_mmvf launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
