// src/kernels/sycl/native_gr_norm.cpp - SYCL port of src/kernels/cuda/native_gr_norm.cu (ggml-cuda/norm.cu).
//
// Contiguous weighted F32 RMSNorm over n_cols, one block per row in CUDA.  The CUDA kernel does a block reduce of
// sum(value^2) with a warp XOR butterfly followed by a shared-memory block reduction.  icpx 2026.1 exposes none of
// the SYCL 2020 group-reduction surface - reduce_over_group, this_work_item/subgroup shfl_down and sycl::local all
// fail to compile here - so the reduce is done in TWO PASSES over a per-(row, lane) partial buffer: pass one has one
// work-item per lane (i % 32) striding n_cols and accumulating value^2, pass two sums the 32 lane partials into the
// row mean, rsqrts it for the scale and applies scale*input*gamma.  The summation order differs from the warp tree,
// so it is not bit-identical, but every product is exact and only the reduction order differs - validated by max_rel
// on Arc.
#include "strata/kernels/native_gr_norm.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {

namespace {
constexpr int WARP = 32;

void check_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}
}  // namespace

void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                                 float epsilon, void* stream) {
    if (n_cols <= 0 || n_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        float* partials = sycl::malloc_device<float>((size_t) n_rows * WARP, qref);
        qref.submit([&](sycl::handler& h) {
            // pass 1: one work-item per (row, lane) accumulates value^2 over its strided columns.
            h.parallel_for(sycl::range<1>((size_t) n_rows * WARP), [=](sycl::id<1> i) {
                const int lane = (int)(i % WARP);
                const int row = (int)(i / WARP);
                const float* in = input + (size_t) row * n_cols;
                float partial = 0.0f;
                for (int col = lane; col < n_cols; col += WARP) {
                    const float v = in[col];
                    partial += v * v;
                }
                partials[(size_t) row * WARP + lane] = partial;
            });
        });
        qref.submit([&](sycl::handler& h) {
            // pass 2: sum the lane partials, scale each column by rsqrt(mean + epsilon).
            h.parallel_for(sycl::range<1>((size_t) n_rows), [=](sycl::id<1> row) {
                float s = 0.0f;
                for (int l = 0; l < WARP; ++l) s += partials[(size_t) row * WARP + l];
                const float mean = s / n_cols;
                const float scale = 1.0f / sycl::sqrt(mean + epsilon);
                const float* in = input + (size_t) row * n_cols;
                const float* g = gamma + (size_t) row * n_cols;
                float* out = output + (size_t) row * n_cols;
                for (int col = 0; col < n_cols; ++col) out[col] = scale * in[col] * g[col];
            });
        });
        sycl::free(partials, qref);
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gr_rms_norm_weighted launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
