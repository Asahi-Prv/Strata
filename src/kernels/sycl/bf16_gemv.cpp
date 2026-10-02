// src/kernels/sycl/bf16_gemv.cpp - SYCLomatic port of src/kernels/cuda/bf16_gemv.cu's coalesced warp-per-row GEMV.
//
// `y[o] = sum_i f32(x[i]) * f32(w[o*n_in + i])` with both sides bf16-valued so every product is exact in f32.
//
// THE WARP-REDUCTION PROBLEM.  The CUDA version reduces a row with `__shfl_down_sync` warp shuffles.  icpx 2026.1
// exposes none of the SYCL 2020 group-reduction surface - `reduce_over_group`, `this_work_item`/subgroup
// `shfl_down`, and `sycl::local` all fail to compile here - so the row reduction is done in TWO PASSES over a
// per-row partial-sum buffer: each work-item is one lane (i % 32) striding the reduction axis, and a second
// pass sums the 32 lane partials.  This preserves the coalescing the warp kernel exists for (consecutive lanes
// touch consecutive addresses in this layout) - the property that took `gr_read` from 262 ms/token to 8.9 -
// rather than silently regressing to one-thread-per-row, which reads `w[(o+k)*n_in + i]` `n_in * 2` bytes apart
// and costs 32 transactions per load.  The reduction ORDER differs from the warp tree (lane-then-row vs the
// shuffle tree), so it is not bit-identical, but it is exact in each product and only the summation order differs.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
constexpr int WARP = 32;
}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    float* partials = sycl::malloc_device<float>((size_t) n_out * WARP, qref);
    try {
        qref.submit([&](sycl::handler& h) {
            // pass 1: one lane (work-item i % 32) per row strides the reduction axis; coalesced per warp.
            h.parallel_for(sycl::range<1>((size_t) n_out * WARP), [=](sycl::id<1> i) {
                int lane = (int)(i % WARP);
                int o = (int)(i / WARP);
                if (o >= (int) n_out) return;
                float acc = 0.0f;
                for (int64_t j = lane; j < n_in; j += WARP)
                    acc += f32_from_bf16(x[j]) * f32_from_bf16(w[(size_t) o * n_in + j]);
                partials[(size_t) o * WARP + lane] = acc;
            });
        });
        qref.submit([&](sycl::handler& h) {
            // pass 2: one work-item per row sums its 32 lane partials.
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                float s = 0.0f;
                for (int j = 0; j < WARP; ++j) s += partials[(size_t) o * WARP + j];
                y[o] = s;
            });
        });
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv launch: %s\n", e.what());
        std::exit(1);
    }
    sycl::free(partials, qref);
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    // Same two-pass partial-sum strategy as `bf16_gemv`, but `threads_per_row` lanes stride each row, so the
    // reduction is spread across more lanes than the 32 a warp gives.  A non-power-of-two `threads_per_row`
    // cannot be used for a clean lane split, mirroring the CUDA source's refusal.
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two\n", threads_per_row);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    float* partials = sycl::malloc_device<float>((size_t) n_out * threads_per_row, qref);
    try {
        qref.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out * threads_per_row), [=](sycl::id<1> i) {
                int lane = (int)(i % threads_per_row);
                int o = (int)(i / threads_per_row);
                if (o >= (int) n_out) return;
                float acc = 0.0f;
                for (int64_t j = lane; j < n_in; j += threads_per_row)
                    acc += f32_from_bf16(x[j]) * f32_from_bf16(w[(size_t) o * n_in + j]);
                partials[(size_t) o * threads_per_row + lane] = acc;
            });
        });
        qref.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                float s = 0.0f;
                for (int j = 0; j < threads_per_row; ++j) s += partials[(size_t) o * threads_per_row + j];
                y[o] = s;
            });
        });
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv_split launch: %s\n", e.what());
        std::exit(1);
    }
    sycl::free(partials, qref);
}

}  // namespace strata::kernels
