// src/kernels/sycl/bf16_gemv.cpp - SYCLomatic port of src/kernels/cuda/bf16_gemv.cu's coalesced warp-per-row GEMV.
//
// `y[o] = sum_i f32(x[i]) * f32(w[o*n_in + i])` with both sides bf16-valued so every product is exact in f32.
//
// THE WARP-REDUCTION.  The CUDA version reduces a row with `__shfl_down_sync` warp shuffles.  icpx 2026.1 has
// no subgroup-shuffle surface, so the row reduction uses a work-group: one work-group per row, one lane
// (`i % threads_per_row`) per work-item striding the reduction axis, and the lane partials reduced through a
// local-accessor buffer with a group barrier.  This preserves the coalescing the warp kernel exists for
// (consecutive lanes touch consecutive addresses in this layout) - the property that took `gr_read` from
// 262 ms/token to 8.9 - and is a SINGLE kernel with no malloc_device scratch.
//
// WHY NOT TWO PASSES.  An earlier port reduced in two separate submits over a per-row `partials` buffer.  icpx
// 2026.1 queues are OUT-OF-ORDER by default: with no dependency between the submits the second pass could run
// before the first, and `sycl::free(partials)` right after could hand the scratch back while it was still in
// flight (the same failure native_bf16_fp32_mmvf, native_gr_norm and native_qsa_rms_norm_weighted had).  The
// work-group keeps the identical lane partials and lane-0 accumulates them in the same lane order, so the result
// is bit-identical to the two-pass form while avoiding the race.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
constexpr int WARP = 32;

// One work-group per row: `lane` work-items stride the reduction axis coalesced; the lane partials are summed
// in lane order by lane 0 (same order the two-pass pass two used, hence bit-identical).
void gemv_rows(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int lane_count,
               sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) lane_count), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * (size_t) lane_count, (size_t) lane_count),
                       [=](sycl::nd_item<1> item) {
                           const int lane = (int) item.get_local_id(0);
                           const long long o = (long long) item.get_group(0);
                           float acc = 0.0f;
                           for (int64_t j = lane; j < n_in; j += lane_count)
                               acc += f32_from_bf16(x[j]) * f32_from_bf16(w[(size_t) o * n_in + j]);
                           partials[lane] = acc;
                           sycl::group_barrier(item.get_group());
                           if (lane == 0) {
                               float s = 0.0f;
                               for (int j = 0; j < lane_count; ++j) s += partials[j];
                               y[o] = s;
                           }
                       });
    });
}
}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        gemv_rows(x, w, y, n_in, n_out, WARP, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv launch: %s\n", e.what());
        std::exit(1);
    }
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    // Same work-group strategy, but `threads_per_row` lanes stride each row, so the reduction is spread across
    // more lanes than the 32 a warp gives.  A non-power-of-two `threads_per_row` cannot be used for a clean lane
    // split, mirroring the CUDA source's refusal.
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two\n", threads_per_row);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        gemv_rows(x, w, y, n_in, n_out, threads_per_row, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "bf16_gemv_split launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
