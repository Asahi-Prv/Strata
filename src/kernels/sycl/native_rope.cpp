// src/kernels/sycl/native_rope.cpp - SYCLomatic port of src/kernels/cuda/native_moe... native_rope.cu's IMRoPE
// (P2.S2, text-only, MIT ggml rope.cu rope_multi).
//
// ONE work-item per rotary pair: `pair` in [0, head_dim/2) of a row rotates (a,b) by (cos theta, sin theta) where
// theta = mrope_pos(mtab, positions[row], pair) * powf(theta_scale, pair); pairs beyond n_rot/2 are copied
// verbatim (the non-rotated tail).  The CUDA launch is a 2D grid `dim3((head_dim/2+127)/128, rows)` with one pair
// per thread, so the port is a flat `parallel_for(range<2>(rows, head_dim/2))` - one work-item per (row, pair) -
// which is the same one-thread-per-pair shape with no reduction, no warp ops, no shared memory.
//
// `mrope_pos` is host+device callable (mrope.hpp's #else branch), so the table read is one function on both
// backends.  The transcendental `pow/sin/cos` are the SYCL math functions (`sycl::pow/sin/cos`), the device-callable
// SYCL equivalents of the CUDA `powf/sinf/cosf`.
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

namespace {

std::atomic<bool> enabled{false};

bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

}  // namespace

void native_rope_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_rope_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot, float freq_base,
                       const int* positions, void* stream) {
    if (!x || !out || !positions || !stream || rows < 1 || rows > 65535 ||
        (head_dim != 128 && head_dim != 256) || n_rot != 64 || !std::isfinite(freq_base) || freq_base <= 1.0f ||
        reinterpret_cast<uintptr_t>(x) % 4 || reinterpret_cast<uintptr_t>(out) % 4 ||
        reinterpret_cast<uintptr_t>(positions) % 4) {
        throw std::invalid_argument(
            "native RoPE requires aligned F32 rows, width 128/256, rotation 64, valid base and explicit stream");
    }
    const size_t bytes = size_t(rows) * head_dim * sizeof(float);
    if ((x != out && overlaps(x, bytes, out, bytes)) ||
        overlaps(positions, size_t(rows) * sizeof(int), out, bytes) ||
        overlaps(positions, size_t(rows) * sizeof(int), x, bytes)) {
        throw std::invalid_argument("native RoPE buffers partially overlap");
    }
    // Match pinned host-side float powf before device fast powf/trigonometry.
    const float theta_scale = sycl::pow(freq_base, -2.0f / n_rot);
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    const int pairs = head_dim / 2;
    // mrope_table() is a host-side reader of the (mutable) capture pointer; read it here and hand the value to the
    // kernel as an argument, exactly as the CUDA source passes `mrope_table()` as the kernel's `mtab` parameter.
    const int32_t* mtab = mrope_table();
    try {
        q->submit([&](sycl::handler& h) {
            // Flat over every (row, pair); the pair axis is the whole head_dim/2, mirroring CUDA's pair/thread.
            h.parallel_for(sycl::range<2>((size_t) rows, (size_t) pairs),
                           [=](sycl::id<2> idx) {
                               const int row = idx[0];
                               const int pair = idx[1];
                               const size_t start = (size_t) row * head_dim;
                               if (pair >= n_rot / 2) {
                                   if (x != out) {
                                       out[start + 2 * pair] = x[start + 2 * pair];
                                       out[start + 2 * pair + 1] = x[start + 2 * pair + 1];
                                   }
                                   return;
                               }
                               // mrope_pos inlined (mrope.hpp's #else branch) - the device lambda cannot call the
                               // host declaration without SYCL_EXTERNAL, so the identical table read sits here.
                               const int pos = mtab ? mtab[(size_t) positions[row] * 3 + pair % 3] : positions[row];
                               const float theta = pos * sycl::pow(theta_scale, (float) pair);
                               const float c = sycl::cos(theta), s = sycl::sin(theta);
                               const float a = x[start + pair], b = x[start + pair + n_rot / 2];
                               out[start + pair] = a * c - b * s;
                               out[start + pair + n_rot / 2] = a * s + b * c;
                           });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_rope_apply launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
