// src/kernels/sycl/native_qsa.cpp - SYCL port of src/kernels/cuda/native_qsa.cu (ggml-cuda/norm.cu, unary.cu).
//
// Two entry points.  `native_qsa_rms_norm_weighted` is a weighted RMSNorm, one block per row, doing a warp-XOR butterfly
// reduce of sum(value^2) plus a shared-memory block reduce - the same structure as native_gr_norm.  icpx 2026.1 has no
// group-reduction surface, so the reduce is two passes over a per-(row, lane) partial buffer: pass one has one
// work-item per lane striding n_cols accumulating value^2, pass two sums the 32 lane partials for the row mean,
// rsqrts it and applies scale*input*gamma.  `native_qsa_gate_apply` is a plain elementwise gate, one work-item per
// element: attn[i] * sigmoid(q_full[head*2*head_dim + head_dim + channel]).
#include "strata/kernels/native_qsa.hpp"

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
#include <string>

namespace strata::kernels {

namespace {
constexpr int WARP = 32;
std::atomic<bool> enabled{false};
constexpr int THREADS = 256;

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes, float* output,
             void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument(
            "native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
}  // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                                  float epsilon, void* stream) {
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    const auto count = elements(n_cols, n_rows);
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
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
        std::fprintf(stderr, "native_qsa_rms_norm_weighted launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim,
                           void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * sizeof(float), output, stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) count), [=](sycl::id<1> i) {
                const size_t head = i / head_dim, channel = i % head_dim;
                const float raw = q_full[head * 2 * head_dim + head_dim + channel];
                const float sigmoid = 1.0f / (1.0f + sycl::exp(-raw));
                output[i] = attn[i] * sigmoid;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_qsa_gate_apply launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
