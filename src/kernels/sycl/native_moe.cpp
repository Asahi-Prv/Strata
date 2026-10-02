// src/kernels/sycl/native_moe.cpp - SYCLomatic port of src/kernels/cuda/native_moe.cu's `native_moe_combine`
// (P2.S2, P3: native MoE, MIT ggml moe-weighted-reduction.cu at 3cf03257f219afbe7334045ff7c6a06ac68c627d).
//
// ONE work-item per output column over `n_embd`, doing the weighted reduction of k expert rows: the first product
// rounds to F32, each following product is FMA-accumulated in expert order, and an optional `shared` row is added
// once at the end (unrouter-weighted).  `docs/...` contract.  No warp ops, no shared memory - just a flat
// range<1>(n_embd), the same shape as the CUDA `<<<(n_embd+255)/256,256>>>` with a column per work-item.
//
// The FMA-accumulate is the CUDA contract (the header says so), so the following products use `sycl::fma` - a single
// round-to-nearest-even multiply-add - rather than a bare `sum += a*b`, which lets the compiler contract differently
// and would move the result just off a .5 ULP; the first product stays a plain multiply that rounds to F32 before the
// loop, matching `float sum = parts[col] * weights[0];`.  The shared add is the source's plain add (a lone f32 add is
// already round-to-nearest-even).
#include "strata/kernels/native_moe.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

namespace {

std::atomic<bool> enabled{false};

bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}

}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                        int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes)
            || (shared && !valid_span(shared, row_bytes))
            || overlap(output, row_bytes, parts, part_bytes) || overlap(output, row_bytes, weights, weight_bytes)
            || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");

    const int ki = (int) k;
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> col) {
                float sum = parts[col] * weights[0];   // first product rounds to F32
                for (int expert = 1; expert < ki; ++expert)
                    sum = sycl::fma(parts[(size_t) expert * n_embd + col], weights[expert], sum);  // FMA, expert order
                if (shared) sum = sum + shared[col];  // added once, unrouter-weighted
                output[col] = sum;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_moe_combine launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
