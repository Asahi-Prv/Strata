// src/kernels/sycl/native_router.cpp - SYCL port of src/kernels/cuda/native_router.cu.
//
// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// THE WARP REDUCTIONS.  The CUDA kernel uses `__shfl_xor_sync` for the max, the sum, and the ten-round
// top-10 selection.  icpx 2026.1 exposes no subgroup-shuffle surface, so the whole 32-lane warp is executed
// by ONE work-item: the softmax max/sum walk all 512 logits serially, and the selection scans all 512 experts
// for each of the ten ranks.  The selection's combine rule `other > best || (other == best && other_id <
// expert)` is associative and commutative, so the chosen experts and their tie-break (lower expert index)
// are exactly the ones the XOR butterfly produces; only the ORDER of the floating-point additions in the
// final `selected_sum` differs (validated by max_rel).  The per-lane `values[16]` register file becomes one
// private `values[512]`, and the NaN -> -FLT_MAX and 2^-14 normalization floor are preserved verbatim.
#include "strata/kernels/native_router.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};

bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                float values[512];
                for (int i = 0; i < 512; ++i) values[i] = logits[i];
                float maximum = -INFINITY;
                for (int i = 0; i < 512; ++i) maximum = sycl::fmax(maximum, values[i]);
                float sum = 0.0f;
                for (int i = 0; i < 512; ++i) {
                    values[i] = sycl::exp(values[i] - maximum);
                    sum += values[i];
                }
                const float reciprocal = 1.0f / sum;
                for (int i = 0; i < 512; ++i) {
                    values[i] *= reciprocal;
                    if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
                }
                float bests[10];
                float selected_sum = 0.0f;
                for (int rank = 0; rank < 10; ++rank) {
                    float best = values[0];
                    int expert = 0;
                    for (int i = 1; i < 512; ++i) {
                        if (values[i] > best) { best = values[i]; expert = i; }
                    }
                    values[expert] = -INFINITY;
                    ids[rank] = expert;
                    bests[rank] = best;
                    // Deliberately accumulate by WINNING EXPERT rank order; multiple selected experts in one
                    // lane add in selection order in the CUDA warp form, which this reproduces.
                    selected_sum += best;
                }
                selected_sum = sycl::fmax(selected_sum, 6.103515625e-5f);
                const float inverse_selected_sum = 1.0f / selected_sum;
                for (int rank = 0; rank < 10; ++rank) weights[rank] = bests[rank] * inverse_selected_sum;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_router_top10 launch: %s\n", e.what());
        std::exit(1);
    }
}
}  // namespace strata::kernels
