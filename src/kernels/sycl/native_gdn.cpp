// src/kernels/sycl/native_gdn.cpp - SYCL port of src/kernels/cuda/native_gdn.cu.
//
// Arithmetic adapted from gated_delta_net.cu at pinned llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Only state addressing differs:
// each work-group owns one Strata (head,column) and retains its four rows in registers.
//
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
// THE WARP REDUCTION.  The CUDA kernel reduces each (head,column) shard with `__shfl_xor_sync` warp
// butterflies.  icpx 2026.1 has no subgroup-shuffle surface, so each CUDA warp becomes one SYCL work-group
// of 32 work-items and the two reductions (the `kv` accumulation and the `attn` readout) go through a
// `sycl::local_accessor<float,1>` with `sycl::group_barrier` - the same lane partials reduced in lane order
// by lane 0.  The summation order differs from the XOR tree, so this is not bit-identical; every product is
// exact and only the reduction order differs (validated by max_rel).  The `(head,column)` ownership, the
// register-resident four `s_shard`/`k_reg`/`q_reg` rows and the in-place state update are unchanged.
#include "strata/kernels/native_gdn.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int S = 128;

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 ||
        shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    const float scale = 1.0f / std::sqrt(float(S));
    const int h_k = int(shape.h_k), h_v = int(shape.h_v);
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-group of 32 work-items per (head, column) replaces the CUDA warp; the group's lane
        // partials are reduced through local memory (the same lane order the two-pass form used).
        qp->submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>((size_t) h_v * S * 32, 32), [=](sycl::nd_item<1> item) {
                const int lane = (int) item.get_local_id(0);
                const long long g = (long long) item.get_group(0);
                const int head = (int) (g / S);
                const int col = (int) (g % S);
                const int q_head = head % h_k;
                float s_shard[4], k_reg[4], q_reg[4];
                for (int r = 0; r < 4; ++r) {
                    const int i = r * 32 + lane;
                    s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
                    k_reg[r] = k[q_head * S + i];
                    q_reg[r] = q[q_head * S + i];
                }
                const float g_val = sycl::exp(gate[head]);
                float kv_shard = 0.0f;
                for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
                red[lane] = kv_shard;
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float s = 0.0f;
                    for (int j = 0; j < 32; ++j) s += red[j];
                    red[0] = s;
                }
                sycl::group_barrier(item.get_group());
                const float kv_col = red[0];
                const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
                float attn_partial = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                    attn_partial += s_shard[r] * q_reg[r];
                }
                red[lane] = attn_partial;
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float s = 0.0f;
                    for (int j = 0; j < 32; ++j) s += red[j];
                    red[0] = s;
                }
                sycl::group_barrier(item.get_group());
                const float attn_col = red[0];
                if (lane == 0) output[head * S + col] = attn_col * scale;
                for (int r = 0; r < 4; ++r) {
                    const int i = r * 32 + lane;
                    state[(size_t(i) * h_v + head) * S + col] = s_shard[r];
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_step launch: %s\n", e.what());
        std::exit(1);
    }
}
}  // namespace strata::kernels
