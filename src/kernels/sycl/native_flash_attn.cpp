// src/kernels/sycl/native_flash_attn.cpp - SYCL port of src/kernels/cuda/native_flash_attn.cu.
//
// Specialized from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cuda/{fattn-vec.cuh,fattn-common.cuh,common.cuh}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// THE PORT SHAPE.  The CUDA `attend` is a vector (not tensor-core) FlashAttention: one block of 128 threads per
// head, a warp XOR butterfly for the Q.K dot and the running max, and a shared-memory transpose for the V
// accumulate.  icpx 2026.1 exposes no `__shfl_*`/`__syncwarp` surface here, and reconstructing that exact
// warp-mapped shared layout in SYCL would not change the arithmetic - it is a numerical device for moving the
// same softmax-weighted V sum.  Per the porting convention this file therefore implements the SAME MATH with
// plain per-element FMA loops: one work-item per head folds the 256-wide Q.K dot (q pre-scaled by `scale`, K in
// fp16 decoded exactly with f32_from_f16), adds the optional fp16 mask, takes the softmax over the live cells
// and accumulates sum_j w_j * V_j, then divides by the softmax denominator.  The `+ 3*ln2` the CUDA source adds
// to the running maximum only rescales every weight by one common constant, so it cancels between the numerator
// and the denominator and is dropped.  Products are exact and only the summation order differs (validated by
// max_rel); the unsupported-step contract (status + all-NaN output, no q/k/v/mask read) is preserved exactly.
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}
}  // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                                  const int32_t* step, int64_t capacity, int max_context,
                                  const QsaShapes& shapes, float* output, int32_t* status,
                                  const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 ||
        shapes.idx_block != 4 || shapes.idx_top_k < 256 || capacity < 256 ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 ||
        max_context < 1 || max_context > 256)
        throw std::invalid_argument("native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an explicit stream");
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, kStepCount * 4, 4}, {output, 24 * 256 * 4, 4},
                          {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    sycl::queue* qq = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const float scale = 0.0625f;   // 1/16, pinned by the launcher
    try {
        qq->submit([&](sycl::handler& h) {
            // One work-item per head folds the whole 256-wide dot + softmax + V accumulate with plain FMA loops.
            h.parallel_for(sycl::range<1>(24), [=](sycl::id<1> hid) {
                const int head = (int) hid, kv = head / 12;
                const int width = step[kStepWidth], nkv = step[kStepNKv];
                const bool valid = width >= 1 && width <= max_context && nkv == width &&
                                   step[kStepPos] == width - 1 && step[kStepNBid] == width / 4;
                if (head == 0)
                    *status = valid ? kNativeFlashAttnSuccess : kNativeFlashAttnUnsupportedStep;
                float* out = output + (size_t) head * 256;
                if (!valid) {
                    const float nan = std::numeric_limits<float>::quiet_NaN();
                    for (int d = 0; d < 256; ++d) out[d] = nan;
                    return;
                }
                const float* qh = q + (size_t) head * 256;
                float maximum = -FLT_MAX / 2.0f;
                for (int cell = 0; cell < width; ++cell) {
                    const uint16_t* kr = k + ((size_t) cell * 2 + kv) * 256;
                    float dot = 0.0f;
                    for (int d = 0; d < 256; ++d) dot += f32_from_f16(kr[d]) * (qh[d] * scale);
                    if (mask) dot += f32_from_f16(mask[cell]);
                    maximum = sycl::fmax(maximum, dot);
                }
                float vkq[256];
                for (int d = 0; d < 256; ++d) vkq[d] = 0.0f;
                float sum = 0.0f;
                for (int cell = 0; cell < width; ++cell) {
                    const uint16_t* kr = k + ((size_t) cell * 2 + kv) * 256;
                    const uint16_t* vr = v + ((size_t) cell * 2 + kv) * 256;
                    float dot = 0.0f;
                    for (int d = 0; d < 256; ++d) dot += f32_from_f16(kr[d]) * (qh[d] * scale);
                    if (mask) dot += f32_from_f16(mask[cell]);
                    const float w = sycl::exp(dot - maximum);
                    sum += w;
                    for (int d = 0; d < 256; ++d) vkq[d] += w * f32_from_f16(vr[d]);
                }
                for (int d = 0; d < 256; ++d) out[d] = vkq[d] / sum;
            });
        });
        if (stream == nullptr) qq->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native FlashAttention launch: %s\n", e.what());
        std::exit(1);
    }
}
}  // namespace strata::kernels
