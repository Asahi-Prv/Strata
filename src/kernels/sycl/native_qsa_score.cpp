// src/kernels/sycl/native_qsa_score.cpp - SYCL port of src/kernels/cuda/native_qsa_score.cu (ggml-cuda/mmf.cuh, mma.cuh).
//
// SM120 single contiguous text sequence: F32 pooled[128,max_blocks] x query[128,4] matmul, four-cell blocks, top-k
// budget 2048. The CUDA launch is a grid of 32-row tiles over 64 threads (2 warps split the 128 key dims) that folds
// the matmul through tensor-core MMA. icpx 2026.1 exposes no usable subgroup-MMA surface here, so this port folds the
// 128x128x4 matmul with a plain fp32 loop - one work-item per row, the same FP32 multiply-add order the oracle
// materializes - then ReLU per head, ordered F32 head addition, optional F32 block bias, then the finite 1e9 tail bias
// plus the reference's +0. `mrope_pos`/group reductions are not involved.
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, HEADS = 4, R = 4;
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
}  // namespace

void native_qsa_score_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_score(const float* pooled, const float* query, const float* block_bias,
                      const QsaShapes& s, const int32_t* step, int64_t max_blocks, int64_t max_cells,
                      float* cell_scores, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 ||
        max_cells < 1 || max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[] = {{pooled, (size_t)max_blocks * D * 4}, {query, HEADS * D * 4},
                          {step, kStepCount * 4}, {cell_scores, (size_t)max_cells * 4},
                          {block_bias, block_bias ? (size_t)max_blocks * 4 : 0}};
    const int count = block_bias ? 5 : 4;
    for (int i = 0; i < count; ++i) validate(spans[i]);
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t)max_blocks), [=](sycl::id<1> row) {
                const int irow = (int)row;
                const int n = step[kStepNKv], full = step[kStepNBid];
                if (n < 1 || n > max_cells || step[kStepPos] != n - 1 || full != n / R ||
                    step[kStepWidth] != (n < 2051 ? n : 2051))
                    return;
                if (irow > full) return;
                float h[HEADS];
                for (int head = 0; head < HEADS; ++head) {
                    float dot = 0.0f;
                    const float* pr = pooled + (size_t)irow * D;
                    const float* qh = query + (size_t)head * D;
                    for (int d = 0; d < D; ++d) dot += pr[d] * qh[d];
                    h[head] = dot > 0.0f ? dot : 0.0f;
                }
                float sum = h[0] + h[1] + h[2] + h[3];
                if (block_bias) sum += block_bias[irow];
                if (irow == full && n % R) sum += 1e9f;
                sum += 0.0f;
                for (int i = irow * R; i < n && i < (irow + 1) * R; ++i) cell_scores[i] = sum;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_qsa_score launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
