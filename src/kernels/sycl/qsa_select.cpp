// src/kernels/sycl/qsa_select.cpp - SYCL port of src/kernels/cuda/qsa_select.cu (the QSA indexer's block scores
// and top-k selection).
//
// BLOCK SCORES.  The CUDA `block_scores_kernel` gives one warp per (query, block); each lane holds four of the
// 128 key dimensions and the per-head dot is closed with a warp XOR butterfly.  icpx 2026.1 has no shuffle, so
// this is one work-item per (query, block) summing the 128-dimension dot serially per indexer head and then
// applying the per-head ReLU and the ordered head sum.  Only the summation order changes.
//
// TOP-K.  The CUDA `block_topk_kernel` is one block per query: a 4-pass radix select over the query's blocks
// (each weighted by its cell count) then the cells emitted ascending with ties to the lowest index.  Reproduced
// here by ONE work-item per query running the identical radix selection serially, then the identical ascending
// emission.  The parallel kernel's per-thread prefix sums are only a parallel spelling of the same prefix over
// blocks in ascending order, so the serial emission produces exactly the same `ids` (the order_key threshold,
// the eq_budget and the tie rule are all preserved).
#include "strata/kernels/qsa_select.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

namespace {

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;

// The order-preserving uint32 key: monotone in the fp32 value, with NaN mapped to the smallest key.
inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

}  // namespace

void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
    if (nq <= 0) return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    const size_t total = (size_t) nq * (size_t) max_blocks;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item per (query, block): the 128-dim per-head dot summed serially, ReLU, ordered head add.
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
                const long long w = (long long) wid;
                const long long qi = w / max_blocks;
                const long long b = w % max_blocks;
                const int32_t* st = steps + qi * kStepCount;
                const long long n_kv = st[kStepNKv], n_bid = st[kStepNBid];
                if (b > n_bid) return;
                const float* key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
                const float* q = q_idx + qi * IDX_HEADS * IDX_DIM;
                float score = 0.0f;
                for (int hh = 0; hh < IDX_HEADS; ++hh) {
                    const float* qh = q + hh * IDX_DIM;
                    float d = 0.0f;
                    for (int d0 = 0; d0 < IDX_DIM; ++d0) d += key[d0] * qh[d0];
                    score += d > 0.0f ? d : 0.0f;
                }
                if (b == n_bid && n_kv % R != 0) score += 1e9f;
                scores[qi * max_blocks + b] = score;
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_block_scores launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream) {
    if (nq <= 0) return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nq), [=](sycl::id<1> qid) {
                const long long qi = (long long) qid;
                const int32_t* st = steps + qi * kStepCount;
                const long long n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
                int32_t* out = ids + qi * cap;
                if (n_kv <= width) {   // everything is selected: the identity, ascending
                    for (long long j = 0; j < n_kv; ++j) out[j] = (int32_t) j;
                    return;
                }
                const float* sc = scores + qi * max_blocks;
                const long long nb = n_bid + 1;   // blocks 0..n_bid, the last possibly empty
                // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
                uint32_t prefix = 0;
                int above = 0;
                for (int shift = 24; shift >= 0; shift -= 8) {
                    int hist[256];
                    for (int i = 0; i < 256; ++i) hist[i] = 0;
                    const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
                    for (long long b = 0; b < nb; ++b) {
                        const int wt = b < n_bid ? R : (int) (n_kv - n_bid * R);
                        if (wt == 0) continue;
                        const uint32_t k = order_key(sc[b]);
                        if ((k & hi_mask) == (prefix & hi_mask)) hist[(k >> shift) & 255] += wt;
                    }
                    int cum = above, dg = 255;
                    for (; dg > 0; --dg) {
                        if (cum + hist[dg] >= width) break;
                        cum += hist[dg];
                    }
                    prefix |= (uint32_t) dg << shift;
                    above = cum;
                }
                const uint32_t thr = prefix;
                const long long eq_budget = width - above;   // cells equal to thr that fit, lowest index first
                // ---- the cells emitted ascending, exactly as the CUDA block's per-thread prefix sums arrange
                long long wpos = 0, eq_left = eq_budget;
                for (long long b = 0; b < nb; ++b) {
                    const int wt = b < n_bid ? R : (int) (n_kv - n_bid * R);
                    if (wt == 0) continue;
                    const uint32_t k = order_key(sc[b]);
                    if (k > thr) {
                        for (int c = 0; c < wt; ++c) out[wpos++] = (int32_t) (b * R + c);
                    } else if (k == thr) {
                        for (int c = 0; c < wt && eq_left > 0; ++c, --eq_left) out[wpos++] = (int32_t) (b * R + c);
                    }
                }
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_block_topk launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
