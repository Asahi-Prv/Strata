// src/kernels/sycl/qsa.cpp - SYCL port of src/kernels/cuda/qsa.cu: the QSA cache, indexer and attention
// (see include/strata/kernels/qsa.hpp).
//
// The CUDA file documents four decisions about the KERNELS; they carry over to this port:
//
// 1. THE THRESHOLD SEARCH IN `topk_512` RATHER THAN A SORT.  The CUDA version computes the width-th largest
//    score as a 32-bit "order key" by binary lifting inside ONE 256-thread block and then emits, in one
//    ascending walk, every cell strictly above it and the first `width - c_gt` cells equal to it.  icpx 2026.1
//    exposes no usable block-scan surface here, so the whole single-block algorithm runs in ONE work-item: the
//    binary-lifting count is an integer comparison (order-independent), so it lands on EXACTLY the same
//    threshold, and the ascending emit walk is unchanged.  The selection set is provably the same as sorting
//    by (score desc, index asc) and taking the first width; no scratch, no sort, deterministic.
//
// 2. ONE WORK-GROUP PER HEAD IN `qsa_attend`.  The CUDA launch is one block of `head_dim` threads per query
//    head with the scores in shared memory.  Here each head is a SYCL work-group of `head_dim` work-items and
//    the score/softmax scratch is a `sycl::local_accessor<float,1>` (capacity `max_ids` scores plus a
//    `head_dim`-sized reduction area).  The max and the softmax sum are reduced through local memory with a
//    `sycl::group_barrier` (no `__shfl_*`), and the three passes over the scores stay in local memory.
//
// 3. NO `__float2half`.  `f16_from_f32` from f16_bits.hpp, for the reason round 193 recorded.
//
// 4. THE POOLING ARITHMETIC IS IN DOUBLE AND ORDER-FIXED.  `pooled` and the spare key are compared BIT-EXACT
//    against the reference, so the `r` raw keys are added in CELL order, the sum of squares runs
//    d = 0..idx_dim-1 sequentially, and the CUDA `__dadd_rn`/`__dmul_rn`/`__ddiv_rn`/`__dsqrt_rn` intrinsics
//    (which block FMA contraction) are reproduced with separate multiply/add statements in double.  icpx has
//    no explicit round-to-nearest double intrinsics, so this depends on contraction not fusing the temporaries;
//    see the NOTE at the bottom.
//
// THE STEP STATE is a device buffer uploaded by a plain `queue::memcpy`; every capturable kernel reads its
// per-token counts from it rather than from a kernel argument.  The host-scalar wrappers fill it and forward.
#include "strata/kernels/qsa.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

void fail(const char* what) {
    std::fprintf(stderr, "qsa: %s\n", what);
    std::exit(1);
}

/// Geometry validation.  A silently wrong `head_dim % 4` would corrupt the gather's uint2 copy and a silently
/// wrong `n_head % n_head_kv` would produce a plausible attention with the wrong key, so both are refused.
void validate(const QsaShapes& s, const char* who) {
    if (s.n_head <= 0 || s.n_head_kv <= 0 || s.head_dim <= 0 || s.idx_dim <= 0 || s.idx_n_head <= 0 ||
        s.idx_block < 2 || s.page_size < 1) {
        std::fprintf(stderr, "qsa: %s: geometry is not set up\n", who);
        std::exit(1);
    }
    if (s.n_head % s.n_head_kv != 0) {
        std::fprintf(stderr, "qsa: %s: n_head %lld is not a multiple of n_head_kv %lld\n", who,
                     (long long) s.n_head, (long long) s.n_head_kv);
        std::exit(1);
    }
    if (s.head_dim % 4 != 0) {
        std::fprintf(stderr, "qsa: %s: head_dim %lld must be a multiple of 4 (the gather copies uint2)\n", who,
                     (long long) s.head_dim);
        std::exit(1);
    }
    if (s.n_rot <= 0 || s.n_rot % 2 != 0 || s.n_rot > s.head_dim || s.n_rot > s.idx_dim) {
        std::fprintf(stderr, "qsa: %s: n_rot %lld must be even and <= head_dim %lld and idx_dim %lld\n", who,
                     (long long) s.n_rot, (long long) s.head_dim, (long long) s.idx_dim);
        std::exit(1);
    }
    if (s.idx_n_head > 32) {
        std::fprintf(stderr, "qsa: %s: idx_n_head %lld > 32 (one warp per indexer head)\n", who,
                     (long long) s.idx_n_head);
        std::exit(1);
    }
}

// ================= 1. kv_append =================

void kv_append_kernel(uint16_t* k_pool, uint16_t* v_pool, const int32_t* table, const int32_t* step,
                      const float* kcur, const float* vcur, int kv_heads, int head_dim, int page_size,
                      uint16_t* host_k, uint16_t* host_v, sycl::queue& qref) {
    const int n = kv_heads * head_dim;
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> idx) {
            const long long pos = (long long) step[kStepPos];
            const long long i = (long long) idx;
            const int hh = (int) (i / head_dim), d = (int) (i - (long long) hh * head_dim);
            // `[page][kv_head][page_size][head_dim]`: one head's consecutive cells are contiguous inside a
            // page.  KV streaming: the host copy (identity layout) always, the VRAM page only if resident.
            const long long page = (long long) table[pos / page_size];
            if (page >= 0) {
                const long long row = (page * kv_heads + hh) * page_size + (pos % page_size);
                k_pool[row * head_dim + d] = f16_from_f32(kcur[i]);
                v_pool[row * head_dim + d] = f16_from_f32(vcur[i]);
            }
            if (host_k != nullptr) {
                const long long row = ((pos / page_size) * kv_heads + hh) * page_size + (pos % page_size);
                host_k[row * head_dim + d] = f16_from_f32(kcur[i]);
                host_v[row * head_dim + d] = f16_from_f32(vcur[i]);
            }
        });
    });
}

// ================= 2. indexer_key_append =================

/// Appends the raw key, and on a block completion pools AND ROTATES the pooled row.  The CUDA kernel is one
/// block of `idx_dim` threads with the `mean` shared in double and a completion rotation after a barrier; here
/// the whole block is ONE work-item, with the per-dim means in a `sycl::local_accessor<double,1>` so the sum
/// of squares reads back the same double values the CUDA shared buffer held.
void indexer_key_append_kernel(const float* raw, const int32_t* pos_dev, int pos_base, const float* w_k_norm,
                               float eps, float* tail, float* dead, float* pooled, int32_t* block_pos,
                               int idx_dim, int r, int n_rot, const float* cos_tab, const float* sin_tab,
                               const int32_t* mtab, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> s_mean(sycl::range<1>((size_t) idx_dim), h);
        h.parallel_for(sycl::nd_range<1>((size_t) 1, (size_t) 1), [=](sycl::nd_item<1>) {
            const int pos = (int) *pos_dev;
            const int slot = pos % r;

            // The RAW tail: a cell that COMPLETES a block is not stored, because the pool below consumes it
            // directly.  That is what makes `idx_block - 1` rows enough.
            if (slot < r - 1) {
                for (int d = 0; d < idx_dim; ++d) tail[(size_t) slot * idx_dim + d] = raw[d];
            }

            // The spare slot's key is `rms_norm(raw[0])`, CONSTANT for the sequence.
            if (pos == 0) {
                double ss = 0.0;
                for (int i = 0; i < idx_dim; ++i) {
                    const double v = (double) raw[i];
                    const double vv = v * v;
                    ss = ss + vv;
                }
                const double inv = 1.0 / sycl::sqrt(ss / (double) idx_dim + (double) eps);
                for (int d = 0; d < idx_dim; ++d) {
                    const double p = (double) raw[d];
                    dead[d] = (float) (p * inv * (double) w_k_norm[d]);
                    // rope at position 0 is the identity here (cos = 1, sin = 0 exactly).
                    pooled[d] = dead[d];
                }
            }
            if (slot != r - 1) return;

            // The mean of this block's r raw keys, in CELL order; the sum of squares below is fixed to d
            // ascending on both sides, so the means stay in double until they are scaled.
            const int b = pos / r;
            for (int d = 0; d < idx_dim; ++d) {
                double m = 0.0;
                for (int j = 0; j < r - 1; ++j) m = m + (double) tail[(size_t) j * idx_dim + d];
                m = m + (double) raw[d];
                m = m / (double) r;
                s_mean[d] = m;
            }
            double ss = 0.0;
            for (int i = 0; i < idx_dim; ++i) {
                const double v = s_mean[i];
                const double vv = v * v;
                ss = ss + vv;
            }
            const double inv = 1.0 / sycl::sqrt(ss / (double) idx_dim + (double) eps);

            for (int d = 0; d < idx_dim; ++d) {
                pooled[(size_t) b * idx_dim + d] = (float) (s_mean[d] * inv * (double) w_k_norm[d]);
                // The spare slot MOVES to b+1 and is rewritten with the same constant value.
                pooled[(size_t) (b + 1) * idx_dim + d] = dead[d];
                // The block's first cell's POSITION, `pos_base + b*r`, and NOT the cell index `b*r`.
                if (d == 0) *block_pos = (int32_t) (pos_base + b * r);
            }

            // ---- the rotation of the row that just completed, IN PLACE.  THE SAME `rope_neox_pair` THE
            // STANDALONE ROPE KERNEL USES; a second transcription of the pairing is how it gets written wrong.
            const int half = n_rot / 2;
            for (int d = 0; d < half; ++d) {
                float* row = pooled + (size_t) b * idx_dim;
                const size_t toff = (size_t) mrope_pos(mtab, pos_base + b * r, d) * half;
                rope_neox_pair(row[d], row[half + d], cos_tab[toff + d], sin_tab[toff + d], row[d],
                               row[half + d]);
            }
        });
    });
}

// ================= 3. qsa_index =================

/// One pooled row per work-item.  The reference's Relu is PER HEAD, so the dots cannot be summed first; the
/// dots run in double, d ascending, and the heads are summed in head order - the same order thread 0 used.
void qsa_index_kernel(const float* pooled, const float* q_idx, const float* bias, int idx_n_head, int idx_dim,
                      int r, const int32_t* step, float* cell_scores, int max_blocks, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) max_blocks), [=](sycl::id<1> row) {
            const long long b = (long long) row;
            const long long n_bid = (long long) step[kStepNBid];
            const long long n_kv = (long long) step[kStepNKv];
            // A BLOCK PAST THE COMPLETED-BLOCK COUNT MUST DO NOTHING: this is what lets the LAYER launch a
            // CONSTANT grid of the maximum block count instead of this token's `n_bid + 1`.
            if (b > n_bid) return;

            double score = 0.0;
            for (int hh = 0; hh < idx_n_head; ++hh) {
                double acc = 0.0;
                for (int d = 0; d < idx_dim; ++d) {
                    const double pv = (double) pooled[(size_t) b * idx_dim + d];
                    const double qv = (double) q_idx[(size_t) hh * idx_dim + d];
                    const double prod = pv * qv;   // __dmul_rn: the product rounds before the add
                    acc = acc + prod;              // __dadd_rn
                }
                score += (acc > 0.0) ? acc : 0.0;
            }
            if (bias != nullptr) score += (double) bias[b];
            // cell_block: cells of block b for b < n_bid, and the WHOLE incomplete tail for b == n_bid.
            long long lo = b * r, hi = lo + r;
            if (b == n_bid) hi = n_kv;
            if (hi > n_kv) hi = n_kv;
            float sc = (float) score;
            if (b == n_bid && n_kv % r != 0) sc += 1e9f;
            for (long long j = lo; j < hi; ++j) cell_scores[j] = sc;
        });
    });
}

// ================= 4. topk_512 =================

/// Total order over f32 as an unsigned key, so one integer comparison orders scores with numpy's float
/// semantics.  `+ 0.0f` maps -0.0 to +0.0; a NaN is mapped to 0, below every real key, i.e. never selected.
inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/// The whole single-block selection in ONE work-item.  Binary lifting finds the largest key v with
/// count(key >= v) >= width; the count is an integer over the whole array, so the threshold is identical to
/// the CUDA block reduction's, and the ascending emit walk realises the tie rule (`first by index`) exactly.
void topk_kernel(const float* scores, const int32_t* step, int32_t* out_ids, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            const long long n_kv = (long long) step[kStepNKv];
            const long long width = (long long) step[kStepWidth];

            uint32_t v = 0u;
            for (int bit = 31; bit >= 0; --bit) {
                const uint32_t cand = v | (1u << bit);
                long long c = 0;
                for (long long j = 0; j < n_kv; ++j)
                    if (order_key(scores[j]) >= cand) ++c;
                if (c >= width) v = cand;
            }
            const uint32_t thr = v;

            long long gt = 0, eq = 0;
            for (long long j = 0; j < n_kv; ++j) {
                const uint32_t k = order_key(scores[j]);
                if (k > thr) ++gt;
                else if (k == thr) ++eq;
            }
            // eq_budget is how many equal-to-threshold cells the selection has room for.  Compare the RANK
            // (the number already emitted), not `c_gt + prefix_eq` - the latter selects nothing at the
            // threshold when scores are distinct.
            const long long eq_budget = width - gt;

            long long w = 0, e = 0;
            for (long long j = 0; j < n_kv; ++j) {
                const uint32_t k = order_key(scores[j]);
                if (k > thr) out_ids[w++] = (int) j;
                else if (k == thr && e < eq_budget) { ++e; out_ids[w++] = (int) j; }
            }
        });
    });
}

// ================= 5. kv_gather =================

void kv_gather_kernel(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* table,
                      const int32_t* ids, const int32_t* step, int kv_heads, int head_dim, int page_size,
                      uint16_t* k_scratch, uint16_t* v_scratch, int max_ids, sycl::queue& qref) {
    const int per = head_dim / 4;   // 4 halfs per uint2
    const long long total = (long long) max_ids * kv_heads * per;
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> idx) {
            const long long n_ids = (long long) step[kStepWidth];
            const long long i = (long long) idx;
            if (i >= n_ids * kv_heads * per) return;
            const long long id = i / ((long long) kv_heads * per);
            const int rem = (int) (i % ((long long) kv_heads * per));
            const int hh = rem / per, qq = rem - hh * per;
            const int cell = ids[id];
            const long long page = (long long) table[cell / page_size];
            const long long src = ((page * kv_heads + hh) * page_size + (cell % page_size)) * per + qq;
            const long long dst = (id * kv_heads + hh) * per + qq;
            reinterpret_cast<sycl::uint2*>(k_scratch)[dst] = reinterpret_cast<const sycl::uint2*>(k_pool)[src];
            reinterpret_cast<sycl::uint2*>(v_scratch)[dst] = reinterpret_cast<const sycl::uint2*>(v_pool)[src];
        });
    });
}

// ================= 6. qsa_attend =================

/// One work-group per query head.  `scores` holds the scores, then is overwritten with the softmax weights.
/// The CUDA shared layout `red[0..31] | scores` becomes `smem[0..local-1] | scores` in a local_accessor; the
/// two warp/block reductions become a serial reduction over the `local` work-items after a group barrier.
void qsa_attend_kernel(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                       const int32_t* step, int n_head, int n_head_kv, int head_dim, float* attn,
                       float* weights, int max_ids, sycl::queue& qref) {
    const int local = head_dim;
    if (local <= 0) return;
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> smem(sycl::range<1>((size_t) (max_ids + local)), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_head * local, (size_t) local),
                       [=](sycl::nd_item<1> item) {
                           const int d = (int) item.get_local_id(0);
                           const int hh = (int) item.get_group(0);
                           const long long n_ids = (long long) step[kStepWidth];
                           // AN EMPTY SELECTION IS HANDLED HERE, not by a host early return.
                           if (n_ids == 0) {
                               for (int i = d; i < head_dim; i += local) attn[(size_t) hh * head_dim + i] = 0.0f;
                               return;
                           }
                           const int kv = hh / (n_head / n_head_kv);   // ops.cpp L8729: iq2 / rv2
                           const float scale = 1.0f / sycl::sqrt((float) head_dim);

                           for (long long j = d; j < n_ids; j += local) {
                               const uint16_t* krow = k_scratch + (j * n_head_kv + kv) * head_dim;
                               float acc = 0.0f;
                               for (int i = 0; i < head_dim; ++i)
                                   acc += f32_from_f16(krow[i]) * q[(size_t) hh * head_dim + i];
                               smem[local + j] = acc * scale;
                           }
                           sycl::group_barrier(item.get_group());

                           float mx = -FLT_MAX;   // not -inf, whose double -> float conversion warns
                           for (long long j = d; j < n_ids; j += local) mx = sycl::fmax(mx, smem[local + j]);
                           smem[d] = mx;
                           sycl::group_barrier(item.get_group());
                           if (d == 0) {
                               float m = smem[0];
                               for (int i = 1; i < local; ++i) m = sycl::fmax(m, smem[i]);
                               smem[0] = m;
                           }
                           sycl::group_barrier(item.get_group());
                           mx = smem[0];

                           float sum = 0.0f;
                           for (long long j = d; j < n_ids; j += local) {
                               const float e = sycl::exp(smem[local + j] - mx);
                               smem[local + j] = e;
                               sum += e;
                           }
                           smem[d] = sum;
                           sycl::group_barrier(item.get_group());
                           if (d == 0) {
                               float s = 0.0f;
                               for (int i = 0; i < local; ++i) s += smem[i];
                               smem[0] = 1.0f / s;
                           }
                           sycl::group_barrier(item.get_group());
                           const float inv = smem[0];

                           float acc = 0.0f;
                           for (long long j = 0; j < n_ids; ++j)
                               acc += (smem[local + j] * inv) *
                                      f32_from_f16(v_scratch[(j * n_head_kv + kv) * head_dim + d]);
                           attn[(size_t) hh * head_dim + d] = acc;
                           if (weights != nullptr) {
                               for (long long j = d; j < n_ids; j += local)
                                   weights[(size_t) hh * n_ids + j] = smem[local + j] * inv;
                           }
                       });
    });
}

// ================= 7. qsa_gate_apply =================

void qsa_gate_apply_kernel(const float* attn, const float* q_full, int n_head, int head_dim, uint16_t* out,
                           sycl::queue& qref) {
    const long long n = (long long) n_head * head_dim;
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> idx) {
            const long long i = (long long) idx;
            const int hh = (int) (i / head_dim), d = (int) (i % head_dim);
            const double g = (double) q_full[((size_t) hh * 2 * head_dim) + head_dim + d];   // the SECOND half
            const double sig = 1.0 / (1.0 + sycl::exp(-g));
            out[i] = f16_from_f32((float) ((double) attn[i] * sig));
        });
    });
}

void qsa_gate_apply_f32_kernel(const float* attn, const float* q_full, int n_head, int head_dim, float* out,
                               sycl::queue& qref) {
    const long long n = (long long) n_head * head_dim;
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> idx) {
            const long long i = (long long) idx;
            const int hh = (int) (i / head_dim), d = (int) (i % head_dim);
            const double g = (double) q_full[((size_t) hh * 2 * head_dim) + head_dim + d];   // the SECOND half
            const double sig = 1.0 / (1.0 + sycl::exp(-g));
            out[i] = (float) ((double) attn[i] * sig);
        });
    });
}

int32_t* step_scratch() {
    static int32_t* d_step = nullptr;
    if (d_step == nullptr) {
        sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
        d_step = sycl::malloc_device<int32_t>((size_t) kStepCount, *q);
        if (d_step == nullptr) {
            std::fprintf(stderr, "qsa: step upload: malloc_device failed\n");
            std::exit(1);
        }
    }
    return d_step;
}

void step_upload_raw(const int32_t* h_step) {
    int32_t* d = step_scratch();
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
    q->memcpy(d, h_step, (size_t) qsa_step_bytes());
    q->wait();
}

/// From a POSITION: fills all four entries; the only form that cannot produce an inconsistent n_kv/n_bid/width.
const int32_t* step_upload(int64_t pos, int64_t n_kv_hint, const QsaShapes& s) {
    int32_t h[kStepCount];
    qsa_step_fill(h, pos, s);
    if (n_kv_hint >= 0 && n_kv_hint != (int64_t) h[kStepNKv]) {
        std::fprintf(stderr, "qsa: step upload: n_kv %lld disagrees with pos+1 = %d\n", (long long) n_kv_hint,
                     h[kStepNKv]);
        std::exit(1);
    }
    step_upload_raw(h);
    return step_scratch();
}

/// From a WIDTH: only that entry is meaningful for `kv_gather`/`qsa_attend`; the rest are set consistently.
const int32_t* step_upload_width(int64_t width, const QsaShapes& s) {
    int32_t h[kStepCount];
    for (int i = 0; i < kStepCount; ++i) h[i] = 0;
    h[kStepWidth] = (int32_t) width;
    (void) s;
    step_upload_raw(h);
    return step_scratch();
}

}  // namespace

// ================= THE CAPTURABLE ENTRY POINTS =================

void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                    const KvHostPools* host) {
    validate(s, "kv_append");
    if (step == nullptr) fail("kv_append: step is null");
    const long long n = s.n_head_kv * s.head_dim;
    (void) n;
    uint16_t* host_k = (host && host->k_pool) ? host->k_pool : nullptr;
    uint16_t* host_v = (host && host->v_pool) ? host->v_pool : nullptr;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        kv_append_kernel(k_pool, v_pool, page_table, step, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
                         (int) s.page_size, host_k, host_v, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_append launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_index_step(const float* pooled, const float* q_idx, const float* bias, const QsaShapes& s,
                    const int32_t* step, int64_t max_blocks, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (step == nullptr) fail("qsa_index: step is null");
    if (max_blocks <= 0) fail("qsa_index: max_blocks must be positive");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qsa_index_kernel(pooled, q_idx, bias, (int) s.idx_n_head, (int) s.idx_dim, (int) s.idx_block, step,
                         cell_scores, (int) max_blocks, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_index launch: %s\n", e.what());
        std::exit(1);
    }
}

void topk_512_step(const float* cell_scores, const QsaShapes& s, int64_t cap, const int32_t* step,
                   int32_t* ids, void* stream) {
    validate(s, "topk_512");
    if (step == nullptr) fail("topk_512: step is null");
    // `cap` is checked against the GEOMETRY'S largest possible width rather than this token's, because this
    // token's is a device value.
    const int64_t width_max = qsa_selection_width(kTopkMaxCells, s);
    if (cap < width_max) {
        std::fprintf(stderr, "qsa: topk_512: cap %lld < the largest possible selection width %lld\n",
                     (long long) cap, (long long) width_max);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        topk_kernel(cell_scores, step, ids, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "topk_512 launch: %s\n", e.what());
        std::exit(1);
    }
}

void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const QsaShapes& s,
                    uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (step == nullptr) fail("kv_gather: step is null");
    if (max_ids <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        kv_gather_kernel(k_pool, v_pool, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim,
                         (int) s.page_size, k_scratch, v_scratch, (int) max_ids, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_gather launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_attend_step(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch,
                     const int32_t* step, int64_t max_ids, const QsaShapes& s, float* attn, float* weights,
                     void* stream) {
    validate(s, "qsa_attend");
    if (step == nullptr) fail("qsa_attend: step is null");
    if (max_ids <= 0) fail("qsa_attend: max_ids must be positive");
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // The local scratch is sized from the CONSTANT capacity `max_ids` (scores) plus `head_dim` (reduction).
        // The CUDA dynamic-shared opt-in has no SYCL analogue; the runtime sizes the local_accessor instead.
        qsa_attend_kernel(q, k_scratch, v_scratch, step, (int) s.n_head, (int) s.n_head_kv, (int) s.head_dim,
                          attn, weights, (int) max_ids, *qp);
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_attend launch: %s\n", e.what());
        std::exit(1);
    }
}

// ================= host entry points =================

void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) fail("qsa_step_fill: pos < 0");
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

void kv_append(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, int64_t pos,
               const float* kcur, const float* vcur, const QsaShapes& s, void* stream) {
    if (pos < 0) { validate(s, "kv_append"); fail("kv_append: pos < 0"); }
    kv_append_step(k_pool, v_pool, page_table, step_upload(pos, -1, s), kcur, vcur, s, stream);
}

void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    validate(s, "indexer_key_append");
    if (pos_dev == nullptr) fail("indexer_key_append: pos_dev is null");
    if (b.tail == nullptr || b.dead == nullptr || b.pooled == nullptr || b.block_pos == nullptr)
        fail("indexer_key_append: the indexer buffers are not all set (tail/dead/pooled/block_pos)");
    const int32_t* mtab = mrope_table();
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        indexer_key_append_kernel(raw, pos_dev, (int) pos_base, w_k_norm, eps, b.tail, b.dead, b.pooled,
                                  b.block_pos, (int) s.idx_dim, (int) s.idx_block, (int) s.n_rot, cos_tab,
                                  sin_tab, mtab, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "indexer_key_append launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_index(const float* pooled, int64_t n_bid, const float* q_idx, const float* bias, const QsaShapes& s,
               int64_t n_kv, float* cell_scores, void* stream) {
    validate(s, "qsa_index");
    if (n_bid < 0 || n_kv <= 0) fail("qsa_index: n_bid < 0 or n_kv <= 0");
    if ((n_kv / s.idx_block) != n_bid) {
        std::fprintf(stderr, "qsa: qsa_index: n_bid %lld is not n_kv %lld / r %lld\n", (long long) n_bid,
                     (long long) n_kv, (long long) s.idx_block);
        std::exit(1);
    }
    qsa_index_step(pooled, q_idx, bias, s, step_upload(n_kv - 1, n_kv, s), n_bid + 1, cell_scores, stream);
}

void topk_512(const float* cell_scores, int64_t n_kv, const QsaShapes& s, int64_t cap, int32_t* ids,
              void* stream) {
    validate(s, "topk_512");
    if (n_kv <= 0) return;
    if (n_kv > kTopkMaxCells) {
        std::fprintf(stderr, "qsa: topk_512: n_kv %lld > kTopkMaxCells %lld (one block; phase 3 replaces this)\n",
                     (long long) n_kv, (long long) kTopkMaxCells);
        std::exit(1);
    }
    topk_512_step(cell_scores, s, cap, step_upload(n_kv - 1, n_kv, s), ids, stream);
}

void kv_gather(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
               int64_t n_ids, const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather");
    if (n_ids <= 0) return;
    kv_gather_step(k_pool, v_pool, page_table, ids, step_upload_width(n_ids, s), n_ids, s, k_scratch,
                   v_scratch, stream);
}

void qsa_attend(const float* q, const uint16_t* k_scratch, const uint16_t* v_scratch, int64_t n_ids,
                const QsaShapes& s, float* attn, float* weights, void* stream) {
    validate(s, "qsa_attend");
    if (n_ids < 0) fail("qsa_attend: n_ids < 0");
    if (n_ids == 0) {
        // The empty selection goes through the kernel's own zero path; `max_ids` must still be positive.
        qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(0, s),
                        qsa_selection_width(kTopkMaxCells, s), s, attn, weights, stream);
        return;
    }
    qsa_attend_step(q, k_scratch, v_scratch, step_upload_width(n_ids, s), n_ids, s, attn, weights, stream);
}

void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    validate(s, "qsa_gate_apply_f32");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qsa_gate_apply_f32_kernel(attn, q_full, (int) s.n_head, (int) s.head_dim, out, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_gate_apply_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_gate_apply(const float* attn, const float* q_full, const QsaShapes& s, uint16_t* out, void* stream) {
    validate(s, "qsa_gate_apply");
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qsa_gate_apply_kernel(attn, q_full, (int) s.n_head, (int) s.head_dim, out, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_gate_apply launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
