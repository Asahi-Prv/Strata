// src/kernels/sycl/qsa_decode_attn.cpp - SYCL port of src/kernels/cuda/qsa_decode_attn.cu (split-K decode
// attention over the selected cells).
//
// The CUDA source is two kernels on one stream: `attn_chunk_kernel` (one block per (chunk, kv_head) with 256
// threads, warp-reduced scores and a shared-memory tile) and `attn_merge_kernel` (the log-sum-exp rescale).  icpx
// 2026.1 exposes no shuffle and no group-reduction surface here, so each phase is run by work-items that walk
// their slice serially:
//
//   chunk: one work-item per (query, chunk, kv_head, query-head).  It computes that head's per-cell scores
//          (the 256-dim k.q dot in one serial sum), the chunk max and exp-sum, then accumulates p.V over the
//          chunk's cells into the partial accumulator.  The per-head ReLU/softmax semantics and the
//          `fma(prob, v, acc)` accumulation order are unchanged; only the reduction order differs.
//   merge: one work-item per (query, query-head); it takes the running max over the chunk partials and folds
//          them with the identical `w = exp(m - M)`, `L += l*w`, `acc += acc_c*w` rescale, then `acc/L`.
//
// The two phases are separate submits with an explicit event dependency: icpx 2026.1 queues are OUT-OF-ORDER, so
// submit order alone does not order them and the merge would otherwise be free to read the partials before the
// chunk pass wrote them.  The partials are the caller's scratch, so there is no malloc_device/free to race on.
//
// FP16 pools, INT8 pools (`kv_q8.hpp`) and q4_0 pools (`kv_q4.hpp`) are selected by which pointer is non-null,
// exactly as in the CUDA dispatch.  `step[kStepWidth]` is read from device memory inside every kernel, as
// everywhere else in QSA.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

constexpr int HD = 256;   // head_dim
constexpr int G = 12;     // query heads per KV head (24 / 2)
constexpr int CHUNK = 64; // cells per block

// One dimension of one cell's key (value == false) or value (value == true), decoded from the pool selected by
// KV_MODE: 0 fp16, 1 int8 codes + fp16 scales, 2 q4_0 blocks.
template <int KV_MODE>
inline float decode_kv(const uint16_t* k_pool, const uint16_t* v_pool, const int8_t* k_q, const int8_t* v_q,
                       const uint16_t* k_scale, const uint16_t* v_scale, const uint8_t* k_q4, const uint8_t* v_q4,
                       bool value, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        const uint16_t* base = value ? v_pool : k_pool;
        return f32_from_f16(base[row * HD + d]);
    } else if constexpr (KV_MODE == 1) {
        const uint16_t* base = value ? v_scale : k_scale;
        const int8_t* codes = value ? v_q : k_q;
        const float sc = f32_from_f16(base[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]);
        return (float) codes[row * HD + d] * sc;
    } else {
        constexpr int blocks_per_head = HD / QK4_0;
        constexpr int bytes_per_head = blocks_per_head * (int) sizeof(block_q4_0);
        const uint8_t* pool = value ? v_q4 : k_q4;
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(pool + row * bytes_per_head) + b;
        const float dd = f32_from_f16(blk->d);
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nib = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nib * dd;
    }
}

// The chunk phase: one work-item per (query, chunk, kv_head, query-head).
template <int KV_MODE>
sycl::event chunk_launch(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int8_t* k_q,
                         const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale, const uint8_t* k_q4,
                         const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids, const int32_t* step,
                         int n_kv_heads, int page_size, float scale, float* part_acc, float* part_m, float* part_l,
                         int n_chunks, int cap, long long scratch_stride, int n_q, sycl::queue& qref) {
    const size_t total = (size_t) n_q * (size_t) n_chunks * (size_t) n_kv_heads * (size_t) G;
    return qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
            int w = (int) wid;
            const int hl = w % G;
            w /= G;
            const int kvh = w % n_kv_heads;
            w /= n_kv_heads;
            const int chunk = w % n_chunks;
            const int z = w / n_chunks;

            const int32_t* zstep = step + (size_t) z * kStepCount;
            const int32_t* zids = ids + (size_t) z * cap;
            const float* zq = q + (size_t) z * (size_t) (n_kv_heads * G * HD);
            float* zacc = part_acc + (size_t) z * (size_t) scratch_stride;
            float* zm = part_m + (size_t) z * (size_t) scratch_stride;
            float* zl = part_l + (size_t) z * (size_t) scratch_stride;

            const int n_ids = zstep[kStepWidth];
            const int c0 = chunk * CHUNK;
            int n_here = n_ids - c0;
            if (n_here > CHUNK) n_here = CHUNK;
            const int slot = kvh * n_chunks + chunk;
            if (n_here <= 0) {
                zm[slot * G + hl] = -FLT_MAX;
                zl[slot * G + hl] = 0.0f;
                return;
            }
            const float* qrow = zq + (size_t) (kvh * G + hl) * HD;

            // scores: this head's k.q for every cell of the chunk, scaled, with the running max.
            float sp[CHUNK];
            float m = -FLT_MAX;
            for (int c = 0; c < n_here; ++c) {
                const int cell = zids[c0 + c];
                const long long page = (long long) page_table[cell / page_size];
                const long long row = ((long long) page * n_kv_heads + kvh) * page_size + (cell % page_size);
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d)
                    dot += decode_kv<KV_MODE>(k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, false, row, d) *
                           qrow[d];
                sp[c] = dot * scale;
                if (sp[c] > m) m = sp[c];
            }
            zm[slot * G + hl] = m;
            float l = 0.0f;
            for (int c = 0; c < n_here; ++c) {
                sp[c] = sycl::exp(sp[c] - m);
                l += sp[c];
            }
            zl[slot * G + hl] = l;

            // values: one dimension at a time, p.V accumulated with fma in cell order.
            for (int d = 0; d < HD; ++d) {
                float acc = 0.0f;
                for (int c = 0; c < n_here; ++c) {
                    const int cell = zids[c0 + c];
                    const long long page = (long long) page_table[cell / page_size];
                    const long long row = ((long long) page * n_kv_heads + kvh) * page_size + (cell % page_size);
                    acc = sycl::fma(sp[c],
                                    decode_kv<KV_MODE>(k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, true,
                                                       row, d),
                                    acc);
                }
                zacc[((size_t) slot * G + hl) * HD + d] = acc;
            }
        });
    });
}

// The merge phase: one work-item per (query, query-head), the log-sum-exp rescale over the chunk partials.
sycl::event merge_launch(const float* part_acc, const float* part_m, const float* part_l, int n_chunks, float* attn,
                         long long scratch_stride, int n_head, int n_q, sycl::event dep, sycl::queue& qref) {
    const size_t total = (size_t) n_q * (size_t) n_head;
    return qref.submit([&](sycl::handler& h) {
        h.depends_on(dep);   // the merge reads the chunk pass's partials
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
            const long long w = (long long) wid;
            const int hh = (int) (w % n_head);
            const int z = (int) (w / n_head);
            const float* zacc = part_acc + (size_t) z * (size_t) scratch_stride;
            const float* zm = part_m + (size_t) z * (size_t) scratch_stride;
            const float* zl = part_l + (size_t) z * (size_t) scratch_stride;
            const int kvh = hh / G, hl = hh % G;

            float M = -FLT_MAX;
            for (int c = 0; c < n_chunks; ++c) M = sycl::fmax(M, zm[(kvh * n_chunks + c) * G + hl]);
            float* arow = attn + ((size_t) z * n_head + hh) * HD;
            for (int d = 0; d < HD; ++d) {
                float L = 0.0f, acc = 0.0f;
                for (int c = 0; c < n_chunks; ++c) {
                    const int slot = kvh * n_chunks + c;
                    const float mm = zm[slot * G + hl];
                    if (mm == -FLT_MAX) continue;
                    const float ww = sycl::exp(mm - M);
                    L = sycl::fma(zl[slot * G + hl], ww, L);
                    acc = sycl::fma(zacc[((size_t) slot * G + hl) * HD + d], ww, acc);
                }
                arow[d] = L > 0.0f ? acc / L : 0.0f;
            }
        });
    });
}

}  // namespace

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr ? 1 : 0);
    if (kv_mode == 2 ? (!pools.v_q4)
                     : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                     : (!pools.k_pool || !pools.v_pool))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    const int n_kv_heads = (int) s.n_head_kv, page_size = (int) s.page_size, n_head = (int) s.n_head;
    // Unpack the pool struct so only plain pointer/int captures reach the kernel lambdas.
    const uint16_t* k_pool = pools.k_pool;
    const uint16_t* v_pool = pools.v_pool;
    const int8_t* k_q = pools.k_q;
    const int8_t* v_q = pools.v_q;
    const uint16_t* k_scale = pools.k_scale;
    const uint16_t* v_scale = pools.v_scale;
    const uint8_t* k_q4 = pools.k_q4;
    const uint8_t* v_q4 = pools.v_q4;
    const int32_t* page_table = pools.page_table;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event ev;
        if (kv_mode == 2)
            ev = chunk_launch<2>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, step,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0, 1, *qp);
        else if (kv_mode == 1)
            ev = chunk_launch<1>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, step,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0, 1, *qp);
        else
            ev = chunk_launch<0>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, step,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, 0, 0, 1, *qp);
        // The merge depends on the chunk pass; an out-of-order queue needs the dependency spelled out.
        merge_launch(part_acc, part_m, part_l, n_chunks, attn, 0, n_head, 1, ev, *qp);
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_decode_attn launch: %s\n", e.what());
        std::exit(1);
    }
}

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr ? 1 : 0);
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / std::sqrt((float) HD);
    const int n_kv_heads = (int) s.n_head_kv, page_size = (int) s.page_size, n_head = (int) s.n_head;
    const uint16_t* k_pool = pools.k_pool;
    const uint16_t* v_pool = pools.v_pool;
    const int8_t* k_q = pools.k_q;
    const int8_t* v_q = pools.v_q;
    const uint16_t* k_scale = pools.k_scale;
    const uint16_t* v_scale = pools.v_scale;
    const uint8_t* k_q4 = pools.k_q4;
    const uint8_t* v_q4 = pools.v_q4;
    const int32_t* page_table = pools.page_table;
    const int nq = (int) n_q, cp = (int) cap;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event ev;
        if (kv_mode == 2)
            ev = chunk_launch<2>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, steps,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, cp, stride, nq, *qp);
        else if (kv_mode == 1)
            ev = chunk_launch<1>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, steps,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, cp, stride, nq, *qp);
        else
            ev = chunk_launch<0>(q, k_pool, v_pool, k_q, v_q, k_scale, v_scale, k_q4, v_q4, page_table, ids, steps,
                                 n_kv_heads, page_size, scale, part_acc, part_m, part_l, n_chunks, cp, stride, nq, *qp);
        // The merge depends on the chunk pass; an out-of-order queue needs the dependency spelled out.
        merge_launch(part_acc, part_m, part_l, n_chunks, attn, stride, n_head, nq, ev, *qp);
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "qsa_decode_attn_batch launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
