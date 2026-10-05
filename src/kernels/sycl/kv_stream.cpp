// src/kernels/sycl/kv_stream.cpp - SYCL port of src/kernels/cuda/kv_stream.cu (KV streaming for the QSA layers).
//
// The CUDA `resolve_kernel` is ONE block of 1,024 threads: it claims missing blocks with an atomicCAS on the
// page table, does a block-wide exclusive scan (`block_scan`, warp shuffles + shared memory), and runs the CLOCK
// victim sweep.  icpx 2026.1 has no shuffle and the whole function is a single block by construction, so the
// resolve is executed by ONE work-item walking the identical algorithm:
//
//   * phase 1 walks every selection in order, so the "claim once" test that the CUDA atomicCAS performs becomes
//     the plain read/modify; `lookups` accumulates the same count.
//   * phase 2 reproduces the parallel sweep exactly by simulating the 1,024 lanes per scan: it computes every
//     lane's `mine`/`cand` from the start-of-iteration state, builds the exclusive prefix `rank`, finds the same
//     `cut` (the lane with rank == want-1), then applies the writes in lane order.  This yields the same victim
//     set, the same `hand` update and the same reference-bit clearing as the scan version.
//   * phase 3 re-points the table in the same k order.
// Nothing here touches shared memory, atomics or a scratch buffer, so the submission is self-contained and
// cannot race an out-of-order queue.
//
// The reset/table kernels are flat ranges.  `resolve` and `copy` are separate submits with an explicit event
// dependency (the copy reads `ctl[2]`, which resolve writes; icpx 2026.1 queues are out-of-order).  The host
// copies (`kv_ring_restore`, `kv_stage_from_host`) become `queue::memcpy` on the USM pointers.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {

namespace {

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// reset: every block evicted, slots free, counters zero.  One work-item per index (a flat range in place of the
// CUDA grid-stride loop).
sycl::event reset_launch(int32_t* page_table, int32_t* slot_block, int32_t* slot_stamp, int32_t* slot_ref,
                         int32_t* ctl, long long n_blocks, long long n_slots, sycl::queue& qref) {
    const size_t total = (size_t) (n_blocks + n_slots + kKvCtlInts);
    return qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
            const long long i = (long long) wid;
            if (i < n_blocks) {
                page_table[i] = -1;
            } else if (i < n_blocks + n_slots) {
                const long long j = i - n_blocks;
                slot_block[j] = -1;
                slot_stamp[j] = -1;
                slot_ref[j] = 0;
            } else {
                const long long k = i - n_blocks - n_slots;
                if (k < kKvCtlInts) ctl[k] = 0;
            }
        });
    });
}

// The whole resolve, serially: one work-item reproduces the CUDA block's phases (see the file header).
sycl::event resolve_launch(int32_t* page_table, int32_t* slot_block, int32_t* slot_stamp, int32_t* slot_ref,
                           int32_t* ctl, int32_t* miss_block, int32_t* miss_slot, long long n_slots,
                           const int32_t* ids, const int32_t* steps, int n_q, int cap, int page_size,
                           sycl::queue& qref) {
    return qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            const int epoch = ctl[0] + 1;
            int n_miss = 0, lookups = 0;
            // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once.
            for (int qq = 0; qq < n_q; ++qq) {
                const int width = steps[qq * kStepCount + kStepWidth];
                const int32_t* qi = ids + (size_t) qq * cap;
                for (int i = 0; i < width; ++i) {
                    const int b = qi[i] / page_size;
                    if (i > 0 && qi[i - 1] / page_size == b) continue;   // ids ascending: one lookup per block
                    ++lookups;
                    const int sl = page_table[b];
                    if (sl >= 0) {
                        slot_stamp[sl] = epoch;
                        slot_ref[sl] = 1;
                    } else if (sl == -1 && page_table[b] == -1) {
                        page_table[b] = -2;
                        miss_block[n_miss++] = b;
                    }
                }
            }
            // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is
            //    never taken; a referenced one loses its bit as the hand passes it and is taken on the next pass.
            const int need = n_miss, n = (int) n_slots;
            int hand = ctl[1], got = 0;
            for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
                int jt[RT];
                bool mine[RT];
                bool cand[RT];
                int rk[RT];
                for (int t = 0; t < RT; ++t) {
                    const long long j = ((long long) hand + t) % n;
                    jt[t] = (int) j;
                    mine[t] = (slot_stamp[j] == epoch);
                    cand[t] = !mine[t] && (slot_block[j] < 0 || slot_ref[j] == 0);
                }
                int run = 0;
                for (int t = 0; t < RT; ++t) {
                    rk[t] = run;
                    if (cand[t]) ++run;
                }
                const int total = run;
                const int want = need - got;
                int cut = RT;
                for (int t = 0; t < RT; ++t) {
                    if (cand[t] && rk[t] == want - 1) {   // the hand stops just past the last slot taken
                        cut = t + 1;
                        break;
                    }
                }
                for (int t = 0; t < RT; ++t) {
                    if (cand[t] && rk[t] < want) {
                        miss_slot[got + rk[t]] = jt[t];
                        slot_stamp[jt[t]] = epoch;   // taken: a sweep that wraps around must not take it twice
                    } else if (t < cut && !mine[t]) {
                        slot_ref[jt[t]] = 0;
                    }
                }
                got += total < want ? total : want;
                hand = (int) (((long long) hand + cut) % n);
            }
            // 3. re-point the table; the copy kernel fills the slots
            const int placed = got < need ? got : need;
            for (int k = 0; k < need; ++k) {
                const int b = miss_block[k];
                if (k >= placed) {
                    page_table[b] = -1;
                    continue;
                }
                const int sl = miss_slot[k];
                const int old = slot_block[sl];
                if (old >= 0) page_table[old] = -1;
                slot_block[sl] = b;
                slot_stamp[sl] = epoch;
                slot_ref[sl] = 1;
                page_table[b] = sl;
            }
            ctl[0] = epoch;
            ctl[1] = hand;
            ctl[2] = placed;
            if (placed < need) ctl[3] = 1;
            unsigned long long* c = reinterpret_cast<unsigned long long*>(ctl + 4);
            c[0] += (unsigned long long) placed;
            c[1] += (unsigned long long) lookups;
            c[2] += 1ull;
        });
    });
}

// copy: one work-item per missed block, copying each 16-byte run from the host copy into its slot.
sycl::event copy_launch(const int32_t* ctl, const int32_t* miss_block, const int32_t* miss_slot, long long n_slots,
                        const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
                        uint8_t* dst0, uint8_t* dst1, uint8_t* dst2, uint8_t* dst3, int len0, int len1, int len2,
                        int len3, int rn, sycl::event dep, sycl::queue& qref) {
    return qref.submit([&](sycl::handler& h) {
        h.depends_on(dep);   // the copy reads ctl[2], which the resolve writes
        h.parallel_for(sycl::range<1>((size_t) n_slots), [=](sycl::id<1> wid) {
            const long long kk = (long long) wid;
            const int nneed = ctl[2];
            if (kk >= nneed) return;
            const long long b = miss_block[kk], sl = miss_slot[kk];
            const uint8_t* srcs[4] = {src0, src1, src2, src3};
            uint8_t* dsts[4] = {dst0, dst1, dst2, dst3};
            const int lens[4] = {len0, len1, len2, len3};
            for (int a = 0; a < rn; ++a) {
                const sycl::uint4* s = reinterpret_cast<const sycl::uint4*>(srcs[a] + b * lens[a]);
                sycl::uint4* d = reinterpret_cast<sycl::uint4*>(dsts[a] + sl * lens[a]);
                const int cnt = lens[a] / 16;
                for (int i = 0; i < cnt; ++i) d[i] = s[i];
            }
        });
    });
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        reset_launch(m.page_table, m.slot_block, m.slot_stamp, m.slot_ref, m.ctl, m.n_blocks, m.n_slots, *qp);
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_stream: reset: %s\n", e.what());
        std::exit(1);
    }
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    const Runs r = runs_of(slots, host, fmt, s);
    const int nq = (int) n_q, cp = (int) cap, ps = (int) s.page_size;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event ev = resolve_launch(m.page_table, m.slot_block, m.slot_stamp, m.slot_ref, m.ctl, m.miss_block,
                                        m.miss_slot, m.n_slots, ids, steps, nq, cp, ps, *qp);
        copy_launch(m.ctl, m.miss_block, m.miss_slot, m.n_slots, r.src[0], r.src[1], r.src[2], r.src[3], r.dst[0],
                    r.dst[1], r.dst[2], r.dst[3], r.len[0], r.len[1], r.len[2], r.len[3], r.n, ev, *qp);
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_stream: resolve: %s\n", e.what());
        std::exit(1);
    }
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    if (n_blocks <= 0) return;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_blocks), [=](sycl::id<1> wid) {
                const long long i = (long long) wid;
                page_table[i] = (int32_t) (i % n_slots);
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_stream: ring table: %s\n", e.what());
        std::exit(1);
    }
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        for (int64_t b = b0; b < b1;) {
            const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
            for (int a = 0; a < r.n; ++a)
                qp->memcpy(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]));
            b += run;
        }
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_stream: ring restore: %s\n", e.what());
        std::exit(1);
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        for (int a = 0; a < r.n; ++a)
            qp->memcpy(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]));
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_stream: stage: %s\n", e.what());
        std::exit(1);
    }
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr) return r;
    // A synchronous read of the counters (debug and the end-of-request summary); on failure keep the zeroed
    // result, matching the CUDA path's graceful return rather than aborting.
    try {
        sycl::queue* qp = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
        qp->memcpy(c, m.ctl, sizeof(c));
        qp->wait();
    } catch (const sycl::exception&) {
        return r;
    }
    unsigned long long u[3];
    std::memcpy(u, c + 4, sizeof(u));
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
