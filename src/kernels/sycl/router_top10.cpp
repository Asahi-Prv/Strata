// src/kernels/sycl/router_top10.cpp - SYCL port of src/kernels/cuda/router_top10.cu (P2.S2: the MoE router).
//
// P2.S2's spec: "BF16 GEMV, softmax / top-k / renormalize per docs/semantics.md; emits (expert_id, weight) x 10
// per token".  This is the softmax/top-k/renormalize half; the BF16 GEMV that produces the logits is a
// separate kernel and is NOT here.
//
// The semantics are transcribed from `ref/moe.py::router`, which is itself transcribed from llama-graph.cpp:
//
//     p   = softmax(logits)                       over ALL experts, not over the selected subset
//     ids = stable argsort(-p)[:k]                ties break by INDEX, ascending
//     w   = gather(p, ids)
//     s   = max(sum(w), 2**-14)                   ggml_clamp(..., 2**-14, INF)
//     return ids, w / s
//
// The two-step structure matters and is not cosmetic: computing softmax over the selected subset instead is
// mathematically identical (softmax is shift-invariant) but it moves the renormalisation, and the CLAMP is
// part of the renormalisation.  Reproducing the gather form is what makes the clamp land in the same place.
//
// WHY THIS IS NO LONGER "NAIVE BY DESIGN", AND WHAT THE OLD REASONING GOT WRONG.
//
// The first version was ONE THREAD PER TOKEN: k passes over the experts, each pass recomputing
// `exp((double) l[e] - mx)` for every expert.  Its comment justified that with
//
//     "5,120 operations per token against 2.36e9 weights of expert matvec - three orders of magnitude smaller
//      than the thing it feeds, so the simplest correct version is also fast enough"
//
// **That is a statement about OPERATION COUNT and it says nothing about TIME, because all 5,120 operations were
// on ONE THREAD while the matvec has thousands.**  Measured (round 218, stage-by-stage inside one block):
//
//     moe_layer   4.077 ms      router_top10   3.387 ms      bf16 gemv   0.270 ms
//
// 3.39 ms per layer x 48 layers = 163 ms of a 289 ms token - **56% of the whole forward pass, in a kernel that
// reads 2.6 MB.**  It was invisible in `router_top10_parity` because that test runs the kernel ONCE and checks
// the right ids, and invisible in every per-layer test for the same reason.
//
// WHAT IS PARALLELISED AND WHAT IS DELIBERATELY NOT:
//
//   * the MAX is a tree reduction of `fmaxf`, which is exact and order-independent - bit-identical to the
//     serial scan it replaces;
//   * the 512 EXPONENTIALS are computed ONCE each, in parallel.  The old kernel computed them ELEVEN TIMES
//     (once for the softmax, then again inside every one of the k passes).  This is where the 3.4 ms was;
//   * the DOUBLE SUM is still accumulated on ONE thread in ASCENDING order.  512 double adds is about a
//     microsecond and it keeps the accumulation order - and therefore the last bits - identical to the
//     reference-faithful serial version.  Parallelising it would be a free speedup and a silent change to the
//     values, and the exp was the cost, not this;
//   * the SELECTION is by RANK rather than by k repeated maxima:
//
//         rank(e) = #{ f : p[f] > p[e] }  +  #{ f < e : p[f] == p[e] }
//
//     which IS "stable descending argsort, ties by ascending index" - the same order the repeated-maximum loop
//     produced - computed in ONE parallel pass over the experts instead of k serial ones.  It is O(n^2)
//     comparisons and that is the right trade here: n = 512, every comparison is independent, and the old
//     version was O(k*n) with a serial `exp` inside.
//
// THE SYCL PORT.  icpx 2026.1 exposes no warp-shuffle surface, so the BLOCK reductions of the CUDA source (the
// fmax tree, the parallel exponentials, the k rounds of block argmax) cannot be transcribed as shuffles.  This
// port is therefore ONE WORK-ITEM PER TOKEN, walking the experts serially - the same shape `native_gr_norm`
// uses - which reproduces the reference-faithful serial version EXACTLY: the max scans ascending, the double sum
// ascends, the selection scans `e` ascending with a strict `>` (lowest index wins a tie), and the renormalise
// preserves order.  It re-derives each expert's exponential from the stored max rather than caching it in shared
// memory (the CUDA `s_ex` buffer); the double computation is deterministic, so `p[e] = (float)(exp(l[e]-mx) *
// inv)` is bit-identical to the CUDA's cached read and only the per-token wall time differs.
#include "strata/kernels/router_top10.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

/// One work-item per token.  `n_tokens` is 1 in decode; the flat range keeps the batch case working without a
/// second code path.  The `taken` set of the k-pass selection is a private `sel` array rather than the CUDA
/// source's shared per-expert bitmap: k <= 64 is already enforced by the caller, and the membership scan is the
/// same O(k) comparison the original serial version performed.
void router_top10_run(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                      sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> tid) {
            const int t = (int) tid;
            const float* l = logits + (size_t) t * n_expert;
            int* ids_row = ids + (size_t) t * k;
            float* w_row = weights + (size_t) t * k;

            // ---- softmax over ALL experts, for stability: the max.  A scan of `fmaxf` is EXACT and
            // order-independent, so this is bit-identical to the serial scan it replaces.
            float mx = -INFINITY;
            for (int e = 0; e < n_expert; ++e) mx = sycl::fmax(mx, l[e]);

            // ---- the sum, ascending, on one work-item.  See the note above on why this is NOT parallelised;
            // the exponentials are re-derived from `mx` below rather than cached.
            double sum = 0.0;
            for (int e = 0; e < n_expert; ++e) sum += sycl::exp((double) l[e] - (double) mx);
            const float inv = (float) (1.0 / sum);

            // ---- the selection, in k passes.  Scanning `e` ascending with a strict `>` keeps the LOWEST index
            // on a tie, which is exactly the rule the rank loop spelled out as `else if (f < e && pf == pe)
            // ++rank`.  A stable descending top-k, ties by index.
            //
            // `p[e] = (float)(exp((double) l[e] - (double) mx) * inv)` reproduces the CUDA expression exactly:
            // `(float)(s_ex[e] * inv)` with `inv` a FLOAT.
            int sel[64];
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;               // a sentinel that loses to every real index
                for (int e = 0; e < n_expert; ++e) {
                    bool taken = false;
                    for (int j = 0; j < i; ++j)
                        if (sel[j] == e) { taken = true; break; }
                    if (taken) continue;
                    const float pe = (float) (sycl::exp((double) l[e] - (double) mx) * (double) inv);
                    if (pe > bv) { bv = pe; bi = e; }
                }
                if (bi < n_expert) {
                    sel[i] = bi;
                    ids_row[i] = bi;
                    w_row[i] = bv;
                } else {
                    sel[i] = -1;
                    ids_row[i] = 0;
                    w_row[i] = 0.0f;
                }
            }

            // ---- renormalise, with ggml's lower clamp.  Order preserved.
            double s = 0.0;
            for (int i = 0; i < k; ++i) s += (double) w_row[i];
            const double sc = sycl::fmax(s, 6.103515625e-05);       // 2**-14
            for (int i = 0; i < k; ++i) w_row[i] = (float) ((double) w_row[i] / sc);
        });
    });
}

}  // namespace

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        router_top10_run(logits, n_tokens, n_expert, k, ids, weights, q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "router_top10 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
