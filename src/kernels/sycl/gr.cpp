// src/kernels/sycl/gr.cpp - SYCL port of src/kernels/cuda/gr.cu (P2.S2: the gated residual / hyper-connection).
//
// See include/strata/kernels/gr.hpp for the semantics, for why the weights are bf16, and for the history of this
// kernel's 134x-off-the-floor first version.  The short form: it was launched `<<<1, 256>>>` so that `xn`/`xq`
// fit in shared memory, which used ONE of 48 SMs and cost 262 ms/token.
//
// THE SHAPE OF THE FIX is preserved by the port: each projection is an ordinary GEMV and is mapped the way the
// shapes want -
//
//     norm    hc work-items, one per residual stream      (each needs its own reduction)
//     down    ONE WORK-GROUP PER OUTPUT ROW, lanes stride the 10240 reduction axis
//     gate    ONE WORK-GROUP PER OUTPUT ROW, lanes stride the 320 reduction axis
//     mean    a flat elementwise pass over n_embd
//
// The CUDA warp reductions use `__shfl_down_sync`; icpx 2026.1 has no subgroup-shuffle surface, so each becomes
// a work-group reduction over a local-accessor buffer with `sycl::group_barrier` (global == groups*local, which
// Arc accepts).  The lane partials are the same and only the tree order differs.  The `hc` blocks and
// `hc_lr`/`hc_dim` grids are the same; the `gr_write` shared per-stream weight is recomputed per element rather
// than staged, the same value from the same `inject` input.
//
// The `native_mmvf` path calls the already-ported native kernels unchanged (weighted F32 RMSNorm, bf16/f32 MMVF,
// the learned SiLU and the pinned combine arithmetic); only the final mixer's unfused order is retained exactly
// as the CUDA source retains it.
#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

/// The bf16 conversions live in `strata/kernels/bf16_bits.hpp`, and this file used to carry its own copies.
/// Two files with a private copy of a conversion whose failure mode is silent wrong bits is one too many -
/// and the reason the shared header exists is that `bf16` and `fp16` are NOT two spellings of one idea, so a
/// routine written for one and reused for the other is wrong in a way that still produces numbers.
///
/// The local names are kept as thin wrappers so the body below is untouched by the swap.
inline float activation_f32(float x) { return x; }
inline float activation_f32(uint16_t x) { return f32_from_bf16(x); }
inline void store_activation(float* dst, int i, float x) { dst[i] = x; }
inline void store_activation(uint16_t* dst, int i, float x) { dst[i] = bf16_from_f32(x); }

/// silu and sigmoid exactly as `ref/gr.py` writes them - in FLOAT, not double.
///
/// This is the opposite of `shared_expert.cu`, which uses double, and the difference is not stylistic:
/// `ref/gr.py`'s arrays are float32, so `x / (1.0 + np.exp(-x))` evaluates in float32 under NEP 50, while
/// `ref/moe.py` works in float64.  Matching the reference's PRECISION is part of transcribing it.
inline float silu_f(float x) { return x / (1.0f + sycl::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

/// The FP32 block-wide sum, for the reason the review's G5 states: this is a GeForce part and FP64 runs at a
/// small fraction of FP32, so a `double` reduction turns a bandwidth kernel into a latency kernel.  icpx 2026.1
/// has no shuffle surface, so the warp/block tree becomes a work-group tree over a local-accessor buffer.
inline float group_sum_f(const sycl::nd_item<1>& item, float* scratch, float v) {
    const int t = (int) item.get_local_id(0);
    const int n = (int) item.get_local_range(0);
    scratch[t] = v;
    sycl::group_barrier(item.get_group());
    for (int off = n / 2; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        sycl::group_barrier(item.get_group());
    }
    return scratch[0];
}

/// `hc` work-items, one per residual stream: per-stream RMSNorm, then the activated stream in both forms.
///
/// ggml_rms_norm reduces over ne[0], and ne[0] is n_embd, so the reduction is PER STREAM.  The `[n_embd, hc]`
/// gamma then scales each stream individually.
///
/// A per-work-item accumulator is only a sum when the block is narrower than the row, which is why the CUDA
/// version is a block reduction: with `n_embd` equal to the thread count every thread would hold exactly one
/// element and normalise by its own square.  One work-item per stream walking the row serially is the
/// scratch-free form; the FP32 accumulator matches `gr_norm_kernel`'s deliberate switch away from `double`.
template <bool FP32_ACT>
void launch_gr_norm(const float* R, const float* w_norm, float eps, int n_embd, float* xn, uint16_t* xq, int hc,
                    sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) hc), [=](sycl::id<1> cid) {
            const int c = (int) cid[0];
            const float* Rc = R + (size_t) c * n_embd;
            float* xnc = xn + (size_t) c * n_embd;
            uint16_t* xqc = xq + (size_t) c * n_embd;
            float ss = 0.0f;
            for (int d = 0; d < n_embd; ++d) {
                const float v = Rc[d];
                ss += v * v;
            }
            const float ms = ss / (float) n_embd;
            const float rs = 1.0f / sycl::sqrt(ms + eps);
            for (int d = 0; d < n_embd; ++d) {
                const float x = Rc[d] * rs * w_norm[(size_t) c * n_embd + d];
                xnc[d] = x;
                if constexpr (!FP32_ACT) xqc[d] = bf16_from_f32(x);
            }
        });
    });
}

/// `lo = silu((bf16(xn) @ w_down.T) / hc)`, ONE WORK-GROUP PER OUTPUT ROW with its lanes splitting the reduction.
template <typename Activation>
void launch_gr_down(const Activation* xq, const uint16_t* w_down, int hc_dim, int hc_lr, int hc, Activation* lq,
                    sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) WARPS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc_lr * THREADS, (size_t) THREADS), [=](sycl::nd_item<1> item) {
            const int k = (int) item.get_group(0);
            const int t = (int) item.get_local_id(0);
            // Lanes within a warp read consecutive addresses, because the full-range work-item index is
            // contiguous across a row; the 8 warp partials meet in the local buffer.
            const uint16_t* row = w_down + (size_t) k * hc_dim;
            float acc = 0.0f;
            for (int i = t; i < hc_dim; i += THREADS)
                acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
            const float total = group_sum_f(item, &part[0], acc);
            if (t == 0) store_activation(lq, k, silu_f(total / (float) hc));
        });
    });
}

/// `gated[i] = xn[i] * sigmoid(bf16(lo) @ w_up.T)`, one work-group (32 lanes) per output row, lanes striding hc_lr.
template <typename Activation>
void launch_gr_gate(const Activation* lq, const uint16_t* w_up, const float* xn, int hc_dim, int hc_lr, float* gated,
                    sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc_dim * 32, (size_t) 32), [=](sycl::nd_item<1> item) {
            const int i = (int) item.get_group(0);
            const int lane = (int) item.get_local_id(0);
            const uint16_t* row = w_up + (size_t) i * hc_lr;
            float acc = 0.0f;
            for (int k = lane; k < hc_lr; k += 32)
                acc += activation_f32(lq[k]) * f32_from_bf16(row[k]);
            const float total = group_sum_f(item, &part[0], acc);
            if (lane == 0) gated[i] = xn[i] * sigmoid_f(total);
        });
    });
}

/// `mixed[d] = mean over c of gated[c][d]`.  Flat elementwise: the mean is over the streams, which are strided by
/// n_embd, so no cross-work-item reduction is needed at all.
void launch_gr_mean(const float* gated, int n_embd, int hc, float* mixed, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> did) {
            const int d = (int) did[0];
            float m = 0.0f;
            for (int c = 0; c < hc; ++c) m += gated[(size_t) c * n_embd + d];
            mixed[d] = m / (float) hc;
        });
    });
}

/// `inject[c] = bf16(xn) @ w_inject[c]`, one value per stream: ONE WORK-GROUP PER STREAM, lanes stride hc_dim.
template <typename Activation>
void launch_gr_inject(const Activation* xq, const uint16_t* w_inject, int hc_dim, int hc, float* inject,
                      sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) 32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) hc * 32, (size_t) 32), [=](sycl::nd_item<1> item) {
            const int c = (int) item.get_group(0);
            const int lane = (int) item.get_local_id(0);
            const uint16_t* row = w_inject + (size_t) c * hc_dim;
            float acc = 0.0f;
            for (int i = lane; i < hc_dim; i += 32)
                acc += activation_f32(xq[i]) * f32_from_bf16(row[i]);
            const float total = group_sum_f(item, &part[0], acc);
            if (lane == 0) inject[c] = total;
        });
    });
}

void launch_gr_write(const float* R, const float* block_out, const float* inject, int n_embd, int hc, float* out,
                     sycl::queue& qref) {
    const long long n = (long long) hc * n_embd;
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> iid) {
            const long long i = (long long) iid[0];
            const int c = (int) (i / n_embd), d = (int) (i % n_embd);
            // w = 2*sigmoid(inject/hc); every stream adds the SAME block output, only the weight differs.
            const float w = 2.0f * sigmoid_f(inject[c] / (float) hc);
            out[i] = R[i] + block_out[d] * w;
        });
    });
}

}  // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) s.hc * (size_t) s.n_embd;
    // ONE table, indexed by the same `k` that assigns the pointers below.  The first version of this function
    // sized three locals named `xn`, `xq`, `lq` in a DIFFERENT order from the pointer assignments - `xq` got
    // hc_lr*2 while `lq` got hc_dim*4 - so for the real geometry xq was given 640 bytes where it needed
    // 20,480 and every region overlapped its neighbour.
    //
    // What that looked like is worth recording: `inject` came out ~20% wrong, and rewriting its reduction
    // from block-wide to warp-wide produced BYTE-IDENTICAL wrong answers.  Two different reductions agreeing
    // exactly is not a coincidence to investigate - it means the reduction is not where the bug is, and the
    // inputs are.  Chasing the reduction instead cost most of a round.
    const size_t sz[5] = {
        hc_dim * sizeof(float),                // 0: xn
        hc_dim * sizeof(uint16_t),             // 1: xq
        (size_t) s.hc_lr * sizeof(uint16_t),   // 2: lq
        hc_dim * sizeof(float),                // 3: gated
        (size_t) s.hc_lr * sizeof(float),      // 4: lo (FP32 activation experiment)
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0) return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n",
                     ws.bytes, gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int) s.n_embd, hc = (int) s.hc, hc_lr = (int) s.hc_lr;
    const int hc_dim = (int) (s.hc * s.n_embd);

    // One setting selects every projection in this call. Captured graphs retain these kernel variants.
    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;

    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        if (use_native) {
            if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
                throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
            // The native kernels are submitted to the same queue; the queue is out-of-order, so each dependent
            // stage is flushed before the next (the CUDA stream ordering this graph relied on is explicit here).
            // The resolved default queue is passed so a null `stream` still reaches a real queue.
            native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, q);
            qref.wait();
            bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, q);
            qref.wait();
            native_gr_down_silu(ws.lo, hc_lr, hc, q);
            qref.wait();
            bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, q);
            qref.wait();
            // The existing null-injection contract identifies the final mixer.
            // Pinned qwen4exp uses fused HC pre for layers and the unfused graph for il=-1.
            native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, q);
            qref.wait();
        } else if (use_fp32) {
            launch_gr_norm<true>(R, w_norm, eps, n_embd, ws.xn, ws.xq, hc, qref);
            qref.wait();
            launch_gr_down<float>(ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo, qref);
            qref.wait();
            launch_gr_gate<float>(ws.lo, w_up, ws.xn, hc_dim, hc_lr, ws.gated, qref);
            qref.wait();
        } else {
            launch_gr_norm<false>(R, w_norm, eps, n_embd, ws.xn, ws.xq, hc, qref);
            qref.wait();
            launch_gr_down<uint16_t>(ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq, qref);
            qref.wait();
            launch_gr_gate<uint16_t>(ws.lq, w_up, ws.xn, hc_dim, hc_lr, ws.gated, qref);
            qref.wait();
        }
        if (!use_native) launch_gr_mean(ws.gated, n_embd, hc, mixed, qref);
        // Absent for the final mixer, and then nothing is written - computing an injection nobody reads would be
        // a claim, not a convenience.
        if (w_inject != nullptr) {
            if (use_native) {
                bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, q);
            } else if (use_fp32) {
                launch_gr_inject<float>(ws.xn, w_inject, hc_dim, hc, inject, qref);
            } else {
                launch_gr_inject<uint16_t>(ws.xq, w_inject, hc_dim, hc, inject, qref);
            }
        }
        if (stream == nullptr) qref.wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gr_read launch: %s\n", e.what());
        std::exit(1);
    }
}

void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s, float* R_out,
              void* stream) {
    if (s.n_embd <= 0 || s.hc <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        if (native_mmvf)
            native_gr_post(R, block_out, inject, R_out, (int) s.n_embd, (int) s.hc, q);
        else
            launch_gr_write(R, block_out, inject, (int) s.n_embd, (int) s.hc, R_out, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gr_write launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
