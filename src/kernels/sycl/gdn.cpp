// src/kernels/sycl/gdn.cpp - SYCL port of src/kernels/cuda/gdn.cu, the gated delta-net's non-projection parts.
//
// See include/strata/kernels/gdn.hpp for the state layout (S, h_v, S) and for why the recurrence needs no barrier
// at all: every line of it touches only one (j, h) column, so one thread owns a column end to end.
//
// The CUDA recurrence is already barrier-free, which makes the port unusually direct: the `gdn_step_kernel`'s
// `<<<grid, JTHREADS>>>` launch becomes a flat range<1>(S*h_v) with ONE WORK-ITEM PER (h, j) COLUMN, so the
// column is still owned end to end by a single work-item and the whole "no barrier" property survives the port.
// The CUDA kernel staged `k`/`q` rows and the per-head `dec`/`beta` scalars in shared memory so a warp could read
// each with a single broadcast; the port drops that staging and reads the caller's buffers directly (the same
// values, in the same order, so every product is identical) - the broadcast was an optimisation, not semantics.
//
// `gdn_conv_step`, `gdn_beta_gate` and the two norms are each one thread per row / per channel in CUDA, so they
// become one work-item per row with the reduction walked serially.  The L2 and output norms accumulate in DOUBLE,
// as `ref/gdn.py` does, so the serial sum matches the reference's precision and only the (unavoidable) summation
// order differs from the CUDA warp shuffle tree.
#include "strata/kernels/gdn.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int JTHREADS = 32;   ///< threads along j, the state's fast axis
/// Heads staged per block.  **THIS IS A LATENCY-HIDING KNOB, NOT A CAPACITY ONE, AND IT WAS SET WRONG.**
///
/// The CUDA kernel is one thread per `(h, j)` COLUMN of the state, so the thread count is fixed at
/// `S * h_v` = 128 * 48 = **6,144** no matter how the work is packed - and `MAX_H` decides only how many of
/// those columns share a block.  A flat range<1> has no notion of "heads per block" at all, so `MAX_H` no longer
/// exists on the launch side; the per-column work is unchanged, which is the only thing the knob was there to
/// hide behind.
constexpr int MAX_H = 1;       ///< heads staged in shared memory per block (kept for the record; see above)

/// ggml_compute_softplus_f32: log1p(exp(x)), with the large-x branch that avoids overflow.
inline float softplus_f(float x) { return x > 20.0f ? x : sycl::log1p(sycl::exp(x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) {
        std::fprintf(stderr, "gdn_step: S = %lld exceeds the staged 128 (`ks`/`qs` are [8][128])\n",
                     (long long) s.S);
        std::exit(1);
    }
    const int S = (int) s.S, h_k = (int) s.h_k, h_v = (int) s.h_v;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-item per (h, j) column, j fastest, so the recurrence's column ownership is preserved exactly.
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) S * (size_t) h_v), [=](sycl::id<1> idx) {
                const int hh = (int) idx[0] / S;   // value head
                const int j = (int) idx[0] % S;    // state column
                const int src = hh % h_k;          // MODULO head pairing
                const float dec = sycl::exp(gate[hh]);  // decay BEFORE the rank-1 update - PROPERTY 5
                const float b = beta[hh];

                float* col = state + (size_t) hh * S + j;        // (S, h_v, S) with j fastest
                const size_t stride = (size_t) h_v * S;

                // pass 1: decay the state and contract it against k.
                float sk = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float st = col[(size_t) i * stride] * dec;
                    col[(size_t) i * stride] = st;
                    sk += st * k[(size_t) src * S + i];
                }
                const float d = (v[(size_t) hh * S + j] - sk) * b;
                // pass 2: the rank-1 update, then read out against q using the UPDATED state
                float dot = 0.0f;
                for (int i = 0; i < S; ++i) {
                    const float st = col[(size_t) i * stride] + k[(size_t) src * S + i] * d;
                    col[(size_t) i * stride] = st;
                    dot += st * q[(size_t) src * S + i];
                }
                o[(size_t) hh * S + j] = dot;
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_step launch: %s\n", e.what());
        std::exit(1);
    }
}

void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv, void* stream) {
    if (channels <= 0 || d_conv < 1) return;
    const int C = (int) channels, dc = (int) d_conv;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // `ggml_ssm_conv`: one work-item per channel; d_conv is 4, so the serial work is a 4-tap dot and a shift.
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> cid) {
                const int c = (int) cid[0];
                float* st = conv_state + (size_t) c * (dc - 1);
                const float* w = kW + (size_t) c * dc;
                float acc = 0.0f;
                for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[i];   // kernel[0] reads the OLDEST state row
                acc += x[c] * w[dc - 1];                                // the new input lands in the LAST tap
                out[c] = acc;
                for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];     // slide: drop oldest, append newest
                st[dc - 2] = x[c];
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_conv_step launch: %s\n", e.what());
        std::exit(1);
    }
}

void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) {
        std::fprintf(stderr, "gdn_l2_norm: cols = %lld exceeds the 1024-wide warp reduction\n", (long long) cols);
        std::exit(1);
    }
    const int C = (int) cols;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-item per row does the DOUBLE accumulation serially - the same precision `ref/gdn.py` uses.
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> rid) {
                float* p = x + (size_t) rid[0] * C;
                double acc = 0.0;
                for (int i = 0; i < C; ++i) acc += (double) p[i] * (double) p[i];
                const float inv = (float) (1.0 / sycl::sqrt(acc + (double) eps));
                for (int i = 0; i < C; ++i) p[i] *= inv;
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_l2_norm launch: %s\n", e.what());
        std::exit(1);
    }
}

/// `beta = sigmoid(beta)`, in place, over the `h_v` per-head scalars.
///
/// **THIS IS `build_layer_attn_linear` L889, AND LEAVING IT OUT WAS THE C1 BUG.**  The reference computes
/// `beta = ggml_sigmoid(ssm_beta @ cur)` and hands the RESULT to the delta net; `gdn_step` is documented as
/// `d[h,j] = (v[h,j] - sk[h,j]) * beta[h]` and applies no sigmoid of its own, so the layer glue owes it one.
/// The engine passed the raw pre-activation, which is unbounded and signed - the delta-net write strength then
/// stops being a fraction and the whole recurrence is wrong from layer 0.
///
/// **IT SURVIVED `gdn_parity` BECAUSE THAT TEST SUPPLIES ITS OWN `beta`.**  The kernel was always right; only
/// the glue between the projection and the kernel was wrong, and `ref/gdn.py` and `ref/model.py` both get it
/// right - so the two transcriptions agreed with each other and neither was compared against the C++ that runs.
void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    if (beta == nullptr || h_v <= 0) return;
    const int n = (int) h_v;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> iid) {
                const int i = (int) iid[0];
                beta[i] = sigmoid_f(beta[i]);
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_beta_gate launch: %s\n", e.what());
        std::exit(1);
    }
}

void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    if (h_v <= 0 || S <= 0) return;
    const int H = (int) h_v, SS = (int) S;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // y = rms_norm(o, eps) * ssm_norm * sigmoid(z), one work-item per head; the norm is the DOUBLE sum of
        // squares over that head's S values, exactly as `ref/gdn.py` forms it.
        qp->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) H), [=](sycl::id<1> hid) {
                const int hh = (int) hid[0];
                const float* po = o + (size_t) hh * SS;
                const float* pz = z + (size_t) hh * SS;
                float* py = y + (size_t) hh * SS;
                double acc = 0.0;
                for (int i = 0; i < SS; ++i) acc += (double) po[i] * (double) po[i];
                const float inv = (float) (1.0 / sycl::sqrt(acc / (double) SS + (double) eps));
                for (int i = 0; i < SS; ++i)
                    py[i] = po[i] * inv * ssm_norm[i] * sigmoid_f(pz[i]);
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_out_norm launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
