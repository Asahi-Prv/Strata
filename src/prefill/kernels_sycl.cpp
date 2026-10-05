// src/prefill/kernels_sycl.cpp - SYCL port of src/prefill/kernels.cu (plan v0.3 P5, the prompt-chunk kernels).
//
// Same public API as include/strata/prefill/kernels.hpp.  Every CUDA `<<<grid, block>>>` becomes a
// `parallel_for`; where the CUDA kernel used warp shuffles or `__syncthreads` (the GDN recurrence, the router,
// the INT8 KV append) the port follows the conventions the rest of the SYCL tree settled on (see
// src/kernels/sycl/native_gr_norm.cpp, bf16_gemv.cpp, native_router.cpp): a serial work-item reproduces the
// per-lane arithmetic, or a work-group with `local_accessor` + `group_barrier` reproduces the block reduction.
// icpx 2026.1 exposes no subgroup/hardware shuffle surface, so nothing here uses one.
//
// Reductions that only reorder a sum (the RMS norms, the INT8 KV max) are exact; the recurrence's cross-column
// normalisation sums in a different order than the CUDA warp tree, so it is not bit-identical, matching the
// documented tolerance of the existing SYCL kernels.
#include "strata/prefill/kernels.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/mrope.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int64_t N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
constexpr int RG = 4, RPG = S / RG;

inline sycl::queue* queue_of(void* stream) {
    return static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
}

inline float sigmoid_dev(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float softplus_dev(float x) { return x > 20.0f ? x : sycl::log1p(sycl::exp(x)); }

void fail(const char* what, const sycl::exception& e) {
    std::fprintf(stderr, "prefill %s: %s\n", what, e.what());
    std::exit(1);
}

template <bool HALF>
void blob_dequant_impl(const uint8_t* blob, uint16_t* gu16, uint16_t* d16, sycl::queue* q) {
    constexpr int64_t N_GU = 1280LL * 640, N_D = 2560LL * 160, TOTAL = N_GU + N_D;
    constexpr int64_t O_D_CODES = 1280LL * 640;
    constexpr int64_t O_GU_SC = O_D_CODES + 2560LL * 160;
    constexpr int64_t O_D_SC = O_GU_SC + 1280LL * 40 * 2;
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) TOTAL), [=](sycl::id<1> wi) {
            const int64_t i = (int64_t) wi[0];
            if (i < N_GU) {
                const int64_t row = i / 640, byte = i % 640;
                const uint8_t c = blob[row * 640 + byte];
                const uint8_t* sp = blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2;
                const float d = strata::kernels::f32_from_f16((uint16_t) (sp[0] | (sp[1] << 8)));
                uint16_t* o = gu16 + row * 2560 + byte * 4;
                for (int k = 0; k < 4; ++k) {
                    const float v = (float) (((c >> (2 * k)) & 3) - 1) * d;
                    o[k] = HALF ? strata::kernels::f16_from_f32(v) : strata::kernels::bf16_from_f32(v);
                }
            } else if (i < TOTAL) {
                const int64_t j = i - N_GU, row = j / 160, byte = j % 160;
                const uint8_t c = blob[O_D_CODES + row * 160 + byte];
                const uint8_t* sp = blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2;
                const float d = strata::kernels::f32_from_f16((uint16_t) (sp[0] | (sp[1] << 8)));
                uint16_t* o = d16 + row * 640 + byte * 4;
                for (int k = 0; k < 4; ++k) {
                    const float v = (float) (((c >> (2 * k)) & 3) - 1) * d;
                    o[k] = HALF ? strata::kernels::f16_from_f32(v) : strata::kernels::bf16_from_f32(v);
                }
            }
        });
    });
}

}  // namespace

// ---------------------------------------------------------------- hyper-connection
void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream) {
    if (T <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (T * HC)), [=](sycl::id<1> wi) {
                const int64_t row = (int64_t) wi[0];      // t * HC + c
                const int c = (int) (row % HC);
                const float* r = R + row * N;
                const float* w = w_norm + c * N;
                float ss = 0.0f;
                for (int64_t d = 0; d < N; ++d) ss += r[d] * r[d];
                const float rs = sycl::rsqrt(ss / (float) N + eps);
                for (int64_t d = 0; d < N; ++d) {
                    const float v = r[d] * rs * w[d];
                    xn[row * N + d] = v;
                    xn16[row * N + d] = strata::kernels::bf16_from_f32(v);
                }
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gr_norm", e); }
}

void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream) {
    const int64_t n = T * LR;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const float x = lo[(int64_t) wi[0]] / (float) HC;
                lo16[(int64_t) wi[0]] = strata::kernels::bf16_from_f32(x / (1.0f + sycl::exp(-x)));
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gr_silu", e); }
}

void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h) {
    const int64_t n = T * N;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / N, d = i % N;
                float s = 0.0f;
                for (int c = 0; c < HC; ++c) {
                    const int64_t j = t * D + c * N + d;
                    s = sycl::fma(xn[j], sigmoid_dev(gated[j]), s);
                }
                s /= (float) HC;
                mixed[i] = s;
                if (mixed16 != nullptr) mixed16[i] = strata::kernels::bf16_from_f32(s);
                if (mixed_h != nullptr) mixed_h[i] = strata::kernels::f16_from_f32(s);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gr_mix", e); }
}

void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    const int64_t n = T * D;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / D, c = (i % D) / N, d = i % N;
                R[i] = sycl::fma(bo[t * N + d], 2.0f * sigmoid_dev(inj[t * inj_ld + c] / (float) HC), R[i]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gr_write", e); }
}

void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    const int64_t n = T * D;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                R[i] = e[(i / D) * N + i % N];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gr_broadcast", e); }
}

// ---------------------------------------------------------------- GDN
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T,
               void* stream) {
    const int64_t n = T * HV;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / HV, hh = i % HV;
                const float v = ab[t * 2 * HV + hh] + dt[hh];
                gate[i] = softplus_dev(v) * ssm_a[hh];
                beta[i] = sigmoid_dev(ab[t * 2 * HV + HV + hh]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gdn_gates", e); }
}

void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    if (T <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        // one work-item per conv channel walks the whole chunk (the CUDA `gdn_conv_kernel`)
        sycl::event conv = q->submit([&](sycl::handler& hh) {
            hh.parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> wi) {
                const int c = (int) wi[0];
                float v0 = history[c * 3], v1 = history[c * 3 + 1], v2 = history[c * 3 + 2];
                const float w0 = conv_w[c * 4], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];
                for (int64_t t = 0; t < T; ++t) {
                    const float x = qkv[t * C + c];
                    const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
                    h[t * C + c] = s / (1.0f + sycl::exp(-s));
                    v0 = v1; v1 = v2; v2 = x;
                }
                history[c * 3] = v0; history[c * 3 + 1] = v1; history[c * 3 + 2] = v2;
            });
        });
        // then the L2 norm of the 2*HK q/k heads of every token (the CUDA `gdn_l2_kernel`)
        q->submit([&](sycl::handler& hh) {
            hh.depends_on(conv);
            hh.parallel_for(sycl::range<1>((size_t) (T * 2 * HK)), [=](sycl::id<1> wi) {
                const int64_t r = (int64_t) wi[0];
                const int64_t t = r / (2 * HK);
                const int head = (int) (r % (2 * HK));
                float* x = h + (size_t) t * C + (size_t) head * S;
                float ss = 0.0f;
                for (int c = 0; c < S; ++c) { const float v = x[c]; ss += v * v; }
                const float s = sycl::rsqrt(ss + eps);
                for (int c = 0; c < S; ++c) x[c] = x[c] * s;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gdn_conv", e); }
}

void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    if (T <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        // One work-group per value head, S*RG work-items: the CUDA block layout (col = tid % S, rg = tid / S)
        // with the per-lane state held in a private RPG array and the cross-lane reductions through local
        // memory, since icpx 2026.1 has no warp shuffle.
        q->submit([&](sycl::handler& hh) {
            sycl::local_accessor<float, 1> skb(sycl::range<1>((size_t) S), hh);
            sycl::local_accessor<float, 1> sqb(sycl::range<1>((size_t) S), hh);
            sycl::local_accessor<float, 1> redb(sycl::range<1>((size_t) (RG * S)), hh);
            sycl::local_accessor<float, 1> sums(sycl::range<1>((size_t) S), hh);
            sycl::local_accessor<float, 1> bc(sycl::range<1>(1), hh);
            hh.parallel_for(sycl::nd_range<1>((size_t) HV * (S * RG), (size_t) (S * RG)),
                            [=](sycl::nd_item<1> item) {
                const int head = (int) item.get_group(0);
                const int tid = (int) item.get_local_id(0);
                const int col = tid % S, rg = tid / S;
                const int qh = head % HK;
                const size_t rs = (size_t) HV * S;
                float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
                float s[RPG];
                for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
                const float g_col = gamma[col];
                for (int64_t t = 0; t < T; ++t) {
                    const float* ht = h + t * C;
                    sycl::group_barrier(item.get_group());
                    if (tid < S) {
                        sqb[tid] = ht[qh * S + tid];
                        skb[tid] = ht[HK * S + qh * S + tid];
                    }
                    sycl::group_barrier(item.get_group());
                    const float g = sycl::exp(gate[t * HV + head]);
                    float kv = 0.0f;
                    for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], skb[rg * RPG + r], kv);
                    redb[rg * S + col] = kv;
                    sycl::group_barrier(item.get_group());
                    const float kv_col = redb[0 * S + col] + redb[1 * S + col] + redb[2 * S + col] + redb[3 * S + col];
                    const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
                    float o = 0.0f;
                    for (int r = 0; r < RPG; ++r) {
                        s[r] = sycl::fma(g, s[r], skb[rg * RPG + r] * delta);
                        o = sycl::fma(s[r], sqb[rg * RPG + r], o);
                    }
                    sycl::group_barrier(item.get_group());
                    redb[rg * S + col] = o;
                    sycl::group_barrier(item.get_group());
                    float oc = 0.0f;
                    if (rg == 0) {
                        oc = (redb[0 * S + col] + redb[1 * S + col] + redb[2 * S + col] + redb[3 * S + col])
                             * sycl::rsqrt((float) S);
                        sums[col] = oc * oc;
                    }
                    sycl::group_barrier(item.get_group());
                    if (tid == 0) {
                        float ss = 0.0f;
                        for (int c = 0; c < S; ++c) ss += sums[c];
                        bc[0] = ss;
                    }
                    sycl::group_barrier(item.get_group());
                    if (rg == 0) {
                        const float ss = bc[0];
                        const float v = oc * sycl::rsqrt(ss / (float) S + eps) * g_col *
                                        sigmoid_dev(z[t * HV * S + head * S + col]);
                        y[t * HV * S + head * S + col] = v;
                        y16[t * HV * S + head * S + col] = strata::kernels::f16_from_f32(v);
                    }
                }
                for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gdn_recurrence", e); }
}

// ---------------------------------------------------------------- MoE
void route(const float* logits, int32_t* ids, float* weights, int64_t T, void* stream) {
    if (T <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) T), [=](sycl::id<1> wi) {
                const int64_t t = (int64_t) wi[0];
                const float* lg = logits + t * 512;
                float values[512];
                for (int i = 0; i < 512; ++i) values[i] = lg[i];
                float mx = -INFINITY;
                for (int i = 0; i < 512; ++i) mx = sycl::fmax(mx, values[i]);
                float sum = 0.0f;
                for (int i = 0; i < 512; ++i) { values[i] = sycl::exp(values[i] - mx); sum += values[i]; }
                const float rcp = 1.0f / sum;
                for (int i = 0; i < 512; ++i) {
                    values[i] *= rcp;
                    if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
                }
                float bests[10];
                float selected_sum = 0.0f;
                for (int rank = 0; rank < 10; ++rank) {
                    float best = values[0];
                    int ex = 0;
                    for (int i = 1; i < 512; ++i) if (values[i] > best) { best = values[i]; ex = i; }
                    values[ex] = -INFINITY;
                    ids[t * 10 + rank] = ex;
                    bests[rank] = best;
                    selected_sum += best;
                }
                selected_sum = sycl::fmax(selected_sum, 6.103515625e-5f);
                const float inv = 1.0f / selected_sum;
                for (int rank = 0; rank < 10; ++rank) weights[t * 10 + rank] = bests[rank] * inv;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("route", e); }
}

void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    sycl::queue* q = queue_of(stream);
    try {
        blob_dequant_impl<false>(blob, gu16, down16, q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("blob_dequant", e); }
}

void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    sycl::queue* q = queue_of(stream);
    try {
        blob_dequant_impl<true>(blob, gu16, down16, q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("blob_dequant_f16", e); }
}

void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (n * 640)), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t r = i / 640, k = i % 640;
                const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
                h16[i] = strata::kernels::f16_from_f32(g / (1.0f + sycl::exp(-g)) * u);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("swiglu_interleaved", e); }
}

void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (n * 640)), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const float a = g[i];
                h16[i] = strata::kernels::f16_from_f32(a / (1.0f + sycl::exp(-a)) * u[i]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("swiglu_pair", e); }
}

void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    const int64_t per = width / 8;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (n * per)), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t r = i / per, j = i % per;
                reinterpret_cast<sycl::uint4*>(dst16)[r * per + j] =
                    reinterpret_cast<const sycl::uint4*>(x16)[(int64_t) src[r] * per + j];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gather_rows16", e); }
}

void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg,
                 float* bo, int64_t T, void* stream) {
    const int64_t n = T * N;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / N, d = i % N;
                float s = 0.0f;
                for (int k = 0; k < 10; ++k) s = sycl::fma(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
                bo[i] = s + shared[i] * sigmoid_dev(sg[t]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("moe_combine", e); }
}

// ---------------------------------------------------------------- QSA helpers
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> wi) {
                float* r = x + (int64_t) wi[0] * ld;
                float ss = 0.0f;
                for (int64_t c = 0; c < cols; ++c) ss += r[c] * r[c];
                const float s = sycl::rsqrt(ss / (float) cols + eps);
                for (int64_t c = 0; c < cols; ++c) r[c] = s * r[c] * w[c];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("rms_rows", e); }
}

void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, float freq_base, void* stream) {
    if (T <= 0) return;
    const float theta_scale = sycl::pow(freq_base, -2.0f / 64.0f);
    const int32_t* mtab = strata::kernels::mrope_table();
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (T * heads)), [=](sycl::id<1> wi) {
                const int64_t row = (int64_t) wi[0];
                const int64_t t = row / heads, hh = row % heads;
                float* p = x + t * ld + hh * dim;
                for (int pair = 0; pair < 32; ++pair) {
                    const float theta = (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) *
                                        sycl::pow(theta_scale, (float) pair);
                    const float c = sycl::cos(theta), s = sycl::sin(theta);
                    const float a = p[pair], b = p[pair + 32];
                    p[pair] = a * c - b * s;
                    p[pair + 32] = a * s + b * c;
                }
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("rope", e); }
}

void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    const int64_t n = T * 24 * 256;
    if (n <= 0) return;
    sycl::queue* sq = queue_of(stream);
    try {
        sq->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / (24 * 256), hh = (i / 256) % 24, d = i % 256;
                q[i] = q_full[t * 24 * 512 + hh * 512 + d];
            });
        });
        if (stream == nullptr) sq->wait();
    } catch (const sycl::exception& e) { fail("split_q", e); }
}

void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    const int64_t n = T * 24 * 256;
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                const int64_t t = i / (24 * 256), hh = (i / 256) % 24, d = i % 256;
                out16[i] = strata::kernels::f16_from_f32(
                    attn[i] * sigmoid_dev(q_full[t * 24 * 512 + hh * 512 + 256 + d]));
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("gate_attn", e); }
}

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host, const strata::kernels::KvHostPools* stage) {
    if (T <= 0) return;
    const strata::kernels::KvHostPools hhost = host ? *host : strata::kernels::KvHostPools{};
    const strata::kernels::KvHostPools hstage = stage ? *stage : strata::kernels::KvHostPools{};
    sycl::queue* q = queue_of(stream);
    try {
        // One work-item per (token, K/V, kv head, 64-value group).  The CUDA block's 64 threads become one
        // serial walk of the group, so the INT8 max and the per-group scale land in the same arithmetic.
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (T * 16)), [=](sycl::id<1> wi) {
                const int64_t idx = (int64_t) wi[0];
                const int64_t t = idx / 16;
                const int rem = (int) (idx % 16);
                const int is_v = rem & 1, kvh = (rem >> 1) & 1, g = (rem >> 2) & 3;
                const float* xbase = (is_v ? V : K) + t * 512 + kvh * 256 + g * 64;
                const int64_t pos = pos0 + t;
                const int64_t page = page_table[pos / page_size];
                const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
                const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
                if (k_pool != nullptr) {
                    uint16_t* pool = is_v ? v_pool : k_pool;
                    uint16_t* hpool = is_v ? hhost.v_pool : hhost.k_pool;
                    uint16_t* spool = is_v ? hstage.v_pool : hstage.k_pool;
                    for (int j = 0; j < 64; ++j) {
                        const uint16_t hb = strata::kernels::f16_from_f32(xbase[j]);
                        if (page >= 0) pool[row * 256 + g * 64 + j] = hb;
                        if (hpool != nullptr) hpool[row_id * 256 + g * 64 + j] = hb;
                        if (spool != nullptr) spool[row_id * 256 + g * 64 + j] = hb;
                    }
                    return;
                }
                float amax = 0.0f;
                for (int j = 0; j < 64; ++j) amax = sycl::fmax(amax, sycl::fabs(xbase[j]));
                const uint16_t sb = strata::kernels::f16_from_f32(amax / 127.0f);
                const float sf = strata::kernels::f32_from_f16(sb);
                int8_t* kq = is_v ? v_q : k_q;
                int8_t* hq = is_v ? hhost.v_q : hhost.k_q;
                int8_t* sq = is_v ? hstage.v_q : hstage.k_q;
                uint16_t* ksc = is_v ? v_scale : k_scale;
                uint16_t* hsc = is_v ? hhost.v_scale : hhost.k_scale;
                uint16_t* ssc = is_v ? hstage.v_scale : hstage.k_scale;
                for (int j = 0; j < 64; ++j) {
                    int qv = 0;
                    if (sf > 0.0f) {
                        qv = (int) sycl::rint(xbase[j] / sf);
                        qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);
                    }
                    if (page >= 0) kq[row * 256 + g * 64 + j] = (int8_t) qv;
                    if (hq != nullptr) hq[row_id * 256 + g * 64 + j] = (int8_t) qv;
                    if (sq != nullptr) sq[row_id * 256 + g * 64 + j] = (int8_t) qv;
                }
                if (page >= 0) ksc[row * 4 + g] = sb;
                if (hq != nullptr) hsc[row_id * 4 + g] = sb;
                if (sq != nullptr) ssc[row_id * 4 + g] = sb;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("kv_append", e); }
}

void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                y[(int64_t) wi[0]] = strata::kernels::f16_from_f32(x[(int64_t) wi[0]]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("to_f16", e); }
}

void round_f16(const float* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                const int64_t i = (int64_t) wi[0];
                y[i] = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(x[i]));
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("round_f16", e); }
}

void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = queue_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> wi) {
                y[(int64_t) wi[0]] = strata::kernels::bf16_from_f32(x[(int64_t) wi[0]]);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) { fail("to_bf16", e); }
}

}  // namespace strata::prefill
