// src/kernels/sycl/fused_gdn.cpp - SYCL port of src/kernels/cuda/fused_gdn.cu (see fused_gdn.hpp).
//
// The fused GDN block fuses the step and its output norm so one block owns one whole value head.  The CUDA
// kernels use warp shuffles (`__shfl_xor_sync`) for their small reductions; icpx 2026.1 exposes no
// subgroup-shuffle surface, so each reduction is a work-group reduction over a local-accessor buffer with
// `sycl::group_barrier` - the same lane partials, only the summation order differs (numerically equivalent,
// validated by max_rel on Arc).  Because a work-group reduction needs every work-item of the group to reach the
// barrier, the CUDA early-returns (`if (lane != 0) return;`) become guards around the write instead.
//
//   S <- g S + k (beta (v - g S^T k))^T      (g = exp(gate[head]), per value head; q/k shared by h_v / h_k heads)
//   o  = (S^T q) / sqrt(128)
//   y  = rmsnorm(o) * gamma * sigmoid(z)
#include "strata/kernels/fused_gdn.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per thread

// A work-group-wide sum over `v`, result broadcast to every work-item.  The partials live in a local-accessor
// buffer because icpx 2026.1 has no subgroup reduction; the tree order is not the CUDA warp XOR order, which is
// the only difference from the source.
inline float group_sum(const sycl::nd_item<1>& item, float* scratch, float v) {
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

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    const int n_heads = channels / S;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-group per 128-channel head; the L2 norm of a head is a work-group reduction of the silent
        // upper part's square sum.  `channels` is a multiple of S, so global == groups*local.
        q->submit([&](sycl::handler& hh) {
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) S), hh);
            hh.parallel_for(sycl::nd_range<1>((size_t) channels, (size_t) S), [=](sycl::nd_item<1> item) {
                const int blk = (int) item.get_group(0);
                const int c = blk * S + (int) item.get_local_id(0);
                const float v0 = history[(size_t) c * 3], v1 = history[(size_t) c * 3 + 1],
                            v2 = history[(size_t) c * 3 + 2], x = qkv[c];
                float sum = v0 * conv_w[(size_t) c * 4] + v1 * conv_w[(size_t) c * 4 + 1] +
                            v2 * conv_w[(size_t) c * 4 + 2] + x * conv_w[(size_t) c * 4 + 3];
                history[(size_t) c * 3] = v1;
                history[(size_t) c * 3 + 1] = v2;
                history[(size_t) c * 3 + 2] = x;
                float y = sum / (1.0f + sycl::exp(-sum));
                if (blk < qk_heads) {
                    const float ss = group_sum(item, &part[0], y * y);
                    y *= 1.0f / sycl::sqrt(ss + eps);
                }
                h[c] = y;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fused_gdn_conv_l2 launch: %s\n", e.what());
        std::exit(1);
    }
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    const int n = n_embd, HV = h_v;
    const int rows = 2 * HV;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-group of 32 (one warp) per row: the lanes stride the reduction axis and the lane partials are
        // summed in lane order, exactly the coalescing the CUDA warp kernel is built around.
        q->submit([&](sycl::handler& hh) {
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) 32), hh);
            hh.parallel_for(sycl::nd_range<1>((size_t) rows * 32, (size_t) 32), [=](sycl::nd_item<1> item) {
                const int row = (int) item.get_group(0);
                const int lane = (int) item.get_local_id(0);
                const bool is_beta = row >= HV;
                const int r = is_beta ? row - HV : row;
                const uint16_t* wsrc = (is_beta ? w_beta : w_alpha) + (size_t) r * n;
                const uint32_t* w4 = reinterpret_cast<const uint32_t*>(wsrc);
                float acc = 0.0f;
                for (int j = lane; j < n / 8; j += 32) {
                    const uint32_t w0 = w4[j * 4 + 0], w1 = w4[j * 4 + 1], w2 = w4[j * 4 + 2], w3 = w4[j * 4 + 3];
                    const float* xp = x + (size_t) j * 8;
                    acc = sycl::fma(f32_from_bf16((uint16_t) w0), xp[0], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) (w0 >> 16)), xp[1], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) w1), xp[2], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) (w1 >> 16)), xp[3], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) w2), xp[4], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) (w2 >> 16)), xp[5], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) w3), xp[6], acc);
                    acc = sycl::fma(f32_from_bf16((uint16_t) (w3 >> 16)), xp[7], acc);
                }
                acc = group_sum(item, &part[0], acc);
                if (lane != 0) return;
                if (is_beta) {
                    beta[r] = 1.0f / (1.0f + sycl::exp(-acc));
                } else {
                    const float v = acc + dt[r];
                    const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                    gate[r] = sp * ssm_a[r];
                }
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fused_gdn_ab launch: %s\n", e.what());
        std::exit(1);
    }
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    const int HK = h_k, HV = h_v;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // One work-group per value head; its S*RG threads are (col, row group), so a row's S columns are one
        // contiguous load.  The old warp staging of k/q is dropped (the values are read from the caller's
        // buffers directly) because a SYCL work-group of S*RG makes a "broadcast row" unnecessary.
        qp->submit([&](sycl::handler& hh) {
            sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) (RG * S)), hh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>((size_t) (S * RG)), hh);
            hh.parallel_for(sycl::nd_range<1>((size_t) HV * (S * RG), (size_t) (S * RG)), [=](sycl::nd_item<1> item) {
                const int head = (int) item.get_group(0);
                const int tid = (int) item.get_local_id(0);
                const int rg = tid / S;             // 0..3
                const int col = tid % S;            // 0..127
                const int qh = head % HK;           // q/k head shared by h_v / h_k value heads
                float s[RPG];
                float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
                const size_t row_stride = (size_t) HV * S;
                for (int r = 0; r < RPG; ++r) s[r] = base[(size_t) r * row_stride];

                sycl::group_barrier(item.get_group());
                const float g = sycl::exp(gate[head]);
                float kv = 0.0f;
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], k[(size_t) qh * S + rg * RPG + r], kv);
                red[rg * S + col] = kv;
                sycl::group_barrier(item.get_group());
                const float kv_col = red[0 * S + col] + red[1 * S + col] + red[2 * S + col] + red[3 * S + col];
                const float delta = (v[(size_t) head * S + col] - g * kv_col) * beta[head];
                float o = 0.0f;
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], k[(size_t) qh * S + rg * RPG + r] * delta);
                    o = sycl::fma(s[r], q[(size_t) qh * S + rg * RPG + r], o);
                    base[(size_t) r * row_stride] = s[r];
                }
                sycl::group_barrier(item.get_group());   // every work-item has read red[] for kv_col
                red[rg * S + col] = o;
                sycl::group_barrier(item.get_group());
                float oc = 0.0f, sq_part = 0.0f;
                if (rg == 0) {
                    oc = (red[0 * S + col] + red[1 * S + col] + red[2 * S + col] + red[3 * S + col]) *
                         (1.0f / sycl::sqrt((float) S));
                    sq_part = oc * oc;
                }
                // RMS over the head's 128 outputs: the CUDA shuffle tree is replaced by a work-group tree over
                // the S*RG partials (rows 1..3 contribute zero), result broadcast to every work-item.
                const float ss = group_sum(item, &wsum[0], sq_part);
                if (rg == 0) {
                    const float scale = 1.0f / sycl::sqrt(ss / (float) S + eps);
                    const float zz = z[(size_t) head * S + col];
                    y[(size_t) head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
                }
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fused_gdn_step_norm launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
