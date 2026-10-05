// src/kernels/sycl/fused_gr.cpp - SYCL port of src/kernels/cuda/fused_gr.cu (see include/strata/kernels/fused_gr.hpp).
//
// The fused read is TWO kernels for the single token (down + up) and THREE for the verify window (norm + down +
// up), where the CUDA source relies on the stream to order the kernels that pass `lo`/`rs`/`R_out`/`xn_scratch`
// between them.  icpx 2026.1 queues are OUT-OF-ORDER, so every such dependency is made explicit with a queue wait
// between the submits; no scratch buffer is introduced (the caller owns `lo`, `rs`, `xn_scratch`).
//
// The CUDA kernels reduce a row with `__shfl_xor_sync`; icpx 2026.1 has no subgroup-shuffle surface, so the row
// reduction uses a local-accessor work-group buffer: each lane writes its strided partial and lane 0 sums the 32
// of them (the same lane partials, only the summation order differs).  `dot8` keeps the two ordered bf16 fmadds
// of the source exactly - the two halves of each packed uint32 are multiplied with `f32_from_bf16`, never a pair
// sum.
//
// The multi-token `gr_down_multi_kernel` stages a `[T][TILE]` float activation tile in up to 80 KB of dynamic
// shared memory on CUDA.  Arc's per-work-group local limit is 64 KB, so this port reads the caller's `xn_scratch`
// directly instead; every `dot8` consumes the identical values in the identical order, so each token's outputs
// are bitwise `fused_gr_read(a[t])` just as the header promises.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int N = 2560;         // n_embd
constexpr int HC = 4;           // streams
constexpr int D = N * HC;       // 10240
constexpr int LR = 320;         // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS;          // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;                      // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;           // 80
constexpr int TILE = 2560;                       // xn floats per token staged at a time in CUDA
constexpr int UPM_COLS = 16;                     // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;         // 160
constexpr int MAXT = kFusedGrMaxT;

inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// 8 bf16 packed in a uint4 against 8 floats: the low half of each uint32 is the first bf16, the high half the
// second.  This keeps the source's two ordered fmadds per weight (NOT a pair-sum then a lone add).
inline float dot8(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, const float* x) {
    float acc = 0.0f;
    acc = sycl::fma(f32_from_bf16((uint16_t) v0), x[0], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) (v0 >> 16)), x[1], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) v1), x[2], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) (v1 >> 16)), x[3], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) v2), x[4], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) (v2 >> 16)), x[5], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) v3), x[6], acc);
    acc = sycl::fma(f32_from_bf16((uint16_t) (v3 >> 16)), x[7], acc);
    return acc;
}

/// `gr_down_kernel`: one block per output row (plus one injection block).  Step 1 writes the per-stream norm to
/// local memory and computes `rs`; step 2 has one warp per row stride the 10240-element reduction across its 32
/// lanes, exactly the coalescing the warp-per-row mapping exists for.
void launch_fused_down(const FusedGrArgs& a, sycl::queue& qref) {
    const float* R = a.R;
    const float* w_norm = a.w_norm;
    const uint16_t* w_down = a.w_down;
    const uint16_t* w_inject = a.w_inject;
    float* lo = a.lo;
    float* rs = a.rs;
    float* inject_out = a.inject_out;
    const bool apply = a.apply;
    const float* bo_prev = a.bo_prev;
    const float* inj_prev = a.inj_prev;
    const float eps = a.eps;

    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> xn_sh(sycl::range<1>((size_t) D), h);
        sycl::local_accessor<float, 1> s_rs(sycl::range<1>((size_t) HC), h);
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) THREADS), h);
        sycl::local_accessor<float, 1> wpart(sycl::range<1>((size_t) THREADS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (DOWN_BLOCKS + 1) * THREADS, (size_t) THREADS),
                       [=](sycl::nd_item<1> item) {
                           const sycl::group<1> g = item.get_group();
                           const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
                           const int blk = (int) item.get_group(0);
                           float gw[HC];
                           for (int c = 0; c < HC; ++c)
                               gw[c] = apply ? 2.0f * sigmoidf_(inj_prev[c] / (float) HC) : 0.0f;
                           // 1. R' * w_norm into local memory, and the per-stream sums of squares of R'.
                           float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                           for (int i = t * 4; i < D; i += THREADS * 4) {
                               const int c = i / N, d = i - c * N;
                               float r0 = R[i], r1 = R[i + 1], r2 = R[i + 2], r3 = R[i + 3];
                               if (apply) {
                                   r0 = sycl::fma(bo_prev[d], gw[c], r0);
                                   r1 = sycl::fma(bo_prev[d + 1], gw[c], r1);
                                   r2 = sycl::fma(bo_prev[d + 2], gw[c], r2);
                                   r3 = sycl::fma(bo_prev[d + 3], gw[c], r3);
                               }
                               const float g0 = w_norm[i], g1 = w_norm[i + 1], g2 = w_norm[i + 2],
                                           g3 = w_norm[i + 3];
                               const float sq = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
                               for (int cc = 0; cc < HC; ++cc)
                                   if (cc == c) ss[cc] += sq;
                               xn_sh[i] = r0 * g0;
                               xn_sh[i + 1] = r1 * g1;
                               xn_sh[i + 2] = r2 * g2;
                               xn_sh[i + 3] = r3 * g3;
                           }
                           float tot[HC];
                           for (int c = 0; c < HC; ++c) {
                               red[t] = ss[c];
                               sycl::group_barrier(g);
                               for (int off = THREADS / 2; off > 0; off >>= 1) {
                                   if (t < off) red[t] += red[t + off];
                                   sycl::group_barrier(g);
                               }
                               tot[c] = red[0];
                           }
                           if (t < HC) {
                               const float s = 1.0f / sycl::sqrt(tot[t] / (float) N + eps);
                               s_rs[t] = s;
                               if (blk == 0) rs[t] = s;
                           }
                           sycl::group_barrier(g);
                           for (int i = t; i < D; i += THREADS) xn_sh[i] *= s_rs[i / N];
                           sycl::group_barrier(g);
                           // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per lane.
                           const bool inject_block = blk == DOWN_BLOCKS;
                           const int row = inject_block ? warp : blk * WARPS + warp;
                           const bool active = !(inject_block && (w_inject == nullptr || warp >= HC));
                           float acc = 0.0f;
                           if (active) {
                               const uint16_t* wrow =
                                   (inject_block ? w_inject : w_down) + (size_t) row * D;
                               const uint32_t* w4 = reinterpret_cast<const uint32_t*>(wrow);
                               for (int j = lane; j < D / 8; j += 32)
                                   acc += dot8(w4[j * 4], w4[j * 4 + 1], w4[j * 4 + 2], w4[j * 4 + 3],
                                               &xn_sh[j * 8]);
                           }
                           wpart[warp * 32 + lane] = acc;
                           sycl::group_barrier(g);
                           if (active && lane == 0) {
                               float s = 0.0f;
                               for (int j = 0; j < 32; ++j) s += wpart[warp * 32 + j];
                               if (inject_block) {
                                   inject_out[row] = s;
                               } else {
                                   const float x = s / (float) HC;
                                   lo[row] = x / (1.0f + sycl::exp(-x));
                               }
                           }
                       });
    });
}

/// `gr_up_kernel`: one block per 32-column tile; each warp processes 16 (stream, column) rows, and the 320-long
/// `lo` dot is split across the lanes.  `R` is updated in place for this block's columns when `apply`.
void launch_fused_up(const FusedGrArgs& a, sycl::queue& qref) {
    const float* R = a.R;
    float* R_out = a.R_out;
    const float* w_norm = a.w_norm;
    const uint16_t* w_up = a.w_up;
    const float* lo = a.lo;
    const float* rs = a.rs;
    float* mixed = a.mixed;
    const bool apply = a.apply;
    const float* bo_prev = a.bo_prev;
    const float* inj_prev = a.inj_prev;

    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lo_sh(sycl::range<1>((size_t) LR), h);
        sycl::local_accessor<float, 1> g_sh(sycl::range<1>((size_t) (HC * UP_COLS)), h);
        sycl::local_accessor<float, 1> wpart(sycl::range<1>((size_t) THREADS), h);
        h.parallel_for(sycl::nd_range<1>((size_t) UP_BLOCKS * THREADS, (size_t) THREADS),
                       [=](sycl::nd_item<1> item) {
                           const sycl::group<1> g = item.get_group();
                           const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
                           const int d0 = (int) item.get_group(0) * UP_COLS;
                           for (int k = t; k < LR; k += THREADS) lo_sh[k] = lo[k];
                           sycl::group_barrier(g);
                           // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40 chunks of 8.
                           for (int r = warp; r < HC * UP_COLS; r += WARPS) {
                               const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
                               const uint32_t* w4 = reinterpret_cast<const uint32_t*>(w_up + (size_t) i * LR);
                               float acc = dot8(w4[lane * 4], w4[lane * 4 + 1], w4[lane * 4 + 2],
                                                w4[lane * 4 + 3], &lo_sh[lane * 8]);
                               if (lane < LR / 8 - 32)
                                   acc += dot8(w4[(32 + lane) * 4], w4[(32 + lane) * 4 + 1],
                                               w4[(32 + lane) * 4 + 2], w4[(32 + lane) * 4 + 3],
                                               &lo_sh[(32 + lane) * 8]);
                               wpart[warp * 32 + lane] = acc;
                               sycl::group_barrier(g);
                               if (lane == 0) {
                                   float s = 0.0f;
                                   for (int j = 0; j < 32; ++j) s += wpart[warp * 32 + j];
                                   float rv = R[i];
                                   if (apply) {
                                       rv = sycl::fma(bo_prev[d0 + dd],
                                                      2.0f * sigmoidf_(inj_prev[c] / (float) HC), rv);
                                       R_out[i] = rv;   // this block owns column d0+dd of every stream
                                   }
                                   const float x = rv * w_norm[i] * rs[c];
                                   g_sh[c * UP_COLS + dd] = x * sigmoidf_(s);
                               }
                               sycl::group_barrier(g);
                           }
                           sycl::group_barrier(g);
                           if (t < UP_COLS) {
                               float s = 0.0f;
                               for (int c = 0; c < HC; ++c) s += g_sh[c * UP_COLS + t];
                               mixed[d0 + t] = s / (float) HC;
                           }
                       });
    });
}

// ================================ plan v0.3 P6: T tokens, one weight read ================================
struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float* xn;
    int T;
};

/// Step 1 of `gr_down_kernel`, one block per token, same work-items and reduction: rs[t] and xn[t] to global.
void launch_norm_multi(const GrMulti& m, sycl::queue& qref) {
    const int n_tok = m.T;
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) THREADS), h);
        sycl::local_accessor<float, 1> s_rs(sycl::range<1>((size_t) HC), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tok * THREADS, (size_t) THREADS),
                       [=](sycl::nd_item<1> item) {
                           const sycl::group<1> g = item.get_group();
                           const int blk = (int) item.get_group(0);
                           const int t = (int) item.get_local_id(0);
                           const FusedGrArgs& a = m.a[blk];
                           float* xn = m.xn + (size_t) blk * D;
                           float gw[HC];
                           for (int c = 0; c < HC; ++c)
                               gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float) HC) : 0.0f;
                           float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                           for (int i = t * 4; i < D; i += THREADS * 4) {
                               const int c = i / N, d = i - c * N;
                               float r0 = a.R[i], r1 = a.R[i + 1], r2 = a.R[i + 2], r3 = a.R[i + 3];
                               if (a.apply) {
                                   r0 = sycl::fma(a.bo_prev[d], gw[c], r0);
                                   r1 = sycl::fma(a.bo_prev[d + 1], gw[c], r1);
                                   r2 = sycl::fma(a.bo_prev[d + 2], gw[c], r2);
                                   r3 = sycl::fma(a.bo_prev[d + 3], gw[c], r3);
                               }
                               const float g0 = a.w_norm[i], g1 = a.w_norm[i + 1], g2 = a.w_norm[i + 2],
                                           g3 = a.w_norm[i + 3];
                               const float sq = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
                               for (int cc = 0; cc < HC; ++cc)
                                   if (cc == c) ss[cc] += sq;
                               xn[i] = r0 * g0;
                               xn[i + 1] = r1 * g1;
                               xn[i + 2] = r2 * g2;
                               xn[i + 3] = r3 * g3;
                           }
                           float tot[HC];
                           for (int c = 0; c < HC; ++c) {
                               red[t] = ss[c];
                               sycl::group_barrier(g);
                               for (int off = THREADS / 2; off > 0; off >>= 1) {
                                   if (t < off) red[t] += red[t + off];
                                   sycl::group_barrier(g);
                               }
                               tot[c] = red[0];
                           }
                           if (t < HC) {
                               s_rs[t] = 1.0f / sycl::sqrt(tot[t] / (float) N + a.eps);
                               a.rs[t] = s_rs[t];
                           }
                           sycl::group_barrier(g);
                           for (int i = t; i < D; i += THREADS) xn[i] *= s_rs[i / N];
                       });
    });
}

/// Step 2 of `gr_down_kernel` for T tokens.  One warp per row, so each lane accumulates the same chunks in the
/// same order as the single-token kernel; the activation tile is read from `xn_scratch` rather than staged
/// because T*TILE floats exceed Arc's local limit (see the file header).
void launch_down_multi(const GrMulti& m, sycl::queue& qref) {
    const int n_tok = m.T;
    const uint16_t* w_down = m.a[0].w_down;
    const uint16_t* w_inject = m.a[0].w_inject;
    const float* xn = m.xn;
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> wall(sycl::range<1>((size_t) THREADS * MAXT), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (DOWN_BLOCKS + 1) * THREADS, (size_t) THREADS),
                       [=](sycl::nd_item<1> item) {
                           const sycl::group<1> g = item.get_group();
                           const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
                           const int blk = (int) item.get_group(0);
                           const bool inject_block = blk == DOWN_BLOCKS;
                           const int row = inject_block ? warp : blk * WARPS + warp;
                           const bool active = !(inject_block && (w_inject == nullptr || warp >= HC));
                           const uint16_t* wrow =
                               (inject_block ? w_inject : w_down) + (size_t) (active ? row : 0) * D;
                           const uint32_t* w4 = reinterpret_cast<const uint32_t*>(wrow);
                           float acc[MAXT];
                           for (int k = 0; k < MAXT; ++k) acc[k] = 0.0f;
                           for (int base = 0; base < D; base += TILE) {
                               uint32_t wv[TILE / 8 / 32 * 4];
                               if (active) {
                                   for (int q = 0; q < TILE / 8 / 32; ++q) {
                                       const int chunk = base / 8 + lane + 32 * q;
                                       for (int cc = 0; cc < 4; ++cc) wv[q * 4 + cc] = w4[chunk * 4 + cc];
                                   }
                               }
                               if (!active) continue;
                               for (int q = 0; q < TILE / 8 / 32; ++q) {
                                   const int j = lane + 32 * q;
                                   for (int k = 0; k < n_tok; ++k)
                                       acc[k] += dot8(wv[q * 4], wv[q * 4 + 1], wv[q * 4 + 2], wv[q * 4 + 3],
                                                      &xn[(size_t) k * D + base + j * 8]);
                               }
                           }
                           for (int k = 0; k < n_tok; ++k)
                               wall[(warp * 32 + lane) * MAXT + k] = active ? acc[k] : 0.0f;
                           sycl::group_barrier(g);
                           if (active && lane == 0) {
                               for (int k = 0; k < n_tok; ++k) {
                                   float s = 0.0f;
                                   for (int j = 0; j < 32; ++j) s += wall[(warp * 32 + j) * MAXT + k];
                                   if (inject_block) {
                                       m.a[k].inject_out[row] = s;
                                   } else {
                                       const float x = s / (float) HC;
                                       m.a[k].lo[row] = x / (1.0f + sycl::exp(-x));
                                   }
                               }
                           }
                       });
    });
}

/// `gr_up_kernel` for T tokens: each row of w_up is read once; the T dots are reduced per warp and lane 0 runs
/// each token's epilogue - the T epilogues in parallel instead of one after another.
void launch_up_multi(const GrMulti& m, sycl::queue& qref) {
    const int n_tok = m.T;
    const uint16_t* w_up = m.a[0].w_up;
    qref.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lo_sh(sycl::range<1>((size_t) (MAXT * LR)), h);
        sycl::local_accessor<float, 1> g_sh(sycl::range<1>((size_t) (MAXT * HC * UPM_COLS)), h);
        sycl::local_accessor<float, 1> wall(sycl::range<1>((size_t) THREADS * MAXT), h);
        h.parallel_for(sycl::nd_range<1>((size_t) UPM_BLOCKS * THREADS, (size_t) THREADS),
                       [=](sycl::nd_item<1> item) {
                           const sycl::group<1> g = item.get_group();
                           const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
                           const int d0 = (int) item.get_group(0) * UPM_COLS;
                           for (int i = t; i < n_tok * LR; i += THREADS) lo_sh[i] = m.a[i / LR].lo[i % LR];
                           sycl::group_barrier(g);
                           for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
                               const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
                               const uint32_t* w4 = reinterpret_cast<const uint32_t*>(w_up + (size_t) i * LR);
                               const uint32_t wa0 = w4[lane * 4], wa1 = w4[lane * 4 + 1],
                                              wa2 = w4[lane * 4 + 2], wa3 = w4[lane * 4 + 3];
                               uint32_t wb0 = 0, wb1 = 0, wb2 = 0, wb3 = 0;
                               if (lane < LR / 8 - 32) {
                                   const int b = (32 + lane) * 4;
                                   wb0 = w4[b];
                                   wb1 = w4[b + 1];
                                   wb2 = w4[b + 2];
                                   wb3 = w4[b + 3];
                               }
                               for (int k = 0; k < n_tok; ++k) {
                                   float acc = dot8(wa0, wa1, wa2, wa3, &lo_sh[k * LR + lane * 8]);
                                   if (lane < LR / 8 - 32)
                                       acc += dot8(wb0, wb1, wb2, wb3, &lo_sh[k * LR + (32 + lane) * 8]);
                                   wall[(warp * 32 + lane) * MAXT + k] = acc;
                               }
                               sycl::group_barrier(g);
                               if (lane == 0) {
                                   for (int k = 0; k < n_tok; ++k) {
                                       float s = 0.0f;
                                       for (int j = 0; j < 32; ++j) s += wall[(warp * 32 + j) * MAXT + k];
                                       const FusedGrArgs& ak = m.a[k];
                                       float rv = ak.R[i];
                                       if (ak.apply) {
                                           rv = sycl::fma(ak.bo_prev[d0 + dd],
                                                          2.0f * sigmoidf_(ak.inj_prev[c] / (float) HC), rv);
                                           ak.R_out[i] = rv;
                                       }
                                       const float x = rv * ak.w_norm[i] * ak.rs[c];
                                       g_sh[(k * HC + c) * UPM_COLS + dd] = x * sigmoidf_(s);
                                   }
                               }
                               sycl::group_barrier(g);
                           }
                           sycl::group_barrier(g);
                           for (int i = t; i < n_tok * UPM_COLS; i += THREADS) {
                               const int k = i / UPM_COLS, col = i - k * UPM_COLS;
                               float s = 0.0f;
                               for (int c = 0; c < HC; ++c) s += g_sh[(k * HC + c) * UPM_COLS + col];
                               m.a[k].mixed[d0 + col] = s / (float) HC;
                           }
                       });
    });
}

}  // namespace

bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    return n_embd == N && hc == HC && hc_lr == LR;
}

void fused_gr_read(const FusedGrArgs& a, void* stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out)) {
        std::fprintf(stderr, "fused_gr_read: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // The up kernel reads `lo`/`rs`/`R_out`, so the out-of-order queue must be flushed between the two.
        launch_fused_down(a, *q);
        q->wait();
        launch_fused_up(a, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fused_gr_read launch: %s\n", e.what());
        std::exit(1);
    }
}

void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream) {
    if (n_tok < 1 || n_tok > kFusedGrMaxT || xn_scratch == nullptr) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    GrMulti m;
    for (int t = 0; t < n_tok; ++t) {
        m.a[t] = a[t];
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed ||
            (x.w_inject && !x.inject_out) || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            x.w_down != a[0].w_down || x.w_up != a[0].w_up || x.w_inject != a[0].w_inject ||
            x.w_norm != a[0].w_norm) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    m.xn = xn_scratch;
    m.T = n_tok;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // Each stage consumes the previous stage's `xn`/`lo`/`rs`, so the out-of-order queue is made to wait.
        launch_norm_multi(m, *q);
        q->wait();
        launch_down_multi(m, *q);
        q->wait();
        launch_up_multi(m, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fused_gr_read_multi launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
