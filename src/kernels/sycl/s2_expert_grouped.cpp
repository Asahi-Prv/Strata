// src/kernels/sycl/s2_expert_grouped.cpp - SYCLomatic port of src/kernels/cuda/s2_expert_grouped.cu (R4's grouped
// GPU expert).
//
// THE ARITHMETIC IS `src/kernels/cpu/expert.cpp`'s, over the same bytes, one work-group per output row.  The CUDA
// source reduces each row with `__shfl_down_sync` warp shuffles; icpx 2026.1 exposes no subgroup-shuffle surface,
// so each row is computed by a 32-lane work-group whose lane partials go through a `sycl::local_accessor<float,1>`
// buffer and a `sycl::group_barrier`, summed in lane order by lane 0.  This is the same pattern `bf16_gemv.cpp`
// and `s2_gemv_q8.cpp` use.  The lane order preserved is `row_dot_s2_q8`'s (lane L takes chunks L, L+32, ...); the
// CUDA version's final warp tree differs, which is the same "same up to float rounding, not the bit" contract the
// CUDA header states for `s_gemv_parity`.  The integer inner loop is exact, so only the reduction order differs.
//
// `__dp4a` becomes the sign-correct byte-wise dot `dp4a` below (exact in int), and the unaligned 4-byte activation
// load uses `load_i32_unaligned` because a `block_q8_0` is 34 bytes so `xb + 2` is only 2-byte aligned - the same
// reason the CUDA source used `memcpy` instead of a cast.
//
// ORDERING.  icpx 2026.1 queues are OUT-OF-ORDER, and each stage here reads the previous stage's output, so the
// B-stage kernels are separated by `q->wait()` exactly as `native_ple_postops.cpp` does; the intermediate's own
// quantization still calls the ported `quantize_q8_0` / `quantize_q8_0_scaled`.
#include "strata/kernels/s2_expert_grouped.hpp"

#include "strata/kernels/quantize_act.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// THE BLOB'S OWN GEOMETRY, from `include/strata/kernels/cpu/expert.hpp`.  Restated as literals because that
// header is the CPU path's and this file must not silently follow it if the two ever disagree: the sizes below
// are what the CPU kernel's indexing computes, and `moe_hit_parity` compares the two end to end.
constexpr int H = 2560;
constexpr int FF = 640;
constexpr int QK = 64;                       // Q2_0's group: one fp16 scale per 64 weights
constexpr int ROW_GU = H / 4;                // 640 B of codes per gate/up row (2 bits per element)
constexpr int ROW_D = FF / 4;                // 160 B per down row
constexpr int SC_GU = H / QK;                // 40 fp16 scales per gate/up row
constexpr int SC_D = FF / QK;                // 10 per down row
constexpr size_t O_D_CODES = (size_t) 2 * FF * ROW_GU;
constexpr size_t O_GU_SCALES = O_D_CODES + (size_t) H * ROW_D;
constexpr size_t O_D_SCALES = O_GU_SCALES + (size_t) 2 * FF * SC_GU * 2;

constexpr int THREADS = 256;
constexpr int WARP = 32;

inline int load_i32_unaligned(const void* p) {
    const uint8_t* b = (const uint8_t*) p;
    return (int) ((uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24));
}

// `__dp4a` with a SIGNED interpretation of both operands, exact in int.
inline int dp4a(int a, int b, int c) {
    int r = c;
    r += (int) (int8_t) (a & 0xFF) * (int) (int8_t) (b & 0xFF);
    r += (int) (int8_t) ((a >> 8) & 0xFF) * (int) (int8_t) ((b >> 8) & 0xFF);
    r += (int) (int8_t) ((a >> 16) & 0xFF) * (int) (int8_t) ((b >> 16) & 0xFF);
    r += (int) (int8_t) ((a >> 24) & 0xFF) * (int) (int8_t) ((b >> 24) & 0xFF);
    return r;
}

inline float f16_at(const uint8_t* p) { return f32_from_f16((uint16_t) (p[0] | (p[1] << 8))); }

/// ONE S2 ROW AGAINST A Q8_0 ACTIVATION, PER LANE.  Every lane accumulates its own float partial over a strided
/// set of 32-element chunks; the caller reduces.  `chunk` indices are absolute so the caller can start the lane
/// at any offset.  `x_scales` is R4.2h and optional (see the CUDA source).
inline float row_dot_s2_q8(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                           const uint8_t* __restrict__ x_q8_0, int n_chunks, int lane,
                           const float* __restrict__ x_scales) {
    float acc = 0.0f;
    for (int c = lane; c < n_chunks; c += WARP) {
        const uint8_t* cb = codes + (size_t) c * 8;             // 8 code bytes = 32 elements
        const uint8_t* xb = x_q8_0 + (size_t) c * 34;           // one block_q8_0
        const float dx = x_scales ? x_scales[c] : f16_at(xb);
        const uint8_t* xq = xb + 2;

        // ---- THE CODES, EXPANDED TO ONE BYTE PER ELEMENT, THEN FOUR AT A TIME INTO `dp4a`.
        int s = 0;      // sum of code * x
        int hx = 0;     // sum of x        - the weight-independent term, as ones * x
        const int ones = 0x01010101;
        for (int j = 0; j < 8; ++j) {
            const unsigned cbyte = cb[j];
            const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                                  (((cbyte >> 6) & 3u) << 24));
            const int xw = load_i32_unaligned(xq + 4 * j);
            s = dp4a(cw, xw, s);
            hx = dp4a(ones, xw, hx);
        }
        // One weight scale per 64 elements, so per TWO 32-element chunks.
        const float dw = f16_at(scales + (size_t) (c >> 1) * 2);
        acc += dw * dx * (float) (s - hx);
    }
    return acc;
}

/// GATE AND UP, ONE WORK-GROUP PER ROW.  The rows are interleaved: row-slot `i` is gate row `i/2` when `i` is even
/// and up row `(i-1)/2` when odd, and the output is gate-major (`[0,n_hits*FF)` gate, then up).
void launch_gu(const uint8_t* blob_base, const int32_t* slot_index, long long blob_bytes,
               const uint8_t* x_q8_0, const float* x_scales, float* gate_up, int n_hits,
               const int32_t* d_count, const int32_t* dst_index, int tok_div, sycl::queue& q) {
    const long long rows_per_hit = 2LL * FF;
    const long long total = (long long) n_hits * rows_per_hit;
    if (total <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) WARP), h);
        h.parallel_for(sycl::nd_range<1>((size_t) total * WARP, WARP), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const long long slot = (long long) item.get_group(0);
            const int hh = (int) (slot / rows_per_hit);
            if (d_count != nullptr && hh >= *d_count) return;     // token graph: capacity layout, device count
            const int i = (int) (slot % rows_per_hit);

            const uint8_t* blob = blob_base + (size_t) slot_index[hh] * (size_t) blob_bytes;
            const uint8_t* xq = x_q8_0;
            const float* xs = x_scales;
            if (tok_div > 0) {   // plan v0.3 P6 verify window: each hit reads its own token's activation
                const int tok = dst_index[hh] / tok_div;
                xq += (size_t) tok * (size_t) (H / 32) * 34;
                if (xs != nullptr) xs += (size_t) tok * (size_t) (H / 32);
            }
            const float acc = row_dot_s2_q8(blob + (size_t) i * ROW_GU,
                                            blob + O_GU_SCALES + (size_t) i * SC_GU * 2, xq, H / 32, lane, xs);
            partials[lane] = acc;
            sycl::group_barrier(item.get_group());
            if (lane == 0) {
                float s = 0.0f;
                for (int l = 0; l < WARP; ++l) s += partials[l];
                const int r = i >> 1;
                const size_t base = (i & 1) ? ((size_t) n_hits * FF + (size_t) hh * FF) : ((size_t) hh * FF);
                gate_up[base + (size_t) r] = s;
            }
        });
    });
}

/// `silu(gate) * up`, in place, over a GATE-MAJOR buffer.
void launch_swiglu(float* gate_up, long long n_pairs, sycl::queue& q) {
    if (n_pairs <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_pairs), [=](sycl::id<1> idx) {
            const long long i = (long long) idx;
            const float g = gate_up[i];
            const float u = gate_up[n_pairs + i];
            gate_up[i] = (g / (1.0f + sycl::exp(-g))) * u;
        });
    });
}

/// DOWN, ONE WORK-GROUP PER ROW.
void launch_down(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index, long long blob_bytes,
                 const uint8_t* h_q8_0, const float* h_scales, float* out, int n_hits, const int32_t* d_count,
                 sycl::queue& q) {
    const long long total = (long long) n_hits * H;
    if (total <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) WARP), h);
        h.parallel_for(sycl::nd_range<1>((size_t) total * WARP, WARP), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const long long row = (long long) item.get_group(0);
            const int hh = (int) (row / H);
            if (d_count != nullptr && hh >= *d_count) return;
            const int r = (int) (row % H);

            const uint8_t* blob = blob_base + (size_t) slot_index[hh] * (size_t) blob_bytes;
            const uint8_t* xb = h_q8_0 + (size_t) hh * (size_t) (FF / 32) * 34;
            const float acc = row_dot_s2_q8(blob + O_D_CODES + (size_t) r * ROW_D,
                                            blob + O_D_SCALES + (size_t) r * SC_D * 2, xb, FF / 32, lane,
                                            h_scales ? h_scales + (size_t) hh * (size_t) (FF / 32) : nullptr);
            partials[lane] = acc;
            sycl::group_barrier(item.get_group());
            if (lane == 0) {
                float s = 0.0f;
                for (int l = 0; l < WARP; ++l) s += partials[l];
                out[(size_t) dst_index[hh] * H + r] = s;
            }
        });
    });
}

// ---------------------------------------------------------------- CPU-order path
// The CPU subtracts the weight bias after its eight FMA accumulators have been reduced.  Moving the subtraction
// into each integer dot, as the legacy kernel does, changes rounding even with equal scales.
void launch_activation_correction(const uint8_t* q8, const float* scales, float* hx, int chunks, sycl::queue& q) {
    if (chunks <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) chunks), [=](sycl::id<1> cid) {
            const int c = (int) cid;
            const uint8_t* q = q8 + (size_t) c * 34 + 2;
            int sum = 0;
            for (int j = 0; j < 32; ++j) sum += (int) (int8_t) q[j];
            hx[c] = sycl::fma(scales[c], (float) sum, 0.0f);   // __fmul_rn
        });
    });
}

inline int dot4(const uint8_t* codes, const uint8_t* q) {
    const unsigned c = *codes;
    const int cw = (int) ((c & 3u) | (((c >> 2) & 3u) << 8) | (((c >> 4) & 3u) << 16) | (((c >> 6) & 3u) << 24));
    const int xw = load_i32_unaligned(q);
    return dp4a(cw, xw, 0);
}

/// Recreates `row_dot_cpu_order`'s shuffle tree in lane order.  Called by lane 0 with every lane's partial and the
/// lane-0 correction already in `loc`; the tree is `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))` and the correction is
/// subtracted last, matching CUDA's `__shfl_down_sync(...,4),(...,1),(...,2)`.
template <typename Acc>
inline float reduce_cpu_order(const Acc& loc_acc, const Acc& loc_corr) {
    const float a0 = loc_acc[0], a1 = loc_acc[1], a2 = loc_acc[2], a3 = loc_acc[3];
    const float a4 = loc_acc[4], a5 = loc_acc[5], a6 = loc_acc[6], a7 = loc_acc[7];
    const float acc = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));
    return acc - loc_corr[0];
}

template <bool DOWN>
void launch_cpu_order_projection(const uint8_t* blob_base, const int32_t* slots, const int32_t* destinations,
                                 long long blob_bytes, const uint8_t* xq, const float* xs, const float* hx,
                                 float* out, int n_hits, sycl::queue& q) {
    constexpr int rows_per_hit = DOWN ? H : 2 * FF;
    const long long total = (long long) n_hits * rows_per_hit;
    if (total <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> loc_acc(sycl::range<1>(8), h);
        sycl::local_accessor<float, 1> loc_corr(sycl::range<1>(8), h);
        h.parallel_for(sycl::nd_range<1>((size_t) total * 8, 8), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const long long row = (long long) item.get_group(0);
            const int hh = (int) (row / rows_per_hit);
            const int r = (int) (row % rows_per_hit);
            const uint8_t* blob = blob_base + (size_t) slots[hh] * (size_t) blob_bytes;
            const int chunks_offset = DOWN ? hh * (FF / 32) : 0;
            const uint8_t* codes = DOWN ? blob + O_D_CODES + (size_t) r * ROW_D : blob + (size_t) r * ROW_GU;
            const uint8_t* scales = DOWN ? blob + O_D_SCALES + (size_t) r * SC_D * 2
                                         : blob + O_GU_SCALES + (size_t) r * SC_GU * 2;

            float acc = 0.0f;
            float corr = 0.0f;
            const int blocks = DOWN ? SC_D : SC_GU;
            for (int b = 0; b < blocks; ++b) {
                const float d = f16_at(scales + 2 * b);
                const int lo = dot4(codes + b * 16 + lane,
                                    xq + (size_t) (2 * b) * 34 + 2 + lane * 4);
                const int hi = dot4(codes + b * 16 + 8 + lane,
                                    xq + (size_t) (2 * b + 1) * 34 + 2 + lane * 4);
                const float dlo = sycl::fma(d, xs[2 * b], 0.0f);        // __fmul_rn(d, xs)
                acc = sycl::fma(dlo, (float) lo, acc);                  // __fmaf_rn
                const float dhi = sycl::fma(d, xs[2 * b + 1], 0.0f);
                acc = sycl::fma(dhi, (float) hi, acc);
                if (lane == 0) {
                    const float hxsum = sycl::fma(1.0f, hx[2 * b], hx[2 * b + 1]);   // __fadd_rn
                    corr = sycl::fma(1.0f, corr, sycl::fma(d, hxsum, 0.0f));         // __fadd_rn(corr, __fmul_rn)
                }
            }
            loc_acc[lane] = acc;
            loc_corr[lane] = corr;
            sycl::group_barrier(item.get_group());
            if (lane == 0) {
                const float value = reduce_cpu_order(loc_acc, loc_corr);
                if (DOWN) out[(size_t) destinations[hh] * H + r] = value;
                else out[((r & 1) ? (size_t) n_hits * FF : 0) + (size_t) hh * FF + (r >> 1)] = value;
            }
        });
    });
}

void launch_cpu_order_swiglu(float* gu, int pairs, sycl::queue& q) {
    if (pairs <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) pairs), [=](sycl::id<1> idx) {
            const int i = (int) idx;
            const float g = gu[i];
            // Accurate fp32 exponential; __expf's approximation would introduce an additional source of error.
            const float eg = sycl::exp(-g);
            gu[i] = (g / (1.0f + eg)) * gu[pairs + i];   // __fmul_rn(__fdiv_rn(g, __fadd_rn(1,eg)), up)
        });
    });
}

void launch_cpu_order_quantize(const float* x, uint8_t* blocks, float* scales, float* hx, int chunks,
                               sycl::queue& q) {
    if (chunks <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) chunks), [=](sycl::id<1> cid) {
            const int c = (int) cid;
            const float* xb = x + c * 32;
            uint8_t* out = blocks + (size_t) c * 34;
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
            const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
            const float inv = s > 0.0f ? 1.0f / s : 0.0f;
            scales[c] = s;
            const uint16_t bits = f16_from_f32(s);
            out[0] = (uint8_t) bits;
            out[1] = (uint8_t) (bits >> 8);
            int sum = 0;
            for (int j = 0; j < 32; ++j) {
                const float t = sycl::fma(xb[j], inv, 0.0f);                     // __fmul_rn
                int v = (int) sycl::fma(1.0f, t, t >= 0.0f ? 0.5f : -0.5f);      // __fadd_rn
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                out[2 + j] = (uint8_t) (int8_t) v;
                sum += v;
            }
            hx[c] = sycl::fma(s, (float) sum, 0.0f);                             // __fmul_rn
        });
    });
}

// Plan v0.3 P4 token graph: which of this layer's routed experts are resident, one work-group of 32.
void launch_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                       int32_t* count, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> flags(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            int s = -1;
            if (lane < k) {
                const int e = ids[lane];
                if (e >= 0 && e < n_expert) s = res_row[e];
            }
            flags[lane] = (uint8_t) (s >= 0 ? 1 : 0);
            sycl::group_barrier(item.get_group());
            unsigned hit = 0;
            for (int l = 0; l < 32; ++l) hit |= (unsigned) (flags[l] & 1u) << l;   // __ballot_sync
            if (s >= 0) {
                const int at = (int) sycl::popcount(hit & ((1u << lane) - 1u));    // __popc
                slot[at] = s;
                dst[at] = lane;
            }
            if (lane == 0) *count = (int32_t) sycl::popcount(hit);
        });
    });
}

// Plan v0.3 P6: the same for up to 128 routed entries (a verify window of T tokens x k): four warps.
void launch_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot,
                             int32_t* dst, int32_t* count, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 1> flags(sycl::range<1>(128), h);
        sycl::local_accessor<uint32_t, 1> wballot(sycl::range<1>(4), h);
        sycl::local_accessor<int, 1> wcount(sycl::range<1>(4), h);
        h.parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> item) {
            const int i = (int) item.get_local_id(0), lane = i & 31, warp = i >> 5;
            int s = -1;
            if (i < n) {
                const int e = ids[i];
                if (e >= 0 && e < n_expert) s = res_row[e];
            }
            flags[i] = (uint8_t) (s >= 0 ? 1 : 0);
            sycl::group_barrier(item.get_group());
            if (lane == 0) {
                unsigned b = 0;
                for (int l = 0; l < 32; ++l) b |= (unsigned) (flags[warp * 32 + l] & 1u) << l;
                wballot[warp] = b;
                wcount[warp] = (int) sycl::popcount(b);
            }
            sycl::group_barrier(item.get_group());
            const unsigned hit = wballot[warp];
            if (s >= 0) {
                int before = 0;
                for (int w = 0; w < warp; ++w) before += wcount[w];
                const int at = before + (int) sycl::popcount(hit & ((1u << lane) - 1u));
                slot[at] = s;
                dst[at] = i;
            }
            if (i == 0) *count = wcount[0] + wcount[1] + wcount[2] + wcount[3];
        });
    });
}

void launch_add_hits(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                     int64_t n_embd, sycl::queue& q) {
    const long long total = (long long) cap * n_embd;
    if (total <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> idx) {
            const long long i = (long long) idx;
            const int hh = (int) (i / n_embd);
            if (hh >= *count) return;
            const size_t row = (size_t) dst[hh] * (size_t) n_embd;
            const size_t col = (size_t) (i % n_embd);
            parts[row + col] += hit_out[row + col];
        });
    });
}

// Plan v0.3 P6: groups built on the device when every expert is resident at `base + id * blob` (the MTP layer):
// one work-group of 128, groups in first-appearance order, entries of a group in routing order.
void launch_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, long long blob,
                           unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                           int32_t* ent_tok, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> e_s(sycl::range<1>(128), h);
        sycl::local_accessor<int, 1> first_s(sycl::range<1>(128), h);
        sycl::local_accessor<int, 1> size_s(sycl::range<1>(128), h);
        sycl::local_accessor<int, 1> gidx_s(sycl::range<1>(128), h);
        sycl::local_accessor<int, 1> gstart_s(sycl::range<1>(129), h);
        h.parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> item) {
            const int i = (int) item.get_local_id(0);
            const int e = i < n ? ids[i] : -1;
            e_s[i] = e;
            sycl::group_barrier(item.get_group());
            int first = i, rank = 0, size = 0;
            if (i < n) {
                for (int j = 0; j < i; ++j)
                    if (e_s[j] == e) { if (first == i) first = j; ++rank; }
                if (first == i)
                    for (int j = i; j < n; ++j) size += e_s[j] == e;
            }
            first_s[i] = first;
            size_s[i] = (i < n && first == i) ? size : 0;
            sycl::group_barrier(item.get_group());
            if (i == 0) {
                int gi = 0, acc = 0;
                for (int j = 0; j < n; ++j)
                    if (first_s[j] == j) {
                        gidx_s[j] = gi;
                        gstart_s[gi] = acc;
                        grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                        grp_start[gi] = acc;
                        acc += size_s[j];
                        ++gi;
                    }
                grp_start[gi] = acc;
                counts[0] = gi;
                counts[1] = acc;
            }
            sycl::group_barrier(item.get_group());
            if (i < n) {
                const int at = gstart_s[gidx_s[first]] + rank;
                ent_dst[at] = i;
                ent_tok[at] = i / k_per_tok;
            }
        });
    });
}

// ---------------------------------------------------------------- grouped (one blob pointer per group)
constexpr int GU_CHUNKS = (H / 32 + 31) / 32;   // 3: chunks of a gate/up row per lane (80 chunks / 32 lanes)
constexpr int GMAX = 8;                          // entries per group (tokens routed to one expert in a window)
constexpr int GU_ROWS = 32;                      // gate/up rows per block: 4 per warp
constexpr int D_ROWS = 64;                       // down rows per block: 8 per warp

/// One activation chunk's contribution, `row_dot_s2_q8`'s inner body with the 32 int8 of the chunk already in
/// aligned words: same dp4a sequence, same float expression, so the result is bitwise the per-entry kernel's.
inline float chunk_dot(uint32_t cb0, uint32_t cb1, const int* xw, float dw, float dx) {
    const uint8_t* cbytes = (const uint8_t*) &cb0;   // cb0 then cb1 are the 8 code bytes
    const uint8_t bytes[8] = {cbytes[0], cbytes[1], cbytes[2], cbytes[3],
                              (uint8_t) (cb1 & 0xFF), (uint8_t) ((cb1 >> 8) & 0xFF),
                              (uint8_t) ((cb1 >> 16) & 0xFF), (uint8_t) ((cb1 >> 24) & 0xFF)};
    int s = 0, hx = 0;
    const int ones = 0x01010101;
    for (int j = 0; j < 8; ++j) {
        const unsigned cbyte = bytes[j];
        const int cw = (int) ((cbyte & 3u) | (((cbyte >> 2) & 3u) << 8) | (((cbyte >> 4) & 3u) << 16) |
                              (((cbyte >> 6) & 3u) << 24));
        s = dp4a(cw, xw[j], s);
        hx = dp4a(ones, xw[j], hx);
    }
    return dw * dx * (float) (s - hx);
}

void launch_gu_grouped(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                       const int32_t* ent_tok, const uint8_t* x_q8_0, const float* x_scales, float* gate_up,
                       int cap_entries, int cap_groups, sycl::queue& q) {
    constexpr int ROW_BYTES = H / 4;
    constexpr int N_TILES = 2 * FF / GU_ROWS;
    const size_t total_groups = (size_t) cap_groups * N_TILES;
    if (total_groups == 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> xs_q(sycl::range<1>((size_t) GMAX * (H / 4)), h);
        sycl::local_accessor<float, 1> xs_d(sycl::range<1>((size_t) GMAX * (H / 32)), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) THREADS), h);
        h.parallel_for(sycl::nd_range<1>(total_groups * THREADS, THREADS), [=](sycl::nd_item<1> item) {
            const size_t grp = item.get_group(0);
            const int g = (int) (grp / N_TILES);
            if (g >= *n_groups) return;
            const int tile = (int) (grp % N_TILES);
            const int e0 = grp_start[g];
            const int ne_raw = grp_start[g + 1] - e0;
            const int ne = ne_raw < GMAX ? ne_raw : GMAX;
            const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (H / 32); i += THREADS) {
                const int k = i / (H / 32), c = i - k * (H / 32);
                const uint8_t* xb = x_q8_0 + (size_t) ent_tok[e0 + k] * (size_t) (H / 32) * 34 + (size_t) c * 34;
                xs_d[k * (H / 32) + c] = x_scales ? x_scales[(size_t) ent_tok[e0 + k] * (H / 32) + c] : f16_at(xb);
                const uint8_t* qq = xb + 2;
                for (int w = 0; w < 8; ++w)
                    xs_q[(k * (H / 4)) + c * 8 + w] = load_i32_unaligned(qq + 4 * w);
            }
            sycl::group_barrier(item.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = tile * GU_ROWS;
            for (int rr = warp; rr < GU_ROWS; rr += 8) {
                const int i = row0 + rr;
                const uint8_t* codes = blob + (size_t) i * ROW_GU;
                const uint8_t* scales = blob + O_GU_SCALES + (size_t) i * SC_GU * 2;
                uint32_t cb0[GU_CHUNKS], cb1[GU_CHUNKS];
                float dw[GU_CHUNKS];
                for (int qq = 0; qq < GU_CHUNKS; ++qq) {
                    const int c = lane + 32 * qq;
                    cb0[qq] = 0;
                    cb1[qq] = 0;
                    dw[qq] = 0.0f;
                    if (c < H / 32) {
                        cb0[qq] = (uint32_t) load_i32_unaligned(codes + (size_t) c * 8);
                        cb1[qq] = (uint32_t) load_i32_unaligned(codes + (size_t) c * 8 + 4);
                        dw[qq] = f16_at(scales + (size_t) (c >> 1) * 2);
                    }
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    for (int qq = 0; qq < GU_CHUNKS; ++qq) {
                        const int c = lane + 32 * qq;
                        if (c >= H / 32) break;
                        acc += chunk_dot(cb0[qq], cb1[qq], &xs_q[k * (H / 4) + c * 8], dw[qq],
                                         xs_d[k * (H / 32) + c]);
                    }
                    partials[t] = acc;
                    sycl::group_barrier(item.get_group());
                    if (lane == 0) {
                        float sum = 0.0f;
                        for (int l = 0; l < 32; ++l) sum += partials[warp * 32 + l];
                        const int e = e0 + k, r = i >> 1;
                        const size_t base = (i & 1) ? ((size_t) cap_entries * FF + (size_t) e * FF)
                                                    : ((size_t) e * FF);
                        gate_up[base + (size_t) r] = sum;
                    }
                    sycl::group_barrier(item.get_group());
                }
            }
        });
    });
}

void launch_down_grouped(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                         const int32_t* ent_dst, const uint8_t* h_q8_0, const float* h_scales, float* out,
                         int cap_groups, sycl::queue& q) {
    constexpr int N_TILES = H / D_ROWS;
    const size_t total_groups = (size_t) cap_groups * N_TILES;
    if (total_groups == 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> hs_q(sycl::range<1>((size_t) GMAX * (FF / 4)), h);
        sycl::local_accessor<float, 1> hs_d(sycl::range<1>((size_t) GMAX * (FF / 32)), h);
        sycl::local_accessor<float, 1> partials(sycl::range<1>((size_t) THREADS), h);
        h.parallel_for(sycl::nd_range<1>(total_groups * THREADS, THREADS), [=](sycl::nd_item<1> item) {
            const size_t grp = item.get_group(0);
            const int g = (int) (grp / N_TILES);
            if (g >= *n_groups) return;
            const int tile = (int) (grp % N_TILES);
            const int e0 = grp_start[g];
            const int ne_raw = grp_start[g + 1] - e0;
            const int ne = ne_raw < GMAX ? ne_raw : GMAX;
            const int t = (int) item.get_local_id(0), lane = t & 31, warp = t >> 5;
            for (int i = t; i < ne * (FF / 32); i += THREADS) {
                const int k = i / (FF / 32), c = i - k * (FF / 32);
                const uint8_t* xb = h_q8_0 + (size_t) (e0 + k) * (size_t) (FF / 32) * 34 + (size_t) c * 34;
                hs_d[k * (FF / 32) + c] = h_scales ? h_scales[(size_t) (e0 + k) * (FF / 32) + c] : f16_at(xb);
                const uint8_t* qq = xb + 2;
                for (int w = 0; w < 8; ++w)
                    hs_q[(k * (FF / 4)) + c * 8 + w] = load_i32_unaligned(qq + 4 * w);
            }
            sycl::group_barrier(item.get_group());
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const int row0 = tile * D_ROWS;
            for (int rr = warp; rr < D_ROWS; rr += 8) {
                const int r = row0 + rr;
                const uint8_t* codes = blob + O_D_CODES + (size_t) r * ROW_D;
                const uint8_t* scales = blob + O_D_SCALES + (size_t) r * SC_D * 2;
                const int c = lane;
                uint32_t cb0 = 0, cb1 = 0;
                float dw = 0.0f;
                if (c < FF / 32) {
                    cb0 = (uint32_t) load_i32_unaligned(codes + (size_t) c * 8);
                    cb1 = (uint32_t) load_i32_unaligned(codes + (size_t) c * 8 + 4);
                    dw = f16_at(scales + (size_t) (c >> 1) * 2);
                }
                for (int k = 0; k < ne; ++k) {
                    float acc = 0.0f;
                    if (c < FF / 32) acc += chunk_dot(cb0, cb1, &hs_q[k * (FF / 4) + c * 8], dw, hs_d[k * (FF / 32) + c]);
                    partials[t] = acc;
                    sycl::group_barrier(item.get_group());
                    if (lane == 0) {
                        float sum = 0.0f;
                        for (int l = 0; l < 32; ++l) sum += partials[warp * 32 + l];
                        out[(size_t) ent_dst[e0 + k] * H + r] = sum;
                    }
                    sycl::group_barrier(item.get_group());
                }
            }
        });
    });
}

void launch_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                       const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                       const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, sycl::queue& q,
                       void* stream) {
    const uint64_t gu_bytes = ((uint64_t) cap_entries * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap_entries * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    launch_gu_grouped(grp_ptr, grp_start, n_groups, ent_tok, x_q8_0, x_scales, gate_up, (int) cap_entries,
                      (int) cap_groups, q);
    q.wait();
    launch_swiglu(gate_up, cap_entries * (long long) FF, q);
    q.wait();
    if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap_entries * (int64_t) FF, stream);
    else quantize_q8_0(gate_up, h_q8_0, cap_entries * (int64_t) FF, stream);
    q.wait();
    launch_down_grouped(grp_ptr, grp_start, n_groups, ent_dst, h_q8_0, x_scales != nullptr ? h_scales : nullptr, out,
                        (int) cap_groups, q);
}

}  // namespace

uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    // R4.2h: the fp32 scales for the INTERMEDIATE's own quantization, one per 32-element chunk per hit.
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) +
           ((xh + 15) & ~15ull);
}

void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());

    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);

    try {
        // 1. gate + up, one launch for every row of every hit.
        launch_gu(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) n_hits, nullptr, nullptr, 0, *q);
        q->wait();
        // 2. silu(gate) * up.
        launch_swiglu(gate_up, n_hits * (long long) FF, *q);
        q->wait();
        // 3. the intermediate's own contract, which is `ggml_mul_mat`'s rule and NOT a choice: the down weight is
        //    Q2_0, whose `vec_dot_type` is Q8_0.  R4.2h: the CPU quantizes this intermediate with fp32 scales too,
        //    so when the caller supplies `x_scales` the intermediate gets the CPU's contract as well.
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, n_hits * (int64_t) FF, stream);
        q->wait();
        // 4. down.
        launch_down(blob_base, slot_index, dst_index, blob_bytes, h_q8_0, x_scales != nullptr ? h_scales : nullptr,
                    out, (int) n_hits, nullptr, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_grouped_s2 launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    if (k < 1 || k > 32) { std::fprintf(stderr, "moe_hit_select: k must be 1..32\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_hit_select(ids, res_row, k, n_expert, slot, dst, count, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_select launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    try {
        launch_gu(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count, nullptr, 0, *q);
        q->wait();
        launch_swiglu(gate_up, cap * (long long) FF, *q);
        q->wait();
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
        q->wait();
        launch_down(blob_base, slot_index, dst_index, blob_bytes, h_q8_0, x_scales != nullptr ? h_scales : nullptr,
                    out, (int) cap, d_count, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_grouped_s2_dev launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_select_multi(const int32_t* ids, const int32_t* res_row, int n, int n_expert, int32_t* slot, int32_t* dst,
                          int32_t* count, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_hit_select_multi: n must be 1..128\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_hit_select_multi(ids, res_row, n, n_expert, slot, dst, count, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_select_multi launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_grouped_s2_multi(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                              const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                              const float* x_scales, int k_per_token, void* scratch, float* out, void* stream) {
    if (cap <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const uint64_t gu_bytes = ((uint64_t) cap * (uint64_t) (2 * FF) * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) cap * (uint64_t) (FF / 32) * 34 + 15) & ~15ull;
    float* gate_up = (float*) scratch;
    uint8_t* h_q8_0 = (uint8_t*) scratch + gu_bytes;
    float* h_scales = (float*) ((uint8_t*) scratch + gu_bytes + q8_bytes);
    try {
        launch_gu(blob_base, slot_index, blob_bytes, x_q8_0, x_scales, gate_up, (int) cap, d_count, dst_index,
                  k_per_token, *q);
        q->wait();
        launch_swiglu(gate_up, cap * (long long) FF, *q);
        q->wait();
        if (x_scales != nullptr) quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, cap * (int64_t) FF, stream);
        else quantize_q8_0(gate_up, h_q8_0, cap * (int64_t) FF, stream);
        q->wait();
        launch_down(blob_base, slot_index, dst_index, blob_bytes, h_q8_0, x_scales != nullptr ? h_scales : nullptr,
                    out, (int) cap, d_count, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_grouped_s2_multi launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_group_resident(const int32_t* ids, int n, int k_per_tok, const uint8_t* base, int64_t blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst,
                        int32_t* ent_tok, void* stream) {
    if (n < 1 || n > 128) { std::fprintf(stderr, "moe_group_resident: n must be 1..128\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_group_resident(ids, n, k_per_tok, base, (long long) blob, grp_ptr, grp_start, counts, ent_dst, ent_tok,
                              *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_group_resident launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_grouped_s2(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                    const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                    const uint8_t* x_q8_0, const float* x_scales, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_grouped_s2(grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, cap_entries, x_q8_0, x_scales,
                          scratch, out, *q, stream);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_grouped_s2 launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_add_hits(parts, hit_out, dst, count, cap, n_embd, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_add launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_hit_grouped_s2_cpu_order(const uint8_t* blob_base, const int32_t* slot_index,
                                  const int32_t* dst_index, int64_t n_hits, int64_t blob_bytes,
                                  const uint8_t* x_q8_0, void* scratch, float* out, void* stream,
                                  const float* x_scales, float* gate_up_trace) {
    if (n_hits <= 0) return;
    if (x_scales == nullptr) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order requires fp32 activation scales\n");
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const uint64_t gu_bytes = ((uint64_t) n_hits * 2 * FF * 4 + 15) & ~15ull;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (FF / 32) * 34 + 15) & ~15ull;
    const uint64_t scale_bytes = ((uint64_t) n_hits * (FF / 32) * 4 + 15) & ~15ull;
    float* gu = (float*) scratch;
    uint8_t* hq = (uint8_t*) scratch + gu_bytes;
    float* hs = (float*) (hq + q8_bytes);
    float* hh = (float*) ((uint8_t*) hs + scale_bytes);
    float* xh = (float*) ((uint8_t*) hh + scale_bytes);
    try {
        launch_activation_correction(x_q8_0, x_scales, xh, H / 32, *q);
        q->wait();
        launch_cpu_order_projection<false>(blob_base, slot_index, dst_index, blob_bytes, x_q8_0, x_scales, xh, gu,
                                           (int) n_hits, *q);
        q->wait();
        if (gate_up_trace != nullptr) {
            q->memcpy(gate_up_trace, gu, (size_t) n_hits * 2 * FF * sizeof(float));
            q->wait();
        }
        launch_cpu_order_swiglu(gu, (int) (n_hits * FF), *q);
        q->wait();
        launch_cpu_order_quantize(gu, hq, hs, hh, (int) (n_hits * (FF / 32)), *q);
        q->wait();
        launch_cpu_order_projection<true>(blob_base, slot_index, dst_index, blob_bytes, hq, hs, hh, out, (int) n_hits,
                                          *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_hit_grouped_s2_cpu_order launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
