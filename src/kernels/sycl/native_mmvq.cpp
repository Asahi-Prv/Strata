// src/kernels/sycl/native_mmvq.cpp - SYCL port of src/kernels/cuda/native_mmvq.cu.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h. See docs/native-mmvq.md for exact scope.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// PORT NOTES (what the CUDA source did and what this file does instead):
//
//   * WARP OPS.  The CUDA oracle partitions each weight row's K blocks over 4 warps x 32 lanes and
//     reduces with __shfl_xor_sync (warp_sum / warp_max); the q8_1 quantizer does the same.  icpx
//     2026.1 exposes none of the SYCL subgroup-shuffle surface, so this port keeps every dot product
//     expression and the quantizer's amax/sum definitions EXACTLY but folds the reduction serially:
//     one work-item per 32-element activation block for the quantizer, and one work-item per output
//     row (for every column) for each format kernel.  The set of per-subchunk dot terms is identical,
//     only the floating-point summation order differs from the warp tree (the same trade native_gr_norm,
//     native_bf16 and native_qsa already make).
//
//   * INTRINSICS.  __dp4a, __byte_perm and __vsubss4 are reimplemented below in pure integer arithmetic
//     (little-endian byte order, signed 8-bit operands, __byte_perm's high selector bit ignored exactly
//     as prmt does).  half / half2 are plain uint16_t / uint32_t fields decoded with the exact bit
//     converters in f16_bits.hpp (round 193 found __half2float / __float2half producing wrong bits).
//
//   * NO SCRATCH, SINGLE KERNEL.  Each format is one submission with no malloc_device scratch buffer:
//     icpx 2026.1 queues are OUT-OF-ORDER, so a two-submit reduction would race (the failure native_bf16
//     documents).  The convenience *_f32 entry points DO compose a quantize kernel and a mmvq kernel, so
//     the mmvq submission takes an explicit sycl::event dependency on the quantizer instead of relying on
//     stream order.
//
//   * MULTI-COLUMN.  The ncols == 1 layout is used for every ncols (the default native_mmvq_set_multi_exact
//     == true layout, so each column is bitwise equal to a single-column call).  The upstream ncols > 1
//     table (native_mmvq_set_multi_exact == false) is NOT ported; the flag is still stored and reported by
//     native_mmvq_multi_exact() but selects the same exact kernels.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int MAX_NCOLS = 8;

// Weight block layouts: identical field order, offsets and sizes as ggml-common.h / the CUDA header.
// `half` and `half2` are raw little-endian bit patterns here; the helpers below decode them.
struct Q5KBlock {
    uint32_t dm;              // half2: low half = d, high half = -dmin
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    uint32_t ds;              // half2: low half = d, high half = sum of the original inputs
    int8_t qs[32];
};
struct Q20Block {
    uint16_t d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};
struct IQ4XSBlock {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    uint32_t dm;              // half2: low half = d, high half = -dmin
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
struct Q40Block {
    uint16_t d;
    uint8_t qs[16];
};
struct Q50Block {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    uint16_t d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

// ---------------------------------------------------------------------------------------------------
// Portable replacements for the handful of CUDA integer/half intrinsics the pinned dot products use.
// ---------------------------------------------------------------------------------------------------

inline float h2f(uint16_t h) { return f32_from_f16(h); }
inline uint16_t f2h(float f) { return f16_from_f32(f); }
// make_half2(float, float): low half from the first argument, high half from the second.
inline uint32_t make_half2(float lo, float hi) {
    return (uint32_t) f2h(lo) | ((uint32_t) f2h(hi) << 16);
}
inline float low2float(uint32_t h) { return f32_from_f16((uint16_t) (h & 0xFFFFu)); }
inline float high2float(uint32_t h) { return f32_from_f16((uint16_t) (h >> 16)); }

// __byte_perm: the 8 source bytes are x bytes 0..3 then y bytes 0..3; each 4-bit selector picks one.
// The selector's high bit is ignored (prmt uses the low three bits), which is exactly what the IQ4
// codebook's two-stage table lookup relies on.
inline uint32_t byte_perm(uint32_t x, uint32_t y, uint32_t s) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (s >> (4 * i)) & 0xFu;
        const uint32_t idx = sel & 0x7u;
        const uint32_t byte = idx < 4 ? ((x >> (8 * idx)) & 0xFFu) : ((y >> (8 * (idx - 4))) & 0xFFu);
        r |= byte << (8 * i);
    }
    return r;
}

// __dp4a: four signed 8-bit multiply-accumulates, exact in 32-bit integer arithmetic.
inline int dp4a(int a, int b, int c) {
    int r = c;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int ai = (int) (int8_t) ((a >> (8 * i)) & 0xFF);
        const int bi = (int) (int8_t) ((b >> (8 * i)) & 0xFF);
        r += ai * bi;
    }
    return r;
}

// __vsubss4: packed four-way signed 8-bit saturating subtract.
inline int vsubss4(int a, int b) {
    int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int av = (int) (int8_t) ((a >> (8 * i)) & 0xFF);
        const int bv = (int) (int8_t) ((b >> (8 * i)) & 0xFF);
        int d = av - bv;
        if (d > 127) d = 127;
        if (d < -128) d = -128;
        r |= ((uint32_t) (d & 0xFF)) << (8 * i);
    }
    return r;
}

struct Int2 {
    int x, y;
};
inline Int2 make_int2(int x, int y) { return Int2{x, y}; }

// A little-endian pair of 16-bit loads from a possibly only 2-byte-aligned block (Q3_K is 110 bytes,
// Q6_K reads qh at odd 8-byte offsets), expressed as four byte reads so no pointer alignment is assumed.
inline int load_int_b2(const void* ptr, int i32) {
    const uint8_t* p = static_cast<const uint8_t*>(ptr) + 4 * i32;
    return (int) (p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t) p[3] << 24));
}

// ---------------------------------------------------------------------------------------------------
// The pinned dot products, transcribed expression-for-expression from vecdotq.cuh.
// ---------------------------------------------------------------------------------------------------

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
inline float q5_q8_dot_impl(const int* vl, const int* vh, const int* u, const uint8_t* sc, const uint8_t* m,
                            uint32_t dm5, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = dp4a(v0i, u[2 * i], dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i], dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    return low2float(dm5) * sumf_d - high2float(dm5) * sumf_m;
}

inline float q5_q8_dot(const Q5KBlock* bq5, const Q81Block* bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (j >= 2) ? 0xFFFFFFFFu : 0u;
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    uint8_t sc[4];
    sc[0] = (uint8_t) (aux[0] & 0xFF);
    sc[1] = (uint8_t) (aux[0] >> 8);
    sc[2] = (uint8_t) (aux[1] & 0xFF);
    sc[3] = (uint8_t) (aux[1] >> 8);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, sc + 2, bq5->dm, d8);
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.  The weight block is only
// 2-byte aligned, so qs is intentionally loaded as int16_t, unlike the 4-byte aligned activation codes.
inline float q2_q8_dot(const Q20Block* w, const Q81Block* x, int iqs) {
    const float d2 = h2f(w->d);
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = (int) byte_perm(0x020100ff, 0x020100ff, (uint32_t) (q >> 0));
        const int qo = (int) byte_perm(0x020100ff, 0x020100ff, (uint32_t) (q >> 2));
        const int qx = (int) byte_perm((uint32_t) qe, (uint32_t) qo, 0x5140);
        const int qy = (int) byte_perm((uint32_t) qe, (uint32_t) qo, 0x7362);
        sumi = dp4a(u, qx, sumi);
        sumi = dp4a(v, qy, sumi);
    }
    const float d8 = low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment; load_int_b2 above keeps the
// pinned helper's pair of 16-bit loads and little-endian combine without assuming alignment.
inline float q3_q8_dot_impl(int vl, int vh, const int* u, const uint8_t* scales, int scale_offset, float d3,
                            const float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        sumf += d8[i] * (dp4a(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

inline float q3_q8_dot(const Q3KBlock* w, const Q81Block* x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = h2f(w->d);
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// The pinned nonlinear IQ4 codebook, packed little-endian into the four 32-bit words the prmt-style
// two-stage byte lookup consumes.  Values are the verbatim ggml `kvalues_iq4nl`.
constexpr uint32_t kIq4nlTable[4] = {0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u};

inline Int2 iq4_table_lookup(int q4) {
    const uint32_t* table32 = kIq4nlTable;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210u | (((uint32_t) q4 & 0x88888888u) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = byte_perm(table32[0], table32[1], (uint32_t) q4 >> shift);
        const uint32_t high = byte_perm(table32[2], table32[3], (uint32_t) q4 >> shift);
        tmp[i] = byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2((int) byte_perm(tmp[0], tmp[1], 0x6420),
                     (int) byte_perm(tmp[0], tmp[1], 0x7531));
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock, computes integer dot
// products, applies signed scale in the integer domain, then multiplies the two half scales and the
// integer sum in the original order.
inline float iq4_xs_q8_dot(const IQ4XSBlock* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const Int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = dp4a(v.x, u0, sumi);
        sumi = dp4a(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = h2f(w->d) * low2float(x[iqs / 4].ds);
    return d * sumi;
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
inline float q4_q8_dot_impl(const int* v, const int* u, const uint8_t* sc, const uint8_t* m, uint32_t dm4,
                            const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = dp4a(v1i, u[2 * i + 1], dp4a(v0i, u[2 * i], 0));
        const int dot2 = dp4a(0x01010101, u[2 * i + 1], dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    return low2float(dm4) * sumf_d - high2float(dm4) * sumf_m;
}

inline float q4_q8_dot(const Q4KBlock* bq4, const Q81Block* bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (j >= 2) ? 0xFFFFFFFFu : 0u;
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    uint8_t sc[4];
    sc[0] = (uint8_t) (aux[0] & 0xFF);
    sc[1] = (uint8_t) (aux[0] >> 8);
    sc[2] = (uint8_t) (aux[1] & 0xFF);
    sc[3] = (uint8_t) (aux[1] >> 8);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, sc + 2, bq4->dm, d8);
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales, signed-byte subtraction, DP4A order,
// and the float accumulation sequence.
inline float q6_q8_dot_impl(int vl, int vh, const int* u, const int8_t* scales, float d, const float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (dp4a(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

inline float q6_q8_dot(const Q6KBlock* w, const Q81Block* x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, h2f(w->d), d8);
}

// The four 32-element formats use native two-byte loads and VDR=2.  The affine Q4_0/Q5_0 correction
// consumes the original-input sum stored in Q8_1, exactly as the pinned CUDA dot does; a signed-integer
// code substitution would differ.
inline float small_q8_dot(const Q40Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float ds_x = low2float(x->ds);
    const float ds_y = high2float(x->ds);
    const float d = h2f(w->d);
    return d * (sumi * ds_x - 4 * ds_y);
}

inline float small_q8_dot(const Q50Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = dp4a(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = dp4a(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float ds_x = low2float(x->ds);
    const float ds_y = high2float(x->ds);
    const float d = h2f(w->d);
    return d * (sumi * ds_x - 8 * ds_y);
}

inline float small_q8_dot(const Q80Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = dp4a(v, u, sumi);
    }
    const float d0 = h2f(w->d);
    const float d1 = low2float(x->ds);
    return d0 * d1 * float(sumi);
}

inline float small_q8_dot(const IQ4NLBlock* w, const Q81Block* x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = dp4a(v.x, q8[i], sumi);
        sumi = dp4a(v.y, q8[i + 4], sumi);
    }
    const float d = h2f(w->d) * low2float(x->ds);
    return d * sumi;
}

// ---------------------------------------------------------------------------------------------------
// Kernels.  Each format is a single scratch-free submission: one work-item per output row walks every
// (K block, sub-chunk) term in order and writes the row's output for every column.
// ---------------------------------------------------------------------------------------------------

// One work-item per 32-element Q8_1 block replaces the CUDA warp of 32 lanes: the block max and block
// sum are the same values the CUDA warp_max / warp_sum produced, and the quantized code is the pinned
// `amax == 0 ? 0 : roundf(x / d)`.
sycl::event quantize_q8_1_blocks(const float* x, Q81Block* y, int n_blocks, sycl::queue& qref,
                                 const sycl::event* dep) {
    return qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_blocks), [=](sycl::id<1> bid) {
            const int b = (int) bid;
            const float* xb = x + (size_t) b * Q8K;
            float amax = 0.0f;
            float sum = 0.0f;
            for (int k = 0; k < Q8K; ++k) {
                const float xi = xb[k];
                amax = sycl::fmax(amax, sycl::fabs(xi));
                sum += xi;
            }
            const float d = amax / 127.0f;
            Q81Block& out = y[b];
            for (int k = 0; k < Q8K; ++k) {
                const float xi = xb[k];
                out.qs[k] = (amax == 0.0f) ? (int8_t) 0 : (int8_t) sycl::round(xi / d);
            }
            out.ds = make_half2(d, sum);
        });
    });
}

// Q5_K: QK=256, QI=32, VDR=2 -> 16 sub-chunks per K block, 8 activation blocks per K block.
void q5_k_rows(const Q5KBlock* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
               sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / QK;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * (QK / Q8K);
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < QI / VDR; ++t) {
                    const int kqs = VDR * t;
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += q5_q8_dot(w + block, x + (size_t) j * x_stride + kby, kqs);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// Q2_0: QK=64, QI=2, VDR=1 -> 2 sub-chunks per K block, 2 activation blocks per K block.
void q2_0_rows(const Q20Block* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
               sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / 64;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * 2;
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < 2; ++t) {
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += q2_q8_dot(w + block, x + (size_t) j * x_stride + kby, t);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// Q3_K: QK=256, QI=16, VDR=1 -> 16 sub-chunks per K block, 8 activation blocks per K block.
void q3_k_rows(const Q3KBlock* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
               sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / 256;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * 8;
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < 16; ++t) {
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += q3_q8_dot(w + block, x + (size_t) j * x_stride + kby, t);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// IQ4_XS: QK=256, QI=32, VDR=4 -> 8 sub-chunks per K block (kqs = 4*t), 8 activation blocks per K block.
void iq4_xs_rows(const IQ4XSBlock* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                 sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / 256;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * 8;
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < 8; ++t) {
                    const int kqs = 4 * t;
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += iq4_xs_q8_dot(w + block, x + (size_t) j * x_stride + kby, kqs);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// Q4_K: QK=256, QI=32, VDR=2 -> 16 sub-chunks per K block, 8 activation blocks per K block.
void q4_k_rows(const Q4KBlock* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
               sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / QK;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * (QK / Q8K);
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < QI / VDR; ++t) {
                    const int kqs = VDR * t;
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += q4_q8_dot(w + block, x + (size_t) j * x_stride + kby, kqs);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// Q6_K: QK=256, QI=32, VDR=1 -> 32 sub-chunks per K block, 8 activation blocks per K block.
void q6_k_rows(const Q6KBlock* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
               sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / 256;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const int kby = kbx * 8;
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < 32; ++t) {
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += q6_q8_dot(w + block, x + (size_t) j * x_stride + kby, t);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0.  `kby` is the Q8_1 block index itself (KBY=1 in the
// generic multi-column traits), and `kqs = 2*t` picks the pinned VDR=2 sub-chunk.
template <typename Weight, int Qi>
void small_rows(const Weight* w, const Q81Block* x, float* y, int n_in, int n_out, int ncols,
                sycl::queue& qref, const sycl::event* dep) {
    const int blocks_per_row = n_in / 32;
    const int x_stride = n_in / Q8K;
    qref.submit([&](sycl::handler& h) {
        if (dep) h.depends_on(*dep);
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> rid) {
            const int row = (int) rid;
            float acc[MAX_NCOLS];
            for (int j = 0; j < ncols; ++j) acc[j] = 0.0f;
            for (int kbx = 0; kbx < blocks_per_row; ++kbx) {
                const size_t block = (size_t) row * blocks_per_row + kbx;
                for (int t = 0; t < Qi / 2; ++t) {
                    const int kqs = 2 * t;
                    for (int j = 0; j < ncols; ++j)
                        acc[j] += small_q8_dot(w + block, x + (size_t) j * x_stride + kbx, kqs);
                }
            }
            for (int j = 0; j < ncols; ++j) y[(size_t) j * n_out + row] = acc[j];
        });
    });
}

// ---------------------------------------------------------------------------------------------------
// Validation: every guard and throw the CUDA source had is kept verbatim.  The CUDA `launch_check()`
// (cudaGetLastError) has no SYCL analogue: `submit()` reports a launch failure by throwing, which the
// per-entry-point try/catch below turns into the same stderr message + exit(1).
// ---------------------------------------------------------------------------------------------------

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > MAX_NCOLS) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null device stream");
}

template <typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        small_rows<Weight, Qi>(static_cast<const Weight*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in,
                               n_out, ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native small MMVQ launch: %s\n", e.what());
        std::exit(1);
    }
}

template <typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
               void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // The quantizer and the mmvq are two submissions; take the quantizer's event as an explicit
        // dependency so the out-of-order queue cannot run the consumers before the producer.
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        small_rows<Weight, Qi>(static_cast<const Weight*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y,
                               n_in, n_out, ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native small MMVQ f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

bool g_multi_exact = true;

}  // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_blocks = (n_in * ncols) / Q8K;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        quantize_q8_1_blocks(x, static_cast<Q81Block*>(x_q8_1), n_blocks, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_quantize_q8_1 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q5_k_rows(static_cast<const Q5KBlock*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                  ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q5_k_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        q5_k_rows(static_cast<const Q5KBlock*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in, n_out,
                  ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q5_k_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q2_0_rows(static_cast<const Q20Block*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                  ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q2_0_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        q2_0_rows(static_cast<const Q20Block*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in, n_out,
                  ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q2_0_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q3_k_rows(static_cast<const Q3KBlock*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                  ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q3_k_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        q3_k_rows(static_cast<const Q3KBlock*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in, n_out,
                  ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q3_k_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        iq4_xs_rows(static_cast<const IQ4XSBlock*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                    ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_iq4_xs_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        iq4_xs_rows(static_cast<const IQ4XSBlock*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in,
                    n_out, ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_iq4_xs_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q4_k_rows(static_cast<const Q4KBlock*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                  ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q4_k_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        q4_k_rows(static_cast<const Q4KBlock*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in, n_out,
                  ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q4_k_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q6_k_rows(static_cast<const Q6KBlock*>(weights), static_cast<const Q81Block*>(x_q8_1), y, n_in, n_out,
                  ncols, *q, nullptr);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q6_k_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        sycl::event dep = quantize_q8_1_blocks(x, static_cast<Q81Block*>(scratch_q8_1), (n_in * ncols) / Q8K, *q,
                                               nullptr);
        q6_k_rows(static_cast<const Q6KBlock*>(weights), static_cast<const Q81Block*>(scratch_q8_1), y, n_in, n_out,
                  ncols, *q, &dep);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_q6_k_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                        void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

}  // namespace strata::kernels
