// src/kernels/sycl/iq_kernels.cpp - SYCLomatic port of src/kernels/cuda/iq_kernels.cu (see the header).
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at third_party/ggml/VERSION.txt; MIT license,
// third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h, included unchanged
// through the GGML_COMMON_DECL_SYCL / GGML_COMMON_IMPL_SYCL paths (the same ones llama.cpp's SYCL backend uses); a
// relative include is used because the kernel build reaches third_party/ggml only through CMake's -I list.
//
// PORTING NOTES (icpx 2026.1 / Arc):
//   * `__shfl_xor_sync` warp reductions become one 32-lane work-group per row with a `sycl::local_accessor` and a
//     `sycl::group_barrier`, exactly the pattern in bf16_gemv.cpp / s2_gemv_q8.cpp.  The lane-strided `row_dot`
//     loop and its coalescing are preserved; only the reduction order differs from the CUDA warp tree.
//   * The CUDA byte/SIMD intrinsics are written out as exact integer helpers: `__byte_perm` -> `byte_perm`,
//     `__vcmpne4`/`__vsub4` -> `vcmpne4`/`vsub4`, `__dp4a` -> `dp4a`, `__popc` -> `popc32`.  All are exact, so the
//     integer inner products are bit-identical to the CUDA kernel.
//   * A `block_q8_0`/`block_q8_1` is 34/36 bytes, so `x + 2` is only 2-byte aligned: 4-byte words are assembled
//     byte-wise (`load_i32`) rather than cast, the SYCL form of the source's `memcpy`.
//   * `block_q8_1`'s `ds` is a CUDA half2; the fp16 `d` is read from the first two raw bytes (`q8_1_low`) instead
//     of the GGML_COMMON_AGGR_U union, whose SYCL member name differs from CUDA's anonymous union.
//   * `__constant__` -> the file-scope `static const` ggml tables, which DPC++ places in the device-visible
//     constant pool (verified: a kernel may read them).
//   * `native_expert_grouped` is a chain of four dependent kernels on an OUT-OF-ORDER queue, so the stages are
//     separated by `q->wait()` the way native_ple_postops.cpp is.
#include "strata/kernels/iq_kernels.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "../../../third_party/ggml/ggml-common.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// ---------------------------------------------------------------- integer intrinsics (exact)
inline uint32_t load_u32(const void* p) {
    const uint8_t* b = (const uint8_t*) p;
    return (uint32_t) b[0] | ((uint32_t) b[1] << 8) | ((uint32_t) b[2] << 16) | ((uint32_t) b[3] << 24);
}
inline int load_i32(const void* p) { return (int) load_u32(p); }

inline unsigned popc32(uint32_t v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    v = (v + (v >> 4)) & 0x0f0f0f0fu;
    return (v * 0x01010101u) >> 24;
}

// PTX `prmt.b32` / `__byte_perm`: nibble i of `s` selects a byte of the 8-byte (a,b); the selector's low 3
// bits index the byte (0-3 from a, 4-7 from b) and bit 3 is IGNORED.  That last detail is what lets
// `get_int_from_table_16` stage the low and high halves of the table and then choose between them with bit 3;
// llama.cpp's own HIP fallback masks each index with `& 0x07070707` for exactly this reason.
inline uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = ((s >> (4 * i)) & 0x7u);
        const uint32_t byte = (sel < 4) ? ((a >> (8 * sel)) & 0xFFu) : ((b >> (8 * (sel - 4))) & 0xFFu);
        r |= byte << (8 * i);
    }
    return r;
}

// `__vcmpne4`: 0xFF in each byte that differs, 0x00 where equal.
inline int vcmpne4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t av = (uint32_t) (a >> (8 * i)) & 0xFFu;
        const uint32_t bv = (uint32_t) (b >> (8 * i)) & 0xFFu;
        r |= (int) ((av != bv ? 0xFFu : 0u) << (8 * i));
    }
    return r;
}

// `__vsub4`: per-byte wrapping subtract (no saturation).
inline int vsub4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t av = (uint32_t) (a >> (8 * i)) & 0xFFu;
        const uint32_t bv = (uint32_t) (b >> (8 * i)) & 0xFFu;
        r |= (int) (((av - bv) & 0xFFu) << (8 * i));
    }
    return r;
}

inline int dp4a(int a, int b, int c) {
    int r = c;
    r += (int) (int8_t) (a & 0xFF) * (int) (int8_t) (b & 0xFF);
    r += (int) (int8_t) ((a >> 8) & 0xFF) * (int) (int8_t) ((b >> 8) & 0xFF);
    r += (int) (int8_t) ((a >> 16) & 0xFF) * (int) (int8_t) ((b >> 16) & 0xFF);
    r += (int) (int8_t) ((a >> 24) & 0xFF) * (int) (int8_t) ((b >> 24) & 0xFF);
    return r;
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
inline int get_int_b2(const void* x, const int i32) { return load_i32((const uint8_t*) x + (size_t) i32 * 4); }
inline int get_int_b4(const void* x, const int i32) { return load_i32((const uint8_t*) x + (size_t) i32 * 4); }
inline uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = popc32(v) & 1;
    const uint32_t s = v ^ (p << 7);
    return s * 0x01010101u;
}

struct int2v { int x, y; };

inline float q8_1_low(const block_q8_1* b) {
    const uint8_t* p = (const uint8_t*) b;
    return f32_from_f16((uint16_t) (p[0] | (p[1] << 8)));
}

inline int2v get_int_from_table_16(const int q4, const int8_t* table) {
    uint32_t t[4];
    for (int i = 0; i < 4; ++i) t[i] = load_u32((const uint8_t*) table + 4 * i);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210u | (((uint32_t) q4 & 0x88888888u) >> 1));
    const uint32_t q4u = (uint32_t) q4;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = byte_perm(t[0], t[1], q4u >> shift);
        const uint32_t high = byte_perm(t[2], t[3], q4u >> shift);
        tmp[i] = byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return { (int) byte_perm(tmp[0], tmp[1], 0x6420), (int) byte_perm(tmp[0], tmp[1], 0x7531) };
}

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
inline float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                               const int iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = (float) bq2_0->d;
    const uint16_t* qs = (const uint16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const int q = (int) (int16_t) qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = (int) byte_perm(0x020100FFu, 0x020100FFu, (uint32_t) (q >> 0));
        const int qo = (int) byte_perm(0x020100FFu, 0x020100FFu, (uint32_t) (q >> 2));
        const int qx = (int) byte_perm((uint32_t) qe, (uint32_t) qo, 0x5140u);
        const int qy = (int) byte_perm((uint32_t) qe, (uint32_t) qo, 0x7362u);
        sumi = dp4a(u, qx, sumi);
        sumi = dp4a(v, qy, sumi);
    }
    const float d8 = q8_1_low(bq8_1_chunk);
    return d2 * d8 * sumi;
}

inline float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                  const int kbx, const int iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = (uint32_t) get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint64_t gp = iq2xxs_grid[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns((uint8_t) (aux32 >> (7 * k0 / 2)));
        const int signs0 = vcmpne4((int) (signs & 0x08040201u), 0);
        const int grid0 = vsub4((int) ((uint32_t) gp ^ (uint32_t) signs0), signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = dp4a(grid0, u0, sumi);
        const int signs1 = vcmpne4((int) (signs & 0x80402010u), 0);
        const int grid1 = vsub4((int) ((uint32_t) (gp >> 32) ^ (uint32_t) signs1), signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = dp4a(grid1, u1, sumi);
    }
    const int ls = (int) (aux32 >> 27) | 1;
    sumi = sumi * ls / 8;
    const float d = (float) bq2->d * q8_1_low(&bq8_1[iqs / 2]);
    return d * sumi;
}

inline float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                 const int iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2v q2_packed = { get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1) };
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint64_t gp = iq2xs_grid[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns((uint8_t) (q2[l0 / 2] >> 9));
        const int signs0 = vcmpne4((int) (signs & 0x08040201u), 0);
        const int grid_l = vsub4((int) ((uint32_t) gp ^ (uint32_t) signs0), signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = vcmpne4((int) (signs & 0x80402010u), 0);
        const int grid_h = vsub4((int) ((uint32_t) (gp >> 32) ^ (uint32_t) signs1), signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = dp4a(grid_l, u0, sumi0);
            sumi0 = dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = dp4a(grid_l, u0, sumi1);
            sumi1 = dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * q8_1_low(&bq8_1[iqs / 2]);
    return d * sumi;
}

inline float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                const int iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint64_t gp = iq2s_grid[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
        const int gp0 = (int) (uint32_t) gp, gp1 = (int) (uint32_t) (gp >> 32);
        const int signs0 = vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                       ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                       ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = vsub4(gp0 ^ signs0, signs0);
        const int grid_h = vsub4(gp1 ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = dp4a(grid_l, u0, sumi0);
            sumi0 = dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = dp4a(grid_l, u0, sumi1);
            sumi1 = dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = (float) bq2->d * q8_1_low(&bq8_1[iqs / 2]);
    return d * sumi;
}

inline float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                  const int kbx, const int iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2v q3_packed = { get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1) };
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = (uint32_t) get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2v grid_pos = { (int) iq3xxs_grid[q3[l0 + 0]], (int) iq3xxs_grid[q3[l0 + 1]] };
        const uint32_t signs = unpack_ksigns((uint8_t) (aux32 >> (7 * l0 / 2)));
        const int signs0 = vcmpne4((int) (signs & 0x08040201u), 0);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = vcmpne4((int) (signs & 0x80402010u), 0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = dp4a(grid_l, u0, sumi);
        sumi = dp4a(grid_h, u1, sumi);
    }
    const int ls = (int) (aux32 >> 28);
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = (float) bq3->d * q8_1_low(&bq8_1[iqs / 2]);
    return d * sumi;
}

inline float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                const int iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2v qs_packed = { get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1) };
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2v grid_pos = { (int) iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                 (int) iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)] };
        const int signs0 = vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) |
                                       ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) |
                                       ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = dp4a(grid_l, u0, sumi);
        sumi = dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = (float) bq3->d * q8_1_low(&bq8_1[iqs / 2]);
    return d * sumi;
}

inline float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                const int iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = (int) iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = dp4a(u0, 0x01010101, sumy);
        sumy = dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = (float) scale.f16 * q8_1_low(&bq8_1[iqs]);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

inline float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                 const int iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    int sumi = 0;
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2v v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = dp4a(v.x, get_int_b4(bq8_1->qs, iqs + l + 0), sumi);
        sumi = dp4a(v.y, get_int_b4(bq8_1->qs, iqs + l + 4), sumi);
    }
    const float d = (float) bq4->d * q8_1_low(bq8_1);
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block).
inline float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int kbx,
                                 const int iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2v v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = dp4a(v.x, u0, sumi);
        sumi = dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) |
                   (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = (float) bq4->d * q8_1_low(&bq8_1[iqs / 4]);
    return d * sumi;
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template <int TY> struct Fmt;
template <> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template <> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };

// One row against one q8_1 activation, the whole work-group: call k = (block, part) is lane-strided.  Returns the
// lane's partial; the caller reduces (the CUDA version reduced inside with __shfl_xor_sync).
template <int TY> inline float row_dot_partial(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + (size_t) kbx * (F::qk / 32), kbx, iqs);
    }
    return s;
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
void launch_quantize_q8_1(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n, sycl::queue& q) {
    const long long nb = (n + 31) / 32;
    if (nb <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> la(sycl::range<1>(32), h);
        sycl::local_accessor<float, 1> ls(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) nb * 32, 32), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const long long ib = (long long) item.get_group(0);
            const long long i = ib * 32 + lane;
            const float xi = i < n ? x[i] : 0.0f;
            la[lane] = sycl::fabs(xi);
            ls[lane] = xi;
            sycl::group_barrier(item.get_group());
            float amax = 0.0f, sum = 0.0f;
            for (int l = 0; l < 32; ++l) { amax = sycl::fmax(amax, la[l]); sum += ls[l]; }
            const float d = amax / 127.0f;
            const int8_t qv = amax == 0.0f ? 0 : (int8_t) sycl::round(xi / d);
            const int iqs = (int) (i % 32);
            y[ib].qs[iqs] = qv;
            if (iqs == 0) {
                const uint16_t db = f16_from_f32(d), sb = f16_from_f32(sum);
                uint8_t* base = (uint8_t*) &y[ib];
                base[0] = (uint8_t) db;
                base[1] = (uint8_t) (db >> 8);
                base[2] = (uint8_t) sb;
                base[3] = (uint8_t) (sb >> 8);
            }
        });
    });
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template <typename dst_t> inline dst_t cvt(float v);
template <> inline float cvt<float>(float v) { return v; }
template <> inline sycl::half cvt<sycl::half>(float v) { return sycl::half(v); }

template <typename dst_t>
inline void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid =
        (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template <typename dst_t>
inline void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template <typename dst_t>
inline void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 =
        (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 =
        (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template <typename dst_t>
inline void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template <typename dst_t>
inline void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32.
template <typename dst_t>
inline void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l)
            y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template <typename dst_t>
inline void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d *
                    ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template <typename dst_t>
inline void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

template <typename dst_t>
inline void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template <typename dst_t>
void launch_dequant_flat(int ty, const void* vx, int64_t nsuper, dst_t* y, sycl::queue& q) {
    if (nsuper <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) nsuper * 32), [=](sycl::id<1> gid) {
            const int i = (int) ((long long) gid / 32);
            const int tid = (int) ((long long) gid % 32);
            dq_dispatch<dst_t>(ty, vx, i, y + (size_t) i * QK_K, tid);
        });
    });
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
template <typename dst_t>
void launch_dequant_gu(int ty, const void* gate, const void* up, int64_t per_row, int64_t n_ff, dst_t* y,
                       sycl::queue& q) {
    const long long nb = n_ff * per_row;
    if (nb <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) nb * 2 * 32), [=](sycl::id<1> gid) {
            const long long block = (long long) gid / 32;
            const int tid = (int) ((long long) gid % 32);
            const int parity = (int) (block / nb);
            const long long i = block % nb;
            const long long r = i / per_row, c = i % per_row;
            dq_dispatch<dst_t>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, tid);
        });
    });
}
void launch_embed_rows(int ty, const uint8_t* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                       int64_t n_embd, float* y, sycl::queue& q) {
    const long long per_row = n_embd / 256;
    const long long nblk = per_row * n_tok;
    if (nblk <= 0) return;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) nblk * 32), [=](sycl::id<1> gid) {
            const long long block = (long long) gid / 32;
            const int tid = (int) ((long long) gid % 32);
            const int t = (int) (block / per_row);
            const long long b = block % per_row;
            const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
            dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, tid);
        });
    });
}

// ---------------------------------------------------------------- mmvq (one work-group per (row, column))
template <int TY>
void launch_mmvq(const uint8_t* w, size_t row_bytes, const block_q8_1* x, float* y, int n_in, int n_out, int ncols,
                 sycl::queue& q) {
    const int nb = n_in / Fmt<TY>::qk;
    const long long total = (long long) n_out * ncols;
    if (total <= 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) total * 32, 32), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const long long gid = (long long) item.get_group(0);
            const int row = (int) (gid % n_out);
            const int col = (int) (gid / n_out);
            const uint8_t* wr = w + (size_t) row * row_bytes;
            const float s = row_dot_partial<TY>(wr, x + (size_t) col * (n_in / 32), nb, lane);
            partials[lane] = s;
            sycl::group_barrier(item.get_group());
            if (lane == 0) {
                float ss = 0.0f;
                for (int l = 0; l < 32; ++l) ss += partials[l];
                y[(size_t) col * n_out + row] = ss;
            }
        });
    });
}

// ---------------------------------------------------------------- grouped native experts
template <int TG>
void launch_native_gu(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                      const int32_t* ent_tok, const block_q8_1* xq, int64_t n_ff, int64_t n_embd, size_t up_off,
                      size_t gu_row, float* gate, float* up, int64_t cap_groups, sycl::queue& q) {
    const long long rows = 2 * n_ff;
    const size_t total = (size_t) cap_groups * (size_t) rows;
    if (total == 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(total * 32, 32), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const size_t gid = item.get_group(0);
            const int g = (int) (gid / (size_t) rows);
            const int row = (int) (gid % (size_t) rows);
            if (g >= *n_groups) return;
            const bool is_up = row >= n_ff;
            const int r = is_up ? (int) (row - n_ff) : row;
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr = blob + (is_up ? up_off : 0) + (size_t) r * gu_row;
            const int nb = (int) (n_embd / Fmt<TG>::qk), xb = (int) (n_embd / 32);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; ++e) {
                const float s = row_dot_partial<TG>(wr, xq + (size_t) ent_tok[e] * (size_t) xb, nb, lane);
                partials[lane] = s;
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float ss = 0.0f;
                    for (int l = 0; l < 32; ++l) ss += partials[l];
                    (is_up ? up : gate)[(size_t) e * n_ff + r] = ss;
                }
                sycl::group_barrier(item.get_group());
            }
        });
    });
}

void launch_swiglu_entries(const float* gate, const float* up, float* h, long long n, sycl::queue& q) {
    if (n <= 0) return;
    q.submit([&](sycl::handler& hh) {
        hh.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> idx) {
            const long long i = (long long) idx;
            const float g = gate[i];
            h[i] = (g / (1.0f + sycl::exp(-g))) * up[i];
        });
    });
}

template <int TD>
void launch_native_down(const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                        const int32_t* ent_dst, const block_q8_1* hq, int64_t n_embd, int64_t n_ff, size_t down_off,
                        size_t d_row, float* out, int64_t cap_groups, sycl::queue& q) {
    const long long rows = n_embd;
    const size_t total = (size_t) cap_groups * (size_t) rows;
    if (total == 0) return;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(total * 32, 32), [=](sycl::nd_item<1> item) {
            const int lane = (int) item.get_local_id(0);
            const size_t gid = item.get_group(0);
            const int g = (int) (gid / (size_t) rows);
            const int r = (int) (gid % (size_t) rows);
            if (g >= *n_groups) return;
            const uint8_t* blob = (const uint8_t*) grp_ptr[g];
            const uint8_t* wr = blob + down_off + (size_t) r * d_row;
            const int nb = (int) (n_ff / Fmt<TD>::qk), hb = (int) (n_ff / 32);
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; ++e) {
                const float s = row_dot_partial<TD>(wr, hq + (size_t) e * (size_t) hb, nb, lane);
                partials[lane] = s;
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float ss = 0.0f;
                    for (int l = 0; l < 32; ++l) ss += partials[l];
                    out[(size_t) ent_dst[e] * n_embd + r] = ss;
                }
                sycl::group_barrier(item.get_group());
            }
        });
    });
}

bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11;
}

}  // namespace

bool iq_supported(int t) noexcept { return is_iq(t); }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_quantize_q8_1(x, (block_q8_1*) y, n, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_1_rows launch: %s\n", e.what());
        std::exit(1);
    }
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const uint8_t* W = (const uint8_t*) w;
    const block_q8_1* X = (const block_q8_1*) x_q8_1;
    try {
        switch (t) {
            case 16: launch_mmvq<16>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 17: launch_mmvq<17>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 18: launch_mmvq<18>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 20: launch_mmvq<20>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 21: launch_mmvq<21>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 22: launch_mmvq<22>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 23: launch_mmvq<23>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 29: launch_mmvq<29>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            case 42: launch_mmvq<42>(W, rb, X, y, n_in, n_out, ncols, *q); break;
            default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "iq_mmvq launch: %s\n", e.what());
        std::exit(1);
    }
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_dequant_flat<sycl::half>(t, src, n / 256, (sycl::half*) dst, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "iq_dequant_f16 launch: %s\n", e.what());
        std::exit(1);
    }
}

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_embed_rows(t, (const uint8_t*) table, row_bytes, tokens, n_tok, n_embd, out, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "iq_embed_rows launch: %s\n", e.what());
        std::exit(1);
    }
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_dequant_flat<float>(t, src, n / 256, dst, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "iq_dequant_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream) {
    const int64_t per_row = n_embd / 256;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch_dequant_gu<sycl::half>(t, gate, up, per_row, n_ff, (sycl::half*) dst, *q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "iq_dequant_gu_f16 launch: %s\n", e.what());
        std::exit(1);
    }
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) +
           (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const block_q8_1* X = (const block_q8_1*) x_q8_1;
    const int64_t n_embd = L.n_embd, n_ff = L.n_ff;
    const size_t up_off = L.up_off, down_off = L.down_off;
    const size_t gu_row = L.gu_row, d_row = L.d_row;
    try {
        switch (L.gu_type) {
            case 16: launch_native_gu<16>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 17: launch_native_gu<17>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 18: launch_native_gu<18>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 21: launch_native_gu<21>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 22: launch_native_gu<22>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 23: launch_native_gu<23>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 29: launch_native_gu<29>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            case 42: launch_native_gu<42>(grp_ptr, grp_start, n_groups, ent_tok, X, n_ff, n_embd, up_off, gu_row, gate, up, cap_groups, *q); break;
            default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
        }
        q->wait();
        const long long nh = (long long) cap_entries * n_ff;
        launch_swiglu_entries(gate, up, h, nh, *q);
        q->wait();
        launch_quantize_q8_1(h, hq, nh, *q);
        q->wait();
        switch (L.d_type) {
            case 20: launch_native_down<20>(grp_ptr, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, down_off, d_row, out, cap_groups, *q); break;
            case 23: launch_native_down<23>(grp_ptr, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, down_off, d_row, out, cap_groups, *q); break;
            case 42: launch_native_down<42>(grp_ptr, grp_start, n_groups, ent_dst, hq, n_embd, n_ff, down_off, d_row, out, cap_groups, *q); break;
            default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_expert_grouped launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
