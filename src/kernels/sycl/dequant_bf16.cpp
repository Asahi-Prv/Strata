// src/kernels/sycl/dequant_bf16.cpp - SYCLomatic port of src/kernels/cuda/dequant_bf16.cu (plan v0.3 P5).
//
// One thread per 32-element DECODED group (the whole `group32` is one work-item, flat range<1> - see
// dequant_q8_0.cpp for why the Arc rejects an nd_range whose global size is not a multiple of the local).  It
// transcribes ggml's `dequantize_row_*` for Q4_0(2), Q5_0(6), Q8_0(8), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14),
// IQ4_NL(20), IQ4_XS(23) and Q2_0(42): a row-major (n_rows, n_cols) block tensor becomes a row-major output
// matrix in the requested element type (BF16 / FP16 / FP32).
//
// The IQ-only formats (16,17,18,21,22,29) that dequant_f16/dequant_f32 delegate to `iq_dequant` in iq_kernels.cu
// use warp reductions unavailable under icpx 2026.1, so those two entry points report them unsupported; the bf16
// output entry point has no such branch and the inline group32 covers IQ4_NL and IQ4_XS directly.
//
// The fp16 block scale is read with `h2f` (reconstruct the little-endian fp16 bits then `f32_from_f16`, the exact
// bit converter - round 193 found `__half2float` wrong).  The bf16 output uses `bf16_from_f32` (bit-identical to
// the source's `f2bf`) and the fp16 output uses `f16_from_f32` (the validated round-to-nearest-even converter),
// not `__float2half_rn`, which produced wrong bits.
#include "strata/kernels/dequant_bf16.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

namespace strata::kernels {

namespace {

// `kvalues_iq4nl`: the non-linear codebook, verbatim from ggml-common.h / ggml-quants.c (see s_gemv.cpp).
constexpr signed char kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,   13,   25,  38,  53,  69,  89, 113};

// One fp16 block scale out of two little-endian bytes, then the exact fp16->fp32 bit converter.
inline float h2f(const uint8_t* p) {
    const uint16_t bits = (uint16_t) (p[0] | (p[1] << 8));
    return f32_from_f16(bits);
}

struct H16 { uint16_t v; };
// Single dispatch instead of overloads: template-dependent `T*` arguments do not resolve an overloaded `put` via ADL
// on icpx, so branch on T.  The three bodies are exactly the CUDA `put(uint16_t*/H16*/float*)` overloads.
template <typename T>
inline void put(T* o, int i, float v) {
    if constexpr (std::is_same_v<T, float>) {
        o[i] = v;                                  // FP32: exact
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        o[i] = bf16_from_f32(v);                   // BF16: round-to-nearest-even, bit-identical to the source f2bf
    } else {
        reinterpret_cast<H16*>(o)[i].v = f16_from_f32(v);  // FP16: round-to-nearest-even (the validated converter)
    }
}

inline void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// One 32-element group `g` (row-major over the whole slice); `out` points at that group's 32 outputs.  Verbatim
// transcription of the CUDA `group32`, the element-type dispatch falling out of the `put` overload chosen by T.
template <int TYPE, typename T>
inline void group32(const uint8_t* row_blocks, int gi_in_row, T* out) {
    if constexpr (TYPE == 42) {                                   // Q2_0: 64 per block of 18 B
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 2) * 18;
        const float d = h2f(b);
        const int e0 = (gi_in_row % 2) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int e = e0 + j;
            const int q = (b[2 + e / 4] >> ((e % 4) * 2)) & 3;
            put(out, j, (float) (q - 1) * d);
        }
    } else if constexpr (TYPE == 2) {                              // Q4_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, (float) ((b[2 + j] & 0x0F) - 8) * d);
            put(out, j + 16, (float) ((b[2 + j] >> 4) - 8) * d);
        }
    } else if constexpr (TYPE == 6) {                              // Q5_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 22;
        const float d = h2f(b);
        const uint32_t qh = (uint32_t) b[2] | ((uint32_t) b[3] << 8) | ((uint32_t) b[4] << 16) | ((uint32_t) b[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = ((qh >> j) << 4) & 0x10;
            const int xh1 = (qh >> (j + 12)) & 0x10;
            put(out, j, (float) (((b[6 + j] & 0x0F) | xh0) - 16) * d);
            put(out, j + 16, (float) (((b[6 + j] >> 4) | xh1) - 16) * d);
        }
    } else if constexpr (TYPE == 8) {                              // Q8_0
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 34;
        const float d = h2f(b);
        for (int j = 0; j < 32; ++j) put(out, j, (float) (int8_t) b[2 + j] * d);
    } else if constexpr (TYPE == 20) {                             // IQ4_NL
        const uint8_t* b = row_blocks + (size_t) gi_in_row * 18;
        const float d = h2f(b);
        for (int j = 0; j < 16; ++j) {
            put(out, j, d * (float) kv_iq4nl[b[2 + j] & 0xf]);
            put(out, j + 16, d * (float) kv_iq4nl[b[2 + j] >> 4]);
        }
    } else if constexpr (TYPE == 11) {                             // Q3_K: hmask[32] qs[64] scales[12] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 110;
        const int gi = gi_in_row % 8, n = gi / 4, jj = gi % 4;
        const uint8_t* hm = b;
        const uint8_t* q = b + 32 + n * 32;
        const uint8_t* sc = b + 96;
        const float d_all = h2f(b + 108);
        uint32_t aux[4];
        std::memcpy(aux, sc, 12);
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
        const int shift = 2 * jj;
        const uint8_t m = (uint8_t) (1u << (n * 4 + jj));
        for (int t = 0; t < 32; ++t) {
            const int is = n * 8 + jj * 2 + (t >= 16 ? 1 : 0);
            const float dl = d_all * (float) (scales[is] - 32);
            put(out, t, dl * (float) ((int) ((q[t] >> shift) & 3) - ((hm[t] & m) ? 0 : 4)));
        }
    } else if constexpr (TYPE == 12) {                             // Q4_K: d dmin scales[12] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 144;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* q = b + 16 + 32 * j64;
        for (int l = 0; l < 32; ++l) put(out, l, d1 * (float) (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
    } else if constexpr (TYPE == 13) {                             // Q5_K: d dmin scales[12] qh[32] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 176;
        const int gi = gi_in_row % 8, j64 = gi / 2, hi = gi % 2;
        const float d = h2f(b), dmin = h2f(b + 2);
        int sc, m;
        scale_min_k4(gi, b + 4, sc, m);
        const float d1 = d * (float) sc, m1 = dmin * (float) m;
        const uint8_t* qh = b + 16;
        const uint8_t* ql = b + 48 + 32 * j64;
        const uint8_t u = (uint8_t) (1u << (2 * j64 + hi));
        for (int l = 0; l < 32; ++l) {
            const int nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
            put(out, l, d1 * (float) (nib + ((qh[l] & u) ? 16 : 0)) - m1);
        }
    } else if constexpr (TYPE == 14) {                             // Q6_K: ql[128] qh[64] scales[16] d
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 210;
        const int gi = gi_in_row % 8, n = gi / 4, qu = gi % 4;
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
        const float d = h2f(b + 208);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qu == 0) q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
            else if (qu == 1) q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
            else if (qu == 2) q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
            else q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
            put(out, l, d * (float) sc[is + 2 * qu] * (float) (q - 32));
        }
    } else if constexpr (TYPE == 23) {                             // IQ4_XS: d scales_h scales_l[4] qs[128]
        const uint8_t* b = row_blocks + (size_t) (gi_in_row / 8) * 136;
        const int ib = gi_in_row % 8;
        const float d = h2f(b);
        const uint16_t scales_h = (uint16_t) (b[2] | (b[3] << 8));
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            put(out, j, dl * (float) kv_iq4nl[qs[j] & 0xf]);
            put(out, j + 16, dl * (float) kv_iq4nl[qs[j] >> 4]);
        }
    }
}

inline bool geometry(int type, int& block_elems, int& block_bytes) {
    switch (type) {
    case 2: block_elems = 32; block_bytes = 18; return true;
    case 6: block_elems = 32; block_bytes = 22; return true;
    case 8: block_elems = 32; block_bytes = 34; return true;
    case 20: block_elems = 32; block_bytes = 18; return true;
    case 11: block_elems = 256; block_bytes = 110; return true;
    case 12: block_elems = 256; block_bytes = 144; return true;
    case 13: block_elems = 256; block_bytes = 176; return true;
    case 14: block_elems = 256; block_bytes = 210; return true;
    case 23: block_elems = 256; block_bytes = 136; return true;
    case 42: block_elems = 64; block_bytes = 18; return true;
    default: return false;
    }
}

inline bool iq_only(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }

// One work-item per group, the flat range<1> the Arc needs.  The `type` switch instantiates the right group32 -
// mirroring the CUDA `STRATA_DQ` macro - so a caller picks its element type once here and dispatches the rest.
template <typename T>
void launch(int type, const uint8_t* blocks, int64_t row0, int64_t rows, int64_t cols, T* out, sycl::queue& q) {
    int be = 0, bb = 0;
    if (!geometry(type, be, bb) || cols % be != 0 || rows <= 0) {
        std::fprintf(stderr, "dequant: unsupported type %d or shape %lld x %lld\n", type, (long long) rows,
                     (long long) cols);
        std::exit(1);
    }
    const int64_t row_bytes = cols / be * bb, gpr = cols / 32, total = rows * gpr;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> g) {
            if (g >= (size_t) total) return;
            const int64_t r = g / gpr, gi = g % gpr;
            const uint8_t* rb = blocks + (size_t) (row0 + r) * row_bytes;
            const size_t off = (size_t) (r * gpr + gi) * 32;
            switch (type) {
            case 2: group32<2>(rb, (int) gi, out + off); break;
            case 6: group32<6>(rb, (int) gi, out + off); break;
            case 8: group32<8>(rb, (int) gi, out + off); break;
            case 11: group32<11>(rb, (int) gi, out + off); break;
            case 12: group32<12>(rb, (int) gi, out + off); break;
            case 13: group32<13>(rb, (int) gi, out + off); break;
            case 14: group32<14>(rb, (int) gi, out + off); break;
            case 20: group32<20>(rb, (int) gi, out + off); break;
            case 23: group32<23>(rb, (int) gi, out + off); break;
            case 42: group32<42>(rb, (int) gi, out + off); break;
            }
        });
    });
    q.wait();
}

}  // namespace

bool dequant_bf16_supported(int ggml_type) noexcept {
    int a, b;
    return geometry(ggml_type, a, b);
}

void dequant_bf16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                  void* stream) {
    if (rows <= 0 || cols <= 0 || !dequant_bf16_supported(ggml_type)) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch<uint16_t>(ggml_type, (const uint8_t*) blocks, row0, rows, cols, out, *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_bf16 launch: %s\n", e.what());
        std::exit(1);
    }
}

void dequant_f16(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, uint16_t* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        std::fprintf(stderr, "dequant_f16: IQ-only type %d not yet ported to SYCL (warp reductions)\n", ggml_type);
        std::exit(1);
    }
    if (rows <= 0 || cols <= 0 || !dequant_bf16_supported(ggml_type)) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        struct H16 { uint16_t v; };
        launch<H16>(ggml_type, (const uint8_t*) blocks, row0, rows, cols, reinterpret_cast<H16*>(out), *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_f16 launch: %s\n", e.what());
        std::exit(1);
    }
}

void dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out,
                 void* stream) {
    if (iq_only(ggml_type)) {
        std::fprintf(stderr, "dequant_f32: IQ-only type %d not yet ported to SYCL (warp reductions)\n", ggml_type);
        std::exit(1);
    }
    if (rows <= 0 || cols <= 0 || !dequant_bf16_supported(ggml_type)) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        launch<float>(ggml_type, (const uint8_t*) blocks, row0, rows, cols, out, *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_f32 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
