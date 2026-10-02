// src/kernels/sycl/quantize_act.cpp - SYCLomatic port of src/kernels/cuda/quantize_act.cu's K-quant and scaled
// activation quantizers (P2.S2).
//
// quantize_act.cu carries five kernels; `quantize_q8_0` and `dequant_q8_0` already live in quantize_q8_0.cpp /
// dequant_q8_0.cpp, so this file is the remaining three, the K-quants that cover 2.89 GiB of dense weights
// (the attention projections and `ssm_out`, the numerically sensitive ones per quantize_act.hpp) and the
// scaled variant the hit path needs for hit/miss parity with the CPU reference.
//
// Same shape as the already-ported Q8_0 kernels: one work-item per block, a flat range<1> (see dequant_q8_0.cpp
// for why the Arc rejects an nd_range whose global size is not an exact multiple of the local size).  The
// fp16 scale is written with `f16_from_f32` (the exact bit converter from f16_bits.hpp, device-callable) rather
// than `__float2half`, which round 198 found producing wrong bits.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace strata::kernels {

namespace {

constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;   // { float d ; int8 qs[256] ; int16 bsums[16] }

/// ggml's `nearest_int` (ggml-quants.c), transcribed rather than replaced.  The magic number is 1.5*2^23:
/// adding it forces the mantissa's integer part into the low bits, and the mask/subtract recover it.  Written as
/// the `bit_cast<int,float>` that `memcpy(&i,&val,4)` did - a reinterpret the target understands in device code
/// - rather than `rintf`, which goes the other way on exact ties and that is the whole point of the constant.
inline int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    return (sycl::bit_cast<int, float>(val) & 0x007fffff) - 0x00400000;
}

}  // namespace

void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % 32 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of 32\n", (long long) n);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / 32;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> b) {
                const float* xb = x + (size_t) b * 32;
                uint8_t* out = blocks + (size_t) b * 34;

                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
                // VERBATIM from cpu/expert.cpp:144-145, including the amax > 0 guard, so the fp32 value written
                // here is bit-identical to the `s` the CPU path used.
                const float s = amax > 0.f ? amax / 127.f : 0.f;
                const float inv = s > 0.f ? 1.f / s : 0.f;
                scales[b] = s;

                const uint16_t d16bits = f16_from_f32(s);
                out[0] = (uint8_t) (d16bits & 0xFF);
                out[1] = (uint8_t) (d16bits >> 8);
                for (int i = 0; i < 32; ++i) {
                    // VERBATIM from cpu/expert.cpp:159-162: reciprocal multiply, then `t + copysign(0.5,t)`
                    // truncated toward zero - `lround`'s rule, round half away from zero.
                    const float t = xb[i] * inv;
                    const float r = t + (t >= 0.f ? 0.5f : -0.5f);
                    int v = (int) r;
                    v = v < -127 ? -127 : (v > 127 ? 127 : v);
                    out[2 + i] = (uint8_t) (int8_t) v;
                }
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_0_scaled launch: %s\n", e.what());
        std::exit(1);
    }
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> b) {
                const float* xb = x + (size_t) b * QK_K;
                uint8_t* out = blocks + (size_t) b * Q8K_BYTES;
                float* d = (float*) out;
                int8_t* qs = (int8_t*) (out + 4);
                int16_t* bsums = (int16_t*) (out + 4 + QK_K);

                // `max` is the SIGNED value at the largest-magnitude position; the strict `>` keeps the FIRST
                // maximum, which is what np.argmax does in the reference transcription too.
                float max = 0.0f, amax = 0.0f;
                for (int j = 0; j < QK_K; ++j) {
                    const float ax = std::fabs(xb[j]);
                    if (ax > amax) {
                        amax = ax;
                        max = xb[j];
                    }
                }
                if (amax == 0.0f) {
                    // ggml zeroes d and qs and `continue`s, leaving bsums untouched; a parity test sees the
                    // difference, so this zeroes bsums as well.
                    *d = 0.0f;
                    for (int j = 0; j < QK_K; ++j) qs[j] = 0;
                    for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
                    return;
                }
                const float iscale = -127.0f / max;          // -127, NOT -128; see quantize_act.hpp
                for (int j = 0; j < QK_K; ++j) {
                    // `iscale * xb[j]` as a standalone float rounds to F32 before nearest_int_dev - the
                    // __fmul_rn the source pins, which avoids the FMA contraction a bare `iscale*xb[j]+12582912`
                    // expression would do and which flips the result just off a .5 boundary.
                    const float prod = iscale * xb[j];
                    const int v = nearest_int_dev(prod);
                    qs[j] = (v > 127) ? (int8_t) 127 : (int8_t) v;   // MIN only - the source has no lower clamp
                }
                for (int j = 0; j < QK_K / 16; ++j) {
                    int sum = 0;
                    for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
                    bsums[j] = (int16_t) sum;
                }
                *d = 1.0f / iscale;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "quantize_q8_K launch: %s\n", e.what());
        std::exit(1);
    }
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "dequant_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> b) {
                const uint8_t* blk = blocks + (size_t) b * Q8K_BYTES;
                float d;
                std::memcpy(&d, blk, 4);
                const int8_t* qs = (const int8_t*) (blk + 4);
                float* out = x + (size_t) b * QK_K;
                for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dequant_q8_K launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
