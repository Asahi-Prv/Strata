// src/kernels/sycl/s_gemv.cpp - SYCLomatic port of src/kernels/cuda/s_gemv.cu's `s_gemv`, the one-thread-per-row
// S-family GEMV (P2.S2).
//
// ONE kernel for S2/S4/S8 because `docs/pack-format.md` gives them ONE decode: `value = cb[code] * scale + offset`
// with a per-TYPE codebook, code bias, group size and offset presence - all passed as arguments.  This is the
// reference the row-split kernels (`s_gemv_split`, `s_gemv_q8k_split`) are checked against, so it keeps the same
// per-element accumulation order and the same offset-before-multiply rule (writing `acc += code*scale*x + offset*x`
// rounds differently, which the Q4_K case could tell).
//
// The CUDA source decodes the non-linear IQ4NL codebook out of a `__constant__` table carried into per-block
// SHARED memory before a `__syncthreads`.  icpx 2026.1 exposes neither __constant__ nor a clean barrier for this
// pattern, so the 16-entry table stays a file-scope constant and `iq4nl_at` reads it straight - the read is
// uniform across a row (same 16 values) so it does not pay the divergent-constant cost this naive kernel is not
// trying to dodge; the split kernel's shared-memory rewrite is the follow-on.  The fp16 activation is read with
// `f32_from_f16` (the exact bit converter from f16_bits.hpp, device-callable) rather than `__half2float`, which
// is what round 193 found producing wrong bits.
#include "strata/kernels/s_gemv.hpp"

#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

// `kvalues_iq4nl`: the non-linear codebook, verbatim from ggml-common.h / ggml-quants.c.
constexpr signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                    1,   13,   25,  38,  53,  69,  89, 113};

// The codebook lookup, available to the device lambda.  `Affine` is `code + bias` (the bias is subtracted from
// the code in the integer domain before the multiply); `Iq4Nl` is the non-linear table, which no bias expresses.
template <int CODE_BITS>
inline float decode(int code, int bias, int codebook) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) kIq4Nl[code & 0x0F];
    return (float) (code + bias);
}

// One work-item per output row, decoding the weight on the fly and accumulating in FP32 - the same shape as the
// CUDA `<<<blocks, THREADS>>>` launch with the total `n_out` rows spread across `n_out` work-items.  Templated on
// the code WIDTH (2, 4 or 8) so the shift-and-mask stays a compile-time constant rather than a per-element divide.
template <int CODE_BITS>
void s_gemv_impl(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
                 int64_t n_in, int64_t n_out, const SForm& form) {
    const int cb = (int) form.codebook;
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
                constexpr int PER_BYTE = 8 / CODE_BITS;
                const long long n_groups = n_in / form.group_elems;
                const long long codes_per_row = n_in / PER_BYTE;
                const uint8_t* c = codes + o * codes_per_row;
                const float* s = scales + o * n_groups;
                const float* off = form.has_offset ? offset + o * n_groups : nullptr;

                float acc = 0.0f;
                for (long long g = 0; g < n_groups; ++g) {
                    const float d = s[g];
                    const float b = off ? off[g] : 0.0f;
                    const long long base = g * form.group_elems;
                    for (int j = 0; j < form.group_elems; ++j) {
                        const long long i = base + j;
                        const int code =
                            (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
                        // The offset belongs to the weight, applied to the decoded value BEFORE the activation
                        // multiply; `code*scale + offset` is not `code*scale*x + offset*x` in rounding.
                        const float w = decode<CODE_BITS>(code, form.code_bias, cb) * d + b;
                        acc += w * f32_from_f16(x[i]);
                    }
                }
                y[o] = acc;
            });
        });
        q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n", (long long) n_in,
                     form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    switch (form.code_bits) {
    case 2:
        s_gemv_impl<2>(x, codes, scales, offset, y, n_in, n_out, form);
        break;
    case 4:
        s_gemv_impl<4>(x, codes, scales, offset, y, n_in, n_out, form);
        break;
    case 8:
        s_gemv_impl<8>(x, codes, scales, offset, y, n_in, n_out, form);
        break;
    default:
        std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
        std::exit(1);
    }
}

}  // namespace strata::kernels
