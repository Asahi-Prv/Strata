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

namespace {

// ---- Q8_K ACTIVATIONS ----------------------------------------------------------------------------------
//
// 292 bytes per 256 elements: `{ f32 d ; int8_t qs[256] ; int16_t bsums[16] }`.  `bsums` is not read - it
// exists for ggml's AVX2 dot product - but the STRIDE includes it, so a buffer from `quantize_q8_K` passes
// straight in.
//
// The block's `d` sits at a 4-byte-aligned offset from the buffer base as long as the base is aligned, which a
// device allocation guarantees, so the `float` load below is aligned.
constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;

inline float q8k_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
    const float d = *reinterpret_cast<const float*>(blk);
    const int8_t q = (reinterpret_cast<const int8_t*>(blk + 4))[i % Q8K_BLOCK_ELEMS];
    return d * (float) q;
}

// **THE Q8_0 ACTIVATION**, which is what the LEGACY block formats want: `block_q8_0` is `{fp16 d; int8 qs[32]}`,
// 34 bytes per 32 elements, produced by `quantize_q8_0`.
//
// THE ENGINE HAD NO GEMV FOR THIS, and it was the last activation-contract gap.  `ffn_down_shexp` is
// IQ4_NL/Q4_0/Q5_0/Q8_0 in EVERY layer, whose `vec_dot_type` is Q8_0 - and its `n_in` is 640, which is not a
// multiple of 256, so Q8_K is STRUCTURALLY IMPOSSIBLE for it rather than merely absent.
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;

inline float q8_0_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    // a `uint16_t` needs 2-byte alignment and the stride is 34, so every block's `d` is aligned
    const float d = f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
    const int8_t q = (reinterpret_cast<const int8_t*>(blk + 2))[i % Q8_0_BLOCK_ELEMS];
    return d * (float) q;
}

// One work-group (`threads_per_row` work-items) per output row, each work-item striding the reduction axis; the
// per-item partials are reduced through a local-accessor buffer with group barriers.  This replaces the CUDA
// block-per-row launch with `threads_per_row` threads and its shared-memory tree, and preserves the lane
// stride (consecutive items touch consecutive codes/activations) that the split exists for.
//
// The reduction is a fixed tree rather than atomics, so the result is DETERMINISTIC for a given threads_per_row
// - a race would make the parity test flaky rather than wrong, which is the worst kind of failing test.
//
// NOTE ON `g = i >> group_shift`.  The group index was `i / group_elems` with `group_elems` a RUNTIME value, and
// a 64-bit integer division on this hardware is a long instruction sequence - one per element, in the hottest
// loop in the engine.  Every group size this format defines is a POWER OF TWO (16, 32, 64), so the divisor is
// passed as its logarithm and the division becomes a shift; the host refuses a non-power-of-two group rather
// than silently computing a wrong index.
//
// The CUDA version copies `kIq4Nl` into per-block shared memory because a divergent 16-entry constant read is
// the pattern constant memory handles worst (measured at 2.12x of this whole kernel).  icpx 2026.1 exposes no
// `__constant__`, so - exactly like the naive `s_gemv` port above - the SYCL port reads the file-scope
// constant directly.  There is no early return before the barrier to worry about: the grid is exactly
// `n_out` work-groups, so every work-group that starts also reaches the reduction.
template <int CODE_BITS>
void s_gemv_split_submit(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                         float* y, int64_t n_in, int64_t n_out, int bias, int codebook, int group_elems,
                         int group_shift, int has_offset, int threads_per_row, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) threads_per_row), h);
        h.parallel_for(
            sycl::nd_range<1>((size_t) n_out * (size_t) threads_per_row, (size_t) threads_per_row),
            [=](sycl::nd_item<1> item) {
                const long long o = (long long) item.get_group(0);
                const int tid = (int) item.get_local_id(0);

                constexpr int PER_BYTE = 8 / CODE_BITS;
                const long long n_groups = n_in / group_elems;
                const uint8_t* c = codes + o * (n_in / PER_BYTE);
                const float* s = scales + o * n_groups;
                const float* off = has_offset ? offset + o * n_groups : nullptr;

                // FOUR ACCUMULATORS, NOT ONE.  The constraint is the DEPENDENCY CHAIN: a single `acc += ...`
                // serialises every FMA behind the previous one's latency, so the machine sits idle.  Four
                // independent accumulators let the scheduler keep four in flight, and the loop is unrolled by
                // four so the chains interleave.  The summation ORDER changes, which is why the parity test
                // carries a tolerance and this output is checked against the naive one.
                //
                // FOUR CONSECUTIVE ELEMENTS PER ITEM, from ONE code load and ONE scale - the same
                // transformation the Q8 kernel got, for the same reason and with the same guard.  Element
                // `i+k` is bits `[k*CODE_BITS, (k+1)*CODE_BITS)` of the little-endian word at byte `i/PER_BYTE`.
                constexpr int QE = 4;
                constexpr int QB = QE * CODE_BITS / 8;             // 1 for S2, 2 for S4, 4 for S8
                constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                long long i = (long long) tid * QE;
                for (; i + QE <= n_in; i += (long long) threads_per_row * QE) {
                    const long long g = i >> group_shift;
                    const float d = s[g];
                    const float b = off ? off[g] : 0.0f;
                    const uint8_t* cp = c + i / PER_BYTE;
                    unsigned v;
                    if (QB == 1) v = cp[0];
                    else if (QB == 2) v = *(const uint16_t*) cp;
                    else v = *(const uint32_t*) cp;
                    // four halves; `i` is a multiple of four
                    const float f0 = f32_from_f16(x[i + 0]);
                    const float f1 = f32_from_f16(x[i + 1]);
                    const float f2 = f32_from_f16(x[i + 2]);
                    const float f3 = f32_from_f16(x[i + 3]);
                    acc0 += (decode<CODE_BITS>((int) (v & MASK), bias, codebook) * d + b) * f0;
                    acc1 += (decode<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook) * d + b) * f1;
                    acc2 +=
                        (decode<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b) * f2;
                    acc3 +=
                        (decode<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b) * f3;
                }
                // the last, partial quad - at most one per item
                for (; i < n_in; i += (long long) threads_per_row * QE) {
                    for (int k = 0; k < QE && i + k < n_in; ++k) {
                        const long long e = i + k;
                        const long long g = e >> group_shift;
                        const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
                        acc0 += (decode<CODE_BITS>(code, bias, codebook) * s[g] + (off ? off[g] : 0.0f)) *
                                f32_from_f16(x[e]);
                    }
                }
                partial[tid] = (acc0 + acc1) + (acc2 + acc3);
                sycl::group_barrier(item.get_group());
                for (int step = threads_per_row / 2; step > 0; step >>= 1) {
                    if (tid < step) partial[tid] += partial[tid + step];
                    sycl::group_barrier(item.get_group());
                }
                if (tid == 0) y[o] = partial[0];
            });
    });
}

// The Q8_K path, with the SAME weight decode as `s_gemv` and a different activation loader.  One work-item per
// output row, the same shape as the CUDA `<<<blocks, THREADS>>>` launch (the `blockDim` only affects how many
// rows share a work-group in CUDA; here each work-item owns one row).
template <int CODE_BITS>
void s_gemv_q8k_submit(const uint8_t* x, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, int bias, int codebook, int group_elems,
                       int has_offset, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_out), [=](sycl::id<1> o) {
            constexpr int PER_BYTE = 8 / CODE_BITS;
            const long long n_groups = n_in / group_elems;
            const long long codes_per_row = n_in / PER_BYTE;
            const uint8_t* c = codes + o * codes_per_row;
            const float* s = scales + o * n_groups;
            const float* off = has_offset ? offset + o * n_groups : nullptr;

            float acc = 0.0f;
            for (long long g = 0; g < n_groups; ++g) {
                const float d = s[g];
                const float b = off ? off[g] : 0.0f;
                const long long base = g * (long long) group_elems;
                for (int j = 0; j < group_elems; ++j) {
                    const long long i = base + j;
                    const int code =
                        (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
                    // The offset belongs to the WEIGHT and is applied before the activation multiply - see the
                    // note in `s_gemv`, where writing it the other way rounded differently.
                    const float w = decode<CODE_BITS>(code, bias, codebook) * d + b;
                    acc += w * q8k_at(x, i);
                }
            }
            y[o] = acc;
        });
    });
}

// WARP (32) work-items per output row, items striding the reduction axis: for a fixed `i` consecutive items read
// consecutive activations, which is the coalescing that matters once the output width is small.
//
// **`group_shift`, NOT `group_elems`, AND IT COSTS A DIVISION PER ELEMENT IF YOU GET IT WRONG.**  The sibling
// fp16 kernel was given this treatment (a 64-bit integer division is a long instruction sequence - one per
// element, in the hottest loop in the engine) and this one was not: it kept `i / group_elems` with
// `group_elems` a RUNTIME value, once per element, in the kernel that carries most of the model's weight
// bytes.  Every group size the format defines is a power of two, so the divisor is passed as its logarithm and
// the division becomes a shift; the host refuses a non-power-of-two group rather than computing a wrong index.
//
// ONE KERNEL FOR BOTH QUANTIZED ACTIVATIONS, templated on which block format `x` holds.  A second copy with a
// different loader would be a second thing to get wrong, and the only difference IS the loader - the weight
// decode, the quad loop, the codebook and the reduction are identical.
//
// The CUDA version uses one warp per row and reduces with `__shfl_down_sync`.  icpx 2026.1 exposes no subgroup
// shuffle, so this port uses one work-group of 32 per row and reduces the same tree through a local-accessor
// buffer with group barriers; the lane partials and the tree order are the same, so the result matches.
template <int CODE_BITS, bool Q8K>
void s_gemv_q8_split_submit(const uint8_t* x, const uint8_t* codes, const float* scales, const float* offset,
                            float* y, int64_t n_in, int64_t n_out, int bias, int codebook, int group_shift,
                            int has_offset, sycl::queue* q) {
    constexpr int WARP = 32;
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>((size_t) WARP), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_out * (size_t) WARP, (size_t) WARP),
                       [=](sycl::nd_item<1> item) {
                           const long long o = (long long) item.get_group(0);
                           const int lane = (int) item.get_local_id(0);

                           constexpr int PER_BYTE = 8 / CODE_BITS;
                           const long long n_groups = n_in >> group_shift;
                           const long long codes_per_row = n_in / PER_BYTE;
                           const uint8_t* c = codes + o * codes_per_row;
                           const float* s = scales + o * n_groups;
                           const float* off = has_offset ? offset + o * n_groups : nullptr;

                           float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                           float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
                           float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
                           float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
                           // ---- SIXTEEN CONSECUTIVE ELEMENTS PER LANE, from FOUR code loads and ONE scale.
                           //
                           // Four consecutive elements share one code word (1, 2 or 4 bytes depending on width)
                           // and lie in one group, because every group size this format defines is a multiple
                           // of four; sixteen do the same whenever `group_elems % 16 == 0`, which the host
                           // asserts.  So a lane takes an octet of code words:
                           //
                           //     four code loads    instead of sixteen
                           //     one scale load     instead of sixteen
                           //     one shift per element instead of a divide, a modulo and a shift
                           //
                           // THE BIT ORDER IS THE SAME ONE: element `i+k` is bits `[k*CODE_BITS,
                           // (k+1)*CODE_BITS)` of the little-endian word at byte `i/PER_BYTE`.
                           //
                           // THE SUMMATION ORDER CHANGES, which is why the parity tests carry tolerances and
                           // compare against the naive kernel.
                           constexpr int QE = 16;
                           // BYTES PER FOUR-CODE WORD, not per octet.
                           constexpr int QW = 4 * CODE_BITS / 8;             // 1 for S2, 2 for S4, 4 for S8
                           constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
                           long long i = (long long) lane * QE;
                           for (; i + QE <= n_in; i += 32 * QE) {
                               const long long g =
                                   i >> group_shift;  // the whole octet is in one group: group_elems % 16 == 0
                               const float d = s[g];
                               const float b = off ? off[g] : 0.0f;
                               const uint8_t* cp = c + i / PER_BYTE;
                               unsigned v, v2, v3, v4;
                               if (QW == 1) {
                                   v = cp[0];
                                   v2 = cp[1];
                                   v3 = cp[2];
                                   v4 = cp[3];
                               } else if (QW == 2) {
                                   v = *(const uint16_t*) cp;
                                   v2 = *(const uint16_t*) (cp + 2);
                                   v3 = *(const uint16_t*) (cp + 4);
                                   v4 = *(const uint16_t*) (cp + 6);
                               } else {
                                   v = *(const uint32_t*) cp;
                                   v2 = *(const uint32_t*) (cp + 4);
                                   v3 = *(const uint32_t*) (cp + 8);
                                   v4 = *(const uint32_t*) (cp + 12);
                               }
                               // ---- THE ACTIVATION BLOCK IS HOISTED OUT OF THE SIXTEEN ELEMENTS.
                               //
                               // `q8k_at(x, i + k)` computes `(i + k) / 256` and `(i + k) % 256` on EVERY
                               // call: a 64-bit integer division AND modulo per element, sixteen times per
                               // lane-iteration.  A 64-bit division is a software routine, and the compiler
                               // cannot strength-reduce it.
                               //
                               // But `QE` is 16, `Q8K_BLOCK_ELEMS` is 256 and `Q8_0_BLOCK_ELEMS` is 32, and
                               // the loop advances `i` by `32 * QE`.  So `i % 256` and `i % 32` are ALWAYS
                               // multiples of 16 - which means all QE elements of one lane-iteration lie
                               // inside ONE block, for either activation kind.  One division and one modulo per
                               // iteration replace sixteen of each, and the sixteen activations become sixteen
                               // consecutive int8 loads from a pointer computed once.  This is exact, not an
                               // approximation: `xi + QE <= blk_elems` holds for every reachable `xi`.
                               const int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
                               const uint8_t* xb =
                                   x + (i / blk_elems) * (Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
                               const int xi = (int) (i % blk_elems);
                               const float xd = Q8K ? *reinterpret_cast<const float*>(xb)
                                                    : f32_from_f16(*reinterpret_cast<const uint16_t*>(xb));
                               const int8_t* xq = reinterpret_cast<const int8_t*>(xb + (Q8K ? 4 : 2)) + xi;

                               const float w0 =
                                   decode<CODE_BITS>((int) (v & MASK), bias, codebook) * d + b;
                               const float w1 = decode<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook) * d + b;
                               const float w2 = decode<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w3 = decode<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w4 = decode<CODE_BITS>((int) (v2 & MASK), bias, codebook) * d + b;
                               const float w5 = decode<CODE_BITS>((int) ((v2 >> CODE_BITS) & MASK), bias, codebook) * d + b;
                               const float w6 = decode<CODE_BITS>((int) ((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w7 = decode<CODE_BITS>((int) ((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               acc0 += w0 * (xd * (float) xq[0]);
                               acc1 += w1 * (xd * (float) xq[1]);
                               acc2 += w2 * (xd * (float) xq[2]);
                               acc3 += w3 * (xd * (float) xq[3]);
                               acc4 += w4 * (xd * (float) xq[4]);
                               acc5 += w5 * (xd * (float) xq[5]);
                               acc6 += w6 * (xd * (float) xq[6]);
                               acc7 += w7 * (xd * (float) xq[7]);
                               const float w8 = decode<CODE_BITS>((int) (v3 & MASK), bias, codebook) * d + b;
                               const float w9 = decode<CODE_BITS>((int) ((v3 >> CODE_BITS) & MASK), bias, codebook) * d + b;
                               const float w10 = decode<CODE_BITS>((int) ((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w11 = decode<CODE_BITS>((int) ((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w12 = decode<CODE_BITS>((int) (v4 & MASK), bias, codebook) * d + b;
                               const float w13 = decode<CODE_BITS>((int) ((v4 >> CODE_BITS) & MASK), bias, codebook) * d + b;
                               const float w14 = decode<CODE_BITS>((int) ((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               const float w15 = decode<CODE_BITS>((int) ((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook) * d + b;
                               acc8 += w8 * (xd * (float) xq[8]);
                               acc9 += w9 * (xd * (float) xq[9]);
                               acc10 += w10 * (xd * (float) xq[10]);
                               acc11 += w11 * (xd * (float) xq[11]);
                               acc12 += w12 * (xd * (float) xq[12]);
                               acc13 += w13 * (xd * (float) xq[13]);
                               acc14 += w14 * (xd * (float) xq[14]);
                               acc15 += w15 * (xd * (float) xq[15]);
                           }
                           // the last, partial octet - at most one per lane; its elements do not share a block
                           for (; i < n_in; i += 32 * QE) {
                               for (int k = 0; k < QE && i + k < n_in; ++k) {
                                   const long long e = i + k;
                                   const long long g = e >> group_shift;
                                   const int code =
                                       (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
                                   acc0 += (decode<CODE_BITS>(code, bias, codebook) * s[g] +
                                            (off ? off[g] : 0.0f)) *
                                           (Q8K ? q8k_at(x, e) : q8_0_at(x, e));
                               }
                           }
                           float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                                       (((acc8 + acc9) + (acc10 + acc11)) +
                                        ((acc12 + acc13) + (acc14 + acc15)));
                           partial[lane] = acc;
                           sycl::group_barrier(item.get_group());
                           for (int step = 16; step > 0; step >>= 1) {
                               if (lane < step) partial[lane] += partial[lane + step];
                               sycl::group_barrier(item.get_group());
                           }
                           if (lane == 0) y[o] = partial[0];
                       });
    });
}

static void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales,
                              const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form,
                              int threads_per_row, sycl::queue* q, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024) {
        std::fprintf(stderr, "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    // Every group size this format defines is a power of two, which is what lets the per-element group index
    // be a shift.  Refuse anything else rather than compute a wrong index quietly.
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP: four consecutive elements share a scale and the kernel reads
    // exactly one.  Every group size this format defines is 16, 32 or 64, so this cannot fire on a real pack.
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    try {
        switch (form.code_bits) {
        case 2:
            s_gemv_split_submit<2>(x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                   form.group_elems, group_shift, ho, threads_per_row, q);
            break;
        case 4:
            s_gemv_split_submit<4>(x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                   form.group_elems, group_shift, ho, threads_per_row, q);
            break;
        case 8:
            s_gemv_split_submit<8>(x, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                   form.group_elems, group_shift, ho, threads_per_row, q);
            break;
        default:
            std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
        }
        if (sync) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv_split launch: %s\n", e.what());
        std::exit(1);
    }
}

bool q8k_form_ok(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of group_elems %d\n", who, (long long) n_in,
                     form.group_elems);
        return false;
    }
    if (n_in % Q8K_BLOCK_ELEMS != 0) {
        // `quantize_q8_K` requires this too, and a partial block would read past the end of the activation.
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of the Q8_K block %d\n", who, (long long) n_in,
                     Q8K_BLOCK_ELEMS);
        return false;
    }
    return true;
}

}  // namespace

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    // The CUDA entry point uses the default stream and `cudaStreamSynchronize`s, so this uses the default queue
    // and waits.
    sycl::queue* q = static_cast<sycl::queue*>(strata::core::default_sycl_queue());
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, q, true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, q, false);
}

// ---- the Q8_K entry points -----------------------------------------------------------------------------

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k")) std::exit(1);
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        switch (form.code_bits) {
        case 4:
            s_gemv_q8k_submit<4>(x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                 form.group_elems, ho, q);
            break;
        case 8:
            s_gemv_q8k_submit<8>(x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                 form.group_elems, ho, q);
            break;
        default:
            // 2-bit S2 never has a Q8_K activation: its `vec_dot_type` is Q8_0.  Refusing is better than
            // running a kernel that would be numerically wrong in a way the caller cannot see.
            std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract "
                                 "(S2 uses Q8_0; see docs/activation-contract.md)\n", form.code_bits);
            std::exit(1);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv_q8k launch: %s\n", e.what());
        std::exit(1);
    }
}

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k_split")) std::exit(1);
    // THE GROUP SIZE IS PASSED AS ITS LOGARITHM: see the note on the kernel.  A non-power-of-two group would
    // make the shift a WRONG INDEX rather than a slow one, so it is refused here.
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    if ((1 << group_shift) != form.group_elems) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a power of two\n", form.group_elems);
        std::exit(1);
    }
    // AND THE OCTET MUST FIT INSIDE ONE GROUP.  Sixteen consecutive elements share a scale, so a group smaller
    // than sixteen would need two of them and the kernel reads exactly one.  Every group size this format
    // defines is 16, 32 or 64, so this cannot fire on a real pack - it is here because a future one could.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        switch (form.code_bits) {
        case 4:
            s_gemv_q8_split_submit<4, true>(x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                            group_shift, ho, q);
            break;
        case 8:
            s_gemv_q8_split_submit<8, true>(x_q8k, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                            group_shift, ho, q);
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8k_split: code_bits %d has no Q8_K contract\n", form.code_bits);
            std::exit(1);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv_q8k_split launch: %s\n", e.what());
        std::exit(1);
    }
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION IS `block_q8_0`, 32 elements per block, so `n_in` must be a multiple of 32 - which is a
    // WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist: `ffn_down_shexp` has
    // n_in = 640, a multiple of 32 that is NOT a multiple of 256.
    if (n_in % Q8_0_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: n_in %lld is not a multiple of %d\n", (long long) n_in,
                     Q8_0_BLOCK_ELEMS);
        std::exit(1);
    }
    // the same two checks `s_gemv_q8k_split` makes, for the same reasons - see the notes there
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;

    const int cb = (int) form.codebook;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        switch (form.code_bits) {
        case 4:
            s_gemv_q8_split_submit<4, false>(x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                             group_shift, ho, q);
            break;
        case 8:
            s_gemv_q8_split_submit<8, false>(x_q8_0, codes, scales, offset, y, n_in, n_out, form.code_bias, cb,
                                             group_shift, ho, q);
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8_0_split: code_bits %d has no Q8_0 contract\n", form.code_bits);
            std::exit(1);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "s_gemv_q8_0_split launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
