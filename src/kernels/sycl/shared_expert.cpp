// src/kernels/sycl/shared_expert.cpp - SYCL port of src/kernels/cuda/shared_expert.cu (P2.S2: the shared expert).
//
// `ref/moe.py::shared_expert`, transcribed:
//
//     h = silu(x @ gate_shexp.T) * (x @ up_shexp.T)      <- SILU GOES ON GATE, not on up
//     h = h @ down_shexp.T                                (nt, n_embd)
//     g = sigmoid(x @ gate_inp_shexp)                     (nt,) ONE SCALAR PER TOKEN
//     return h * g[:, None]
//
// TWO THINGS HERE WERE BELIEVED WRONG FOR MANY ROUNDS and `docs/semantics.md` records both.  SILU GOES ON THE
// GATE TENSOR: the opposite reading is plausible, produces the right shapes, and is wrong.  And
// `ffn_gate_inp_shexp` is `(n_embd,)` whose output is "one value per token" - a SCALAR gate obtained by dotting
// that vector with the hidden state, NOT a per-expert gate and NOT a per-dimension elementwise one.  Both were
// live readings of the same shapes until the source comment settled them.
//
// The result is ADDED to the routed output - not router-weighted, not renormalised against it.
//
// The legacy canonical projections use their explicit Q8_0/Q8_K activation images. The optional native
// BF16 path affects only the scalar gate: it reads the original F32 input and uses pinned CUDA MMVF plus
// an FP32 sigmoid. Native projection overrides independently select CUDA Q8_1 MMVQ and FP32 SwiGLU.
//
// THE SYCL PORT.  icpx 2026.1 exposes no warp-shuffle surface, so every block reduction below (the scalar
// gate's 2560-element double dot, the `<<<1,256>>>` tree) becomes ONE WORK-ITEM walking the reduction serially -
// the shape `native_gr_norm` uses.  The GEMV/quantization dependency helpers keep their own SYCL ports and are
// called by name.  The caller's non-null streams are IN-ORDER, so submission order is execution order and no
// host barrier is needed between dependent stages - which is also what keeps the function recordable into a
// command graph, where a host `q->wait()` is illegal.  Only the null stream (the out-of-order default queue)
// is followed by a gated `if (stream == nullptr) q->wait()`.  The FP32 fast-math intrinsics of the native
// path (`__expf`, `__fdividef`) are approximated with `sycl::exp` and plain division - recorded, not hidden: the
// native swiglu/sigmoid bytes are not reproducible without `--use_fast_math`.
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

// `f32_to_f16` used to live here as a private copy.  Round 198 found it - and the identical one in
// `quantize_act.cu` - returning a NaN for every FINITE value that overflows fp16 (1e30, 65536, 1e45) instead
// of saturating to inf, because `exp >= 31` conflates an f32 inf/NaN with an out-of-range finite exponent.
// This file's fixture is all O(1), so it could not see it.  The conversion now lives in one place, with the
// regimes spelled out.

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works in
// float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
void swiglu_run(const float* gate, const float* up, float* out, int n, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            const double x = (double) gate[i];
            out[i] = (float) (x / (1.0 + sycl::exp(-x))) * up[i];
        });
    });
}

void native_swiglu_run(const float* gate, const float* up, float* out, int n, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            // Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply.  `__expf`/`__fdividef`
            // are the CUDA fast-math operations; SYCL has no exact equivalent, so this is the plain f32 form.
            out[i] = gate[i] / (1.0f + sycl::exp(-gate[i])) * up[i];
        });
    });
}

void to_f16_run(const float* in, uint16_t* out, int n, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] = f16_from_f32(in[i]); });
    });
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).
//
// **ONE THREAD WAS THE LARGEST SINGLE KERNEL IN THE MODEL, AND THE COMMENT DEFENDING IT SAID WHY IT SHOULD
// NOT BE.**  It read: "n_embd is 2560, so this is a short serial loop and the parallelism is elsewhere."
// 2560 iterations is not short when each one is a DOUBLE multiply-add.  This card's double rate is **1/64** of
// its float rate, so the loop is a 2560-long dependency chain through a 1/64-rate unit - tens of microseconds
// from the chain alone, before counting 2560 sequential global loads issued by one thread with no coalescing.
//
// MEASURED, and this is what the number looks like from outside: `shared_expert` is **0.2374 ms of a
// 0.8883 ms block - 26.7% of the whole block, 11.4 ms per token** - for three projections that read 5.5 MB and
// whose memory floor is 0.009 ms.  It is seven kernels, and six of them are GEMVs or elementwise passes over
// 640-2560 elements.  This is the one that cannot be explained by bandwidth.
//
// THE DOUBLE ACCUMULATOR IS KEPT.  `ref/gr.py` reduces in float64 and `block_sum` in `gr.cu` exists for the
// same reason - the summation order of a tree is not the reference's, so the defence is enough precision that
// the order stops mattering.  The SYCL port returns to ONE work-item for the reducing dot (no warp surface to
// split into ten terms per thread), but the double accumulator and the final double sigmoid are exact.
//
// **BOTH OPERANDS ARE BF16, AND THAT IS THE REFERENCE'S CONTRACT, NOT A STORAGE DETAIL.**
void scalar_gate_run(const uint16_t* x_bf16, const uint16_t* w_bf16, float* out, int n_embd, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            double acc = 0.0;
            for (int i = 0; i < n_embd; ++i)
                acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
            out[0] = (float) (1.0 / (1.0 + sycl::exp(-acc)));
        });
    });
}

void native_scalar_sigmoid_run(float* gate, sycl::queue* q) {
    // Match the single-token CUDA sigmoid's FP32 fast-math operations.  The dot product was already reduced by
    // the pinned native MMVF implementation.  `__fdividef`/`__expf` have no exact SYCL equivalent.
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { gate[0] = 1.0f / (1.0f + sycl::exp(-gate[0])); });
    });
}

void scale_run(float* out, const float* g, int n, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { out[i] *= g[0]; });
    });
}

void scale_rows_run(float* out, const float* g, int n, int n_tok, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_tok * (size_t) n), [=](sycl::id<1> idx) {
            const int t = (int) ((size_t) idx / (size_t) n);
            const int i = (int) ((size_t) idx % (size_t) n);
            out[(size_t) t * n + i] *= g[t];
        });
    });
}

}  // namespace

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }

namespace {

/// The MoE block's final combination.  See the header for the two readings it exists to pin.
void moe_combine_run(const float* parts, const float* weights, const float* shared, float* y, int n_embd, int k,
                     sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_embd), [=](sycl::id<1> j) {
            // Accumulate in DOUBLE, in the reference's own order (`for i in range(k): out[t] += w[t,i] * g[0]`).
            // f32 would be defensible for ten terms, but the reference is float64 and this is one line.
            double acc = 0.0;
            for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
            // The SHARED output is added PLAIN - not router-weighted, not renormalised against the routed sum.
            if (shared) acc += (double) shared[j];
            y[j] = (float) acc;
        });
    });
}

}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    sycl::queue* q = static_cast<sycl::queue*>(stream);
    try {
        native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
        native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
        native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
        if (stream == nullptr) q->wait();
        const int n = (int) (n_ff * n_tok);
        native_swiglu_run(gate, up, gate, n, q);
        if (stream == nullptr) q->wait();
        native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
        native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
        if (stream == nullptr) q->wait();
        for (int t = 0; t < n_tok; ++t) {
            if (native_bf16) {
                bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
                native_scalar_sigmoid_run(g + t, q);
            } else {
                scalar_gate_run(x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t, (int) n_embd, q);
            }
            if (stream == nullptr) q->wait();
        }
        scale_rows_run(out, g, (int) n_embd, n_tok, q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "shared_expert_multi launch: %s\n", e.what());
        std::exit(1);
    }
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH.  This used to be four `cudaMalloc`s and four `cudaFree`s per call, which
    // is illegal during stream capture AND a token-path allocation - P2.T10 forbids both.
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`: the three
    // families cannot be told apart by the other fields, so the kind is carried rather than derived.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    try {
        // gate and up projections, then silu(gate) * up in place in `gate`
        if (native_gate || native_up)
            native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
        if (native_gate)
            native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
        else
            gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
        if (stream == nullptr) q->wait();
        if (native_up)
            native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
        else
            gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
        if (stream == nullptr) q->wait();
        if (native_projection)
            native_swiglu_run(gate, up, gate, (int) n_ff, q);
        else
            swiglu_run(gate, up, gate, (int) n_ff, q);
        if (stream == nullptr) q->wait();

        // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT - which is
        // what `ggml_mul_mat` does for every matmul in the model.  It used to be rounded to fp16 with no
        // justification beyond "the kernel takes fp16".
        if (native_down) {
            native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
            native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
        } else if (down_form.act_kind == 1) {
            if (n_ff % 256 != 0) {
                std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                     "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
                std::exit(1);
            }
            quantize_q8_K(gate, h_q8k, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else if (down_form.code_bits == 2) {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        } else {
            quantize_q8_0(gate, h_q8_0, n_ff, stream);
            gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
        }
        if (stream == nullptr) q->wait();

        // the per-token scalar gate, then the multiply.  Note the gate is computed from `x`, the ORIGINAL hidden
        // state, not from anything the expert produced. The historical branch uses BF16-rounded inputs; the
        // native branch uses the pinned CUDA FP32 activation contract.
        // `<<<1, 256>>>`: one block, because the output is ONE scalar and a second block would only add a global
        // round trip.  256 threads is the reduction's width, not the problem's size.
        if (use_native) {
            bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
            native_scalar_sigmoid_run(g, q);
        } else {
            scalar_gate_run(x_bf16, gate_inp_bf16, g, (int) n_embd, q);
        }
        if (stream == nullptr) q->wait();
        scale_run(out, g, (int) n_embd, q);

        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "shared_expert launch: %s\n", e.what());
        std::exit(1);
    }
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        moe_combine_run(parts, weights, shared, y, (int) n_embd, (int) k, q);
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "moe_combine launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
