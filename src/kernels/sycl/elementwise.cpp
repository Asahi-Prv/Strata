// src/kernels/sycl/elementwise.cpp - SYCLomatic port of the simplest glue kernels in src/kernels/cuda/elementwise.cu.
//
// These are the plain elementwise members of the P2.S5 glue set: `add_inplace` (dst += src),
// `scale_inplace` (x *= s) and `silu_inplace` (x / (1 + exp(-x))).  They are one work-item per element, a flat
// range<1>, so this is the same shape as the CUDA `<<<grid_for(n), THREADS>>>` launch with the total work
// `n` spread across `n` work-items instead of across grid*THREADS with a bounds check.  `silu_inplace` keeps
// the source's double-precision intermediate (matching `ref/gdn.py`) rather than letting the compiler do an
// f32 exp.  `gdn_gate` (a softplus) and `f32_to_bf16_bulk` (the bf16 conversion) are the same one-per-element
// shape.
#include "strata/kernels/elementwise.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.  `exp(89)`
// overflows f32 and `exp(20)` is already ~5e8 where `log1p` loses precision, so above 20 the result is `x`.
inline float softplus_dev(float x) { return x > 20.0f ? x : std::log1pf(std::expf(x)); }
}  // namespace

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { dst[i] += src[i]; });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "add_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { x[i] *= s; });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "scale_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
                const double v = (double) x[i];
                x[i] = (float) (v / (1.0 + std::exp(-v)));
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "silu_inplace launch: %s\n", e.what());
        std::exit(1);
    }
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
                const int64_t h = (int64_t) i % h_v;
                gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_gate launch: %s\n", e.what());
        std::exit(1);
    }
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = bf16_from_f32(x[i]); });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "f32_to_bf16_bulk launch: %s\n", e.what());
        std::exit(1);
    }
}

// The packed-code embedding decoder.  The CUDA kernel is one thread per output element with a
// `blockIdx*blockDim + threadIdx` index, which is exactly a flat `range<1>(n)` work-item.  The bit unpack is
// the source's, byte-for-byte: `per_byte` codes per byte, low bits first, then a separately-rounded multiply
// and add (the source's `__fmul_rn`/`__fadd_rn`, which is why `a*b + off` is kept as two operations rather
// than an FMA).
void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) {
                const int64_t i = (int64_t) id;
                const int per_byte = 8 / code_bits;
                const unsigned mask = (1u << code_bits) - 1u;
                const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
                const int64_t group = i / group_elems;
                const float product = sycl::fma((float) (code + code_bias), scales[group], 0.0f);
                out[i] = sycl::fma(product, 1.0f, offsets ? offsets[group] : 0.0f);
            });
        });
        // The CUDA launcher did NOT sync_if_needed here; there is no wait on the default queue either.
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "embedding_gather launch: %s\n", e.what());
        std::exit(1);
    }
}

// `y[i] = f16(x[i])`, using the shared round-to-nearest-even converter in f16_bits.hpp.
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) { y[i] = f16_from_f32(x[i]); });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "f32_to_f16_bulk launch: %s\n", e.what());
        std::exit(1);
    }
}

// Weighted RMSNorm over the last axis.  The CUDA kernel is one WARP per row with a shuffle reduction; icpx
// 2026.1 has no subgroup-shuffle surface (see native_gr_norm.cpp), so this is one work-item per row doing the
// whole reduction serially, exactly like native_gr_norm.cpp.  The reduction order differs from the warp tree,
// so it is not bit-identical, but the value is the same MEAN of squares (`acc / cols`, matching
// `ref/qsa.py::rms_norm`, not a `gdn_l2_norm` sum) and the per-element multiply is unchanged.
//
// **THE ROW GUARD'S SEMANTICS.**  In CUDA the grid was rounded up to whole 4-warp blocks and a missing
// `if (row >= rows) return;` let rows 2 and 3 of the 2-row QSA k-norm write 2*cols floats past `b.kcur` into
// `b.vcur` - the QSA bug.  Here the range is exactly `range<1>(rows)`, so no out-of-range work-item can exist
// and no guard is needed; the guard is therefore structurally absent rather than omitted, and the semantics
// (never write past `rows`) are preserved by construction.
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> id) {
                float* r = x + (size_t) id * (size_t) cols;
                float acc = 0.0f;
                for (int64_t c = 0; c < cols; ++c) acc += r[c] * r[c];
                const float scale = sycl::rsqrt(acc / (float) cols + eps);
                for (int64_t c = 0; c < cols; ++c) r[c] = (w ? r[c] * w[c] : r[c]) * scale;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "rms_norm_weighted launch: %s\n", e.what());
        std::exit(1);
    }
}

// THE DOORBELL.  One work-item, one INCREMENT - the cost is the launch.  The fence must be ordered BEFORE the
// increment so the host, which is the reader, cannot observe the ring before the published payload lands;
// `atomic_fence(seq_cst, system)` is the SYCL stand-in for `__threadfence_system()`.
void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                volatile uint32_t* s = d_seq;
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                *s = *s + 1u;
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "doorbell_ring launch: %s\n", e.what());
        std::exit(1);
    }
}

// Spins until the host-written `*d_flag` equals the ring value `*d_seq` (both mapped pinned memory), then
// fences so everything the host wrote before the flag is visible to the nodes that follow.  The CUDA
// `__nanosleep(100)` backoff has no portable SYCL equivalent here, so the poll is a bare spin, matching the
// existing `wait_flag_ge` port in verify_kernels.cpp.  The volatile loads are what stop the compiler from
// hoisting the poll out of the loop.
void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                const volatile uint32_t* vf = d_flag;
                const volatile uint32_t* vs = d_seq;
                const uint32_t want = *vs;
                while (*vf != want) {
                }
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            });
        });
        // The CUDA launcher did not sync here: this kernel is captured once per layer, not run to completion.
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "doorbell_wait launch: %s\n", e.what());
        std::exit(1);
    }
}

// The doorbell's payload and its ring in ONE launch: 1024 work-items (CUDA `<<<1, 1024>>>`), a grid-stride
// copy of `x`, the first `k` ids/weights, a system fence, a work-group barrier so every work-item's stores are
// ordered before the ring, then work-item 0 fences again and increments the mapped sequence.
void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) {
        std::fprintf(stderr, "doorbell_publish: k too large\n");
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        const int ni = (int) n;
        const int ki = (int) k;
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> item) {
                const int tid = (int) item.get_local_id(0);
                for (int i = tid; i < ni; i += 1024) x_out[i] = x[i];
                if (tid < ki) {
                    ids_out[tid] = ids[tid];
                    weights_out[tid] = weights[tid];
                }
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                sycl::group_barrier(item.get_group());
                if (tid == 0) {
                    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
                    volatile uint32_t* s = d_seq;
                    *s = *s + 1u;
                }
            });
        });
        // The CUDA launcher did not sync here.
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "doorbell_publish launch: %s\n", e.what());
        std::exit(1);
    }
}

// Copies `n` floats from mapped pinned host memory into device memory with a kernel, so the handoff stays on
// the compute queue.  `n` is a multiple of 4 and both pointers are 16-byte aligned, so the copy is done in
// 16-byte `sycl::float4` units exactly like the CUDA `float4` kernel.  The CUDA source cast the volatile
// source pointer back to `const float4*` before reading, so this is faithful rather than a simplification.
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::float4* d4 = reinterpret_cast<sycl::float4*>(dst);
    const sycl::float4* s4 = reinterpret_cast<const sycl::float4*>(src);
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n4), [=](sycl::id<1> i) { d4[i] = s4[i]; });
        });
        // The CUDA launcher did not sync here.
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "copy_from_mapped launch: %s\n", e.what());
        std::exit(1);
    }
}

// The int32 twin of `copy_from_mapped`, one 128-work-item block striding `n` like the CUDA `<<<1, 128>>>`,
// with the source read through a volatile pointer as the CUDA kernel does.
void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const int ni = (int) n;
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> item) {
                const int tid = (int) item.get_local_id(0);
                const volatile int32_t* vs = src;
                for (int i = tid; i < ni; i += 128) dst[i] = vs[i];
            });
        });
        // The CUDA launcher did not sync here.
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "copy_i32_from_mapped launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
