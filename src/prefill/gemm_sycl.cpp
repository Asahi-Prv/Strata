// src/prefill/gemm_sycl.cpp - SYCL port of src/prefill/gemm.cu (plan v0.3 P5, the batched prefill GEMMs).
//
// Same public API as include/strata/prefill/gemm.hpp.  The cuBLAS handle becomes a oneMKL GEMM call on the
// queue carried as `void* stream`; a CUDA `cudaStream_t` and a `sycl::queue*` are both opaque 64-bit handles,
// so the wire format is unchanged (see strata/core/device_runtime.hpp).
//
// LAYOUT.  gemm.hpp states the one convention every projection uses: Y[T, N] = X[T, K] . W[N, K]^T with W
// row-major (the GGUF / pack layout), X row-major, Y FP32 with row stride ldy.  oneMKL's row_major::gemm
// computes C = alpha*op(A)*op(B) + beta*C with A m x k, B k x n, C m x n.  Taking m = T, n = N, k = K,
// A = X (nontrans, lda = K), B = W (trans, stored N x K row-major with ldb = K) and C = Y (ldc = ldy) gives
// exactly Y[t, n] = sum_k X[t, k] * W[n, k] with the same beta semantics the cublasGemmEx call had.
//
// The CUDA version used `cublasGemmEx(handle, OP_T, OP_N, N, T, K, ..., W, OP16, K, X, OP16, K, ..., Y, F32,
// ldy, CUBLAS_COMPUTE_32F)` - a column-major view of the same identity.  FP16 inputs go through
// `sycl::half` with an FP32 output (the `(half, half, float, float)` oneMKL overload); BF16 inputs through
// `oneapi::mkl::bfloat16` with an FP32 output.  At compile time the task's toolchain has oneMKL, so that is
// the path taken; a plain tiled SYCL GEMM is kept behind `__has_include` for a toolchain without it.
#include "strata/prefill/gemm.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#if defined(__has_include)
#if __has_include(<oneapi/mkl.hpp>)
#include <oneapi/mkl.hpp>
#define STRATA_PREFILL_HAVE_ONEMKL 1
#endif
#endif
#ifndef STRATA_PREFILL_HAVE_ONEMKL
#define STRATA_PREFILL_HAVE_ONEMKL 0
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::prefill {

// The 16-bit type a GEMM operand pointer is reinterpreted to.  With oneMKL this is its own alias; in the
// plain-SYCL fallback there is no such type, so a two-byte wrapper that widens with the validated bit
// converter stands in (it must name `oneapi::mkl::bfloat16` in neither path's absence).
#if STRATA_PREFILL_HAVE_ONEMKL
using Bf16T = oneapi::mkl::bfloat16;
#else
struct Bf16T {
    uint16_t bits;
    explicit operator float() const { return strata::kernels::f32_from_bf16(bits); }
};
#endif

namespace {

sycl::queue* queue_of(void* stream) {
    return static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
}

#if !STRATA_PREFILL_HAVE_ONEMKL
// A plain tiled SYCL GEMM, used only when oneMKL is not installed.  One work-group computes a TILE x TILE
// block of Y; the K loop walks the reduction axis with the two tiles staged in local memory.  bf16/f16 bits
// are widened to f32 exactly (every product then matches the tensor-core path up to the accumulation order).
template <typename HalfT>
void gemm16_fallback(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                     float beta, sycl::queue* q) {
    constexpr int TILE = 16;
    const HalfT* Xh = reinterpret_cast<const HalfT*>(X);
    const HalfT* Wh = reinterpret_cast<const HalfT*>(W);
    const int64_t blocks_n = (N + TILE - 1) / TILE;
    try {
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> xa(sycl::range<1>((size_t)(TILE * TILE)), h);
            sycl::local_accessor<float, 1> wa(sycl::range<1>((size_t)(TILE * TILE)), h);
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>((size_t)((T + TILE - 1) / TILE) * TILE, (size_t)blocks_n * TILE),
                                  sycl::range<2>(TILE, TILE)),
                [=](sycl::nd_item<2> item) {
                    const int64_t row = (int64_t) item.get_global_id(0), col = (int64_t) item.get_global_id(1);
                    const int lr = (int) item.get_local_id(0), lc = (int) item.get_local_id(1);
                    float acc = 0.0f;
                    for (int64_t k0 = 0; k0 < K; k0 += TILE) {
                        const int64_t kr = k0 + lr, kc = k0 + lc;
                        xa[lr * TILE + lc] = (row < T && kr < K) ? (float) Xh[row * K + kr] : 0.0f;
                        wa[lr * TILE + lc] = (col < N && kc < K) ? (float) Wh[col * K + kc] : 0.0f;
                        sycl::group_barrier(item.get_group());
                        for (int kk = 0; kk < TILE; ++kk) acc += xa[lr * TILE + kk] * wa[kk * TILE + lc];
                        sycl::group_barrier(item.get_group());
                    }
                    if (row < T && col < N) {
                        float* y = Y + row * ldy + col;
                        *y = (beta == 0.0f) ? acc : beta * *y + acc;
                    }
                });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "prefill gemm: fallback gemm: %s\n", e.what());
        std::exit(1);
    }
}
#endif

template <typename HalfT>
void gemm16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
            float beta, sycl::queue* q, const char* what) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#if STRATA_PREFILL_HAVE_ONEMKL
    const float alpha = 1.0f;
    try {
        oneapi::mkl::blas::row_major::gemm(*q, oneapi::mkl::transpose::N, oneapi::mkl::transpose::T,
                                           (std::int64_t) T, (std::int64_t) N, (std::int64_t) K, alpha,
                                           reinterpret_cast<const HalfT*>(X), (std::int64_t) K,
                                           reinterpret_cast<const HalfT*>(W), (std::int64_t) K, beta, Y,
                                           (std::int64_t) ldy);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "prefill gemm: %s: %s\n", what, e.what());
        std::exit(1);
    }
#else
    (void) what;
    gemm16_fallback<HalfT>(X, W, Y, T, N, K, ldy, beta, q);
#endif
}

}  // namespace

Gemm::~Gemm() {
    if (external_) return;
    if (scratch_ == nullptr && workspace_ == nullptr) return;
    try {
        sycl::context ctx = queue_of(stream_)->get_context();
        if (scratch_ != nullptr) sycl::free(scratch_, ctx);
        if (workspace_ != nullptr) sycl::free(workspace_, ctx);
    } catch (...) {
        // A backend that has already torn down cannot free; nothing useful to do at destruction.
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    (void) err;
    (void) ws_bytes;
    stream_ = stream;
    external_ = true;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    sycl::queue* q = queue_of(stream);
    stream_ = stream;
    try {
        const size_t ws = 32u << 20;
        workspace_ = sycl::malloc_device(ws, *q);
        if (workspace_ == nullptr) { err = "prefill gemm: workspace"; return false; }
        if (scratch_elems > 0) {
            scratch_ = static_cast<uint16_t*>(sycl::malloc_device((size_t) scratch_elems * 2, *q));
            if (scratch_ == nullptr) {
                err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
                return false;
            }
        }
    } catch (const sycl::exception& e) {
        err = std::string("prefill gemm: ") + e.what();
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    gemm16<Bf16T>(X, W, Y, T, N, K, ldy, beta, queue_of(stream_), "bf16");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    gemm16<sycl::half>(X, W, Y, T, N, K, ldy, beta, queue_of(stream_), "f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices (identical to gemm.cu).
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) {
            std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K);
            std::exit(1);
        }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
