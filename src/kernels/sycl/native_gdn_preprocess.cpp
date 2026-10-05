// src/kernels/sycl/native_gdn_preprocess.cpp - SYCL port of src/kernels/cuda/native_gdn_preprocess.cu
// (ggml-cuda/norm.cu, common.cuh, unary.cu, ssm-conv.cu, scale.cu).
//
// Five small GDN preprocessing kernels.  Two of them (`l2_norm`, `out_norm`) are one block per row in CUDA doing a
// warp-XOR butterfly reduce of sum(value^2) plus a shared-memory second level (norm_sum).  icpx 2026.1 exposes none
// of the group-reduction surface (see native_gr_norm.cpp) and an out-of-order queue makes a two-submit partials
// scratch unsafe, so the reduce is done by ONE work-item per row walking the 128 columns serially - the same work, a
// different summation order (validated by max_rel).  `conv_silu`, `beta_sigmoid` and `gate_softplus` are plain
// elementwise, one work-item per element, the same flat `<<<grid, THREADS>>>` shape with `c = blockIdx*blockDim +
// threadIdx` spread across `c` work-items.  This file is compiled with the fast-math contraction the pinned backend
// uses, so the multiply-adds stay contracted as in the CUDA source.
//
// WHY NO __shfl_*: the CUDA norm kernels reduce with `__shfl_xor_sync` then `__syncthreads`; neither the shuffle nor
// a clean block barrier is available here, so the serial per-row pass replaces both.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights,
                          float* raw_output, float* silu_output, int64_t channels,
                          int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (((channels + 255) / 256) * 256)), [=](sycl::id<1> gid) {
                const int c = (int) gid;
                if (c >= (int) channels) return;
                // The native SSM kernel adds its zero bias even when there is no bias input.
                const float values[4] = {history[(size_t) c * 3], history[(size_t) c * 3 + 1],
                                         history[(size_t) c * 3 + 2], input[c]};
                float sum = 0.0f;
                for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[(size_t) c * 4 + tap];
                sum = sum + 0.0f;
                raw_output[c] = sum;
                silu_output[c] = sum / (1.0f + sycl::exp(-sum));
                for (int tap = 0; tap < 3; ++tap) history[(size_t) c * 3 + tap] = values[tap + 1];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_conv_silu launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    // The CUDA launcher passes epsilon/S and 1/sqrt(S) as kernel arguments (native_gdn_preprocess.cu L163); keep
    // both constants on the host so the arithmetic is identical.
    const float eps_scaled = epsilon / (float) S;
    const float scale_after = 1.0f / std::sqrt((float) S);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            // One work-item per row does the whole 128-wide reduction serially (no lane partials, no scratch
            // buffer), the same scratch-free shape native_gr_norm.cpp uses on icpx 2026.1.
            h.parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> rowid) {
                float* row = input + (size_t) rowid * S;
                float partial = 0.0f;
                for (int col = 0; col < S; ++col) partial += row[col] * row[col];
                const float scale = 1.0f / sycl::sqrt(partial / (float) S + eps_scaled);
                for (int col = 0; col < S; ++col) {
                    // Preserve the FP32 store boundary between RMSNorm and ggml_scale.
                    const float normalized = scale * row[col];
                    row[col] = sycl::fma(normalized, scale_after, 0.0f);
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_l2_norm launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (((heads + 255) / 256) * 256)), [=](sycl::id<1> gid) {
                const int i = (int) gid;
                if (i < (int) heads) beta[i] = 1.0f / (1.0f + sycl::exp(-beta[i]));
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_beta_gate launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
                     float* gate, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) (((heads + 255) / 256) * 256)), [=](sycl::id<1> gid) {
                const int i = (int) gid;
                if (i >= (int) heads) return;
                const float value = alpha[i] + dt[i];
                const float softplus = value > 20.0f ? value : sycl::log(1.0f + sycl::exp(value));
                gate[i] = softplus * ssm_a[i];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_gate launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
                         float* destination, int64_t heads, int64_t cols,
                         float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            // One work-item per head reduces the 128 columns serially and writes the whole row - scratch-free.
            h.parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> hid) {
                const size_t offset = (size_t) hid * S;
                const float* in = output + offset;
                float partial = 0.0f;
                for (int col = 0; col < S; ++col) partial += in[col] * in[col];
                const float scale = 1.0f / sycl::sqrt(partial / (float) S + epsilon);
                for (int col = 0; col < S; ++col) {
                    // RMSNorm+gamma is one pinned fused operator, followed by sigmoid*mul.
                    const float weighted = (scale * in[col]) * gamma[col];
                    const float sigmoid = 1.0f / (1.0f + sycl::exp(-z[offset + col]));
                    destination[offset + col] = weighted * sigmoid;
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_gdn_out_norm launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}
}  // namespace strata::kernels
