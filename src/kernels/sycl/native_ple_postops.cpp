// src/kernels/sycl/native_ple_postops.cpp - SYCL port of src/kernels/cuda/native_ple_postops.cu (qwen4exp build_ple).
//
// Single-token PLE block. Three of the six stages are the same weighted RMSNorm the engine already has as
// `native_gr_rms_norm_weighted`, so those are called directly. The other three are tiny: `gate_kernel` folds the
// key x query dot product (SUM_ROWS over N=2560 with eight partial lanes), scales it by 1/sqrt(N), takes a signed
// square root then a sigmoid; `broadcast_kernel` multiplies the n_embd value by each of the four stream gates; and
// `conv_residual_kernel` is a dilation-3, kernel-4 conv over the row-fastest history rows plus a silu into the
// hidden+gated residual. icpx 2026.1 has no group-reduction surface, so the gate dot is one work-item per stream
// walking all 2560 dims serially - the same fp32 multiply-add the oracle materializes.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
struct Span { const void* p; std::size_t bytes; std::size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<std::uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<std::uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden,
                        const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key, D * 4, 4}, {hidden, D * 4, 4}, {value, N * 4, 4},
                           {history, HISTORY * D * 4, 4}, {w.norm_key, D * 4, 4}, {w.norm_query, D * 4, 4},
                           {w.norm_conv, D * 4, 4}, {w.conv1d_f16, 4 * D * 2, 2}};
    const Span outputs[] = {{b.key, D * 4, 4}, {b.query, D * 4, 4}, {b.gate, H * 4, 4}, {b.gated, D * 4, 4},
                            {b.normalized, D * 4, 4}, {b.conv, D * 4, 4}, {b.result, D * 4, 4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    // Capture all caller-owned output pointers into locals so the device lambdas below never capture the `b`
    // struct reference. On icpx 2026.1 / Arc, a lambda that captures the `b` reference (via [=]) reads stale
    // pointers from the freed native_gr `partials` scratch in a later submission; local pointer copies avoid it.
    const float* p_key = b.key;
    const float* p_qry = b.query;
    float* p_gate = b.gate;
    float* p_gated = b.gated;
    float* p_norm = b.normalized;
    float* p_conv = b.conv;
    float* p_res = b.result;
    const float* value_local = value;
    const float* history_local = history;
    const uint16_t* cw_local = w.conv1d_f16;
    const float* hidden_local = hidden;
    // grouped_norm(projected_key) -> key, grouped_norm(hidden) -> query (temporary, may alias normalized).
    native_gr_rms_norm_weighted(projected_key, w.norm_key, b.key, N, H, NG_RMS_EPS, q);
    native_gr_rms_norm_weighted(hidden, w.norm_query, b.query, N, H, NG_RMS_EPS, q);
    // icpx 2026.1 on Arc does not reliably order a same-queue kernel read against the prior kernel's write into a
    // malloc_device buffer (worst with reused addresses), so the gate below can read stale key/query. Flush the two
    // norms before the dependent kernels launch; the same rule applies to the norm3 output before conv below.
    q->wait();
    // gate = sigmoid(sign(sqrt(max(|s|,eps))) * s) with s = (key.query)/sqrt(N).
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)H), [=](sycl::id<1> row) {
            float sum = 0.0f;
            const float* kr = p_key + (size_t)row * N;
            const float* qr = p_qry + (size_t)row * N;
            for (int d = 0; d < N; ++d) sum += kr[d] * qr[d];
            const float scale = 1.0f / sycl::sqrt((float)N);
            const float s = scale * sum;
            const float mag = sycl::sqrt(std::fmax(std::fabs(s), 1e-6f));
            const float sign = (s > 0.0f) - (s < 0.0f);
            p_gate[row] = 1.0f / (1.0f + sycl::exp(-sign * mag));
        });
    });
    // The broadcast reads p_gate, so flush the gate before it launches (same out-of-order ordering rule as above).
    q->wait();
    // broadcast: gated[c] = value[c % N] * gate[c / N].
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)D), [=](sycl::id<1> c) {
            p_gated[c] = value_local[c % N] * p_gate[c / N];
        });
    });
    // The conv input norm reads p_gated, so flush the broadcast before it launches.
    q->wait();
    // grouped_norm(gated) -> normalized (the conv input).
    native_gr_rms_norm_weighted(b.gated, w.norm_conv, b.normalized, N, H, NG_RMS_EPS, q);
    q->wait();
    // dilation-3, kernel-4 conv over the row-fastest history rows; silu into hidden + gated + conv.
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t)D), [=](sycl::id<1> c) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const float x = (k == 3) ? p_norm[c] : history_local[(size_t)c * HISTORY + 3 * k];
                const float wt = f32_from_f16(cw_local[(size_t)c * 4 + k]);
                const float term = x * wt;
                sum = (k == 0) ? term : sum + term;
            }
            const float activation = sum / (1.0f + sycl::exp(-sum));
            p_conv[c] = activation;
            p_res[c] = hidden_local[c] + p_gated[c] + activation;
        });
    });
    if (stream == nullptr) q->wait();
}

}  // namespace strata::kernels
