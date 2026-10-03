// src/kernels/sycl/native_qsa_indexer.cpp - SYCL port of src/kernels/cuda/native_qsa_indexer.cu (qwen4exp, one block of
// THREADS over IDX_DIM=128).  A single append is <<<1, 256>>> on 128 elements: SET_ROWS stores fp16-rounded raw to the
// (idx_block-1)=3 tail slots, the spare slot pools the four raw-key slices (mean), an RMSNorm of the pooled key with a
// warp+block reduce, then an IMRoPE rotary over the first n_rot=64 dims.  icpx 2026.1 has no group-reduction surface, so
// the single block is executed by ONE work-item that walks the 128 elements serially - the same work, a different
// reduction order (validated by max_rel).  `mrope_pos` is inlined (mrope.hpp's #else branch) for the same reason as
// native_rope.cpp.
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace strata::kernels {

namespace {
constexpr int D = 128, R = 4, ROT = 64;
std::atomic<bool> enabled{false};

struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
}  // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s,
                               int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || max_cells < 1 || max_cells > INT32_MAX ||
        pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) ||
        epsilon <= 0.0f || !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument(
            "native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw, (size_t) D * 4}, {relative_pos_device, 4}, {gamma, (size_t) D * 4},
                          {b.tail, (size_t) (R - 1) * D * 4}, {b.dead, (size_t) D * 4},
                          {b.pooled, (size_t)(max_cells / R + 1) * D * 4}, {b.block_pos, 4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i)
        for (int j = i + 1; j < 7; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = sycl::pow(freq_base, -2.0f / ROT);
    const int32_t* mtab = mrope_table();
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                // The kernel is ONE work-item walking all D elements serially, so the pooled key lives on the work-item
                // stack instead of a malloc_device/free pair: on icpx 2026.1 / Arc the device-side free executes on
                // the host before the kernel finishes and, when stream != nullptr the function does not wait, so each
                // per-call free leaked until the queue was drained - a few appends ran OUT_OF_DEVICE_MEMORY. A
                // scratch-free submission is self-contained, exactly like native_gr_rms_norm_weighted.
                float vals[D];
                const int pos = *relative_pos_device;
                if (pos < 0 || pos >= max_cells) return;
                const int slot = pos % R;
                if (slot < R - 1) {
                    for (int d = 0; d < D; ++d) b.tail[(size_t) slot * D + d] = f32_from_f16(f16_from_f32(raw[d]));
                }
                if (pos != 0 && slot != R - 1) return;
                float mean = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float incoming = f32_from_f16(f16_from_f32(raw[d]));
                    const float sum = (pos == 0) ? (incoming + incoming + incoming + incoming)
                                                 : (b.tail[(size_t) 0 * D + d] + b.tail[(size_t) 1 * D + d] +
                                                    b.tail[(size_t) 2 * D + d] + incoming);
                    mean = 0.25f * sum;
                    vals[d] = mean;
                }
                float square_sum = 0.0f;
                for (int d = 0; d < D; ++d) square_sum += vals[d] * vals[d];
                const float scale = 1.0f / sycl::sqrt(square_sum / D + epsilon);
                for (int d = 0; d < D; ++d) vals[d] = scale * vals[d] * gamma[d];
                const int bidx = pos / R;
                const int rope_pos = pos == 0 ? 0 : pos_base + R * bidx;
                for (int d = 0; d < D; ++d) {
                    float y = vals[d];
                    if (d < ROT) {
                        const int pair = d % (ROT / 2);
                        const int mpos = (pos == 0) ? 0
                                                    : (mtab ? mtab[(size_t) rope_pos * 3 + pair % 3] : rope_pos);
                        const float theta = mpos * sycl::pow(theta_scale, (float) pair);
                        const float c = sycl::cos(theta), s = sycl::sin(theta);
                        const float a = vals[pair], z = vals[pair + ROT / 2];
                        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
                    }
                    b.pooled[(size_t) bidx * D + d] = y;
                    if (pos == 0) b.dead[d] = y;
                    else b.pooled[(size_t)(bidx + 1) * D + d] = b.dead[d];
                    if (d == 0 && pos != 0) *b.block_pos = rope_pos;
                    }
                });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "native_qsa_indexer_append launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
