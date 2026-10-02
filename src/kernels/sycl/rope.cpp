// src/kernels/sycl/rope.cpp - SYCLomatic port of src/kernels/cuda/rope.cu (P2.S2, P3).
//
// The NEUX partial RoPE kernel.  This is the EASIEST shape to port: one thread per row, no warp reduction, no
// shared memory, no __syncthreads - just a bounds check and a rotation.  SYCLomatic turns `blockIdx * blockDim
// + threadIdx` into `get_global_id`, the `<<<grid, block>>>` launch into a `parallel_for` over range<1>(rows),
// and the stream argument into a sycl::queue.  The cos/sin table is still built on the HOST in float64 (see
// rope.cu for why), so build_rope_table is unchanged - only the launch side moves.
//
// Compiled with -fsycl by the SYCL build.  It names the CUDA runtime nowhere: the launch goes through the queue
// in the `void* stream` wire format (see strata/core/device.hpp).
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

// UNCHANGED from rope.cu - host-side, no kernel, so no backend is involved.
void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

namespace {

// The launch config: 128 threads per block on CUDA.  SYCL needs no block size for a flat range(rows) kernel,
// so the 128 disappears and the launch is just parallel_for over one work-item per row.
constexpr uint32_t kThreads = 128;

}  // namespace

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    // The null stream (CUDA's default) is the SYCL default queue; non-null is a queue we were handed.
    const bool on_default = (stream == nullptr);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());

    const int32_t* mtab = mrope_table();
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> r) {
                const int half = n_rot / 2;
                const float* xr = x + static_cast<size_t>(r) * head_dim;
                float* orow = out + static_cast<size_t>(r) * head_dim;

                for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];   // the PARTIAL rotation's untouched tail
                for (int i = 0; i < half; ++i) {
                    const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;  // the image path's per-pair position
                    rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
                }
            });
        });
        // CUDA only syncs the null stream (a non-default stream is ordered by the caller); SYCL matches by
        // waiting only on the default queue.  submit() throws on a launch error the way cudaGetLastError did.
        if (on_default) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "rope_neox_apply launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
