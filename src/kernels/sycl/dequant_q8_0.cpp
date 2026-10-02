// src/kernels/sycl/dequant_q8_0.cpp - SYCLomatic port of src/kernels/cuda/quantize_act.cu's dequant_q8_0 (P2.S2).
//
// The inverse of `quantize_q8_0`: each 34-byte block_q8_0 (`{ fp16 d ; int8 qs[32] }`) becomes 32 FP32 elements,
// `out[i] = q * d`.  One work-item per block, a flat range rather than an nd_range - the Arc runtime rejects an
// nd_range whose global size is not an exact multiple of the local size, and n/32 is not (see dequant_s2.cpp).
// `f32_from_f16` is the validated bit converter from f16_bits.hpp; it is called from the lambda so icpx
// compiles it for device, its std::memcpy lowering being a memcpy the target understands.
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace strata::kernels {

namespace {
constexpr int QK8_0 = 32;   // 32 elements per block_q8_0, 34 bytes of storage
}  // namespace

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) nb), [=](sycl::id<1> b) {
            const uint8_t* blk = blocks + b * 34;
            const uint32_t dbits = (uint32_t) blk[0] | ((uint32_t) blk[1] << 8);
            const float d = f32_from_f16((uint16_t) dbits);
            float* out = x + b * QK8_0;
            for (int i = 0; i < QK8_0; ++i) {
                out[i] = (float) (int8_t) (int32_t) blk[2 + i] * d;   // reinterpret the byte as a signed int8
            }
        });
    });
    if (stream == nullptr) q->wait();
}

}  // namespace strata::kernels
