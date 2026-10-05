// src/kernels/sycl/kv_q8.cpp - SYCL port of src/kernels/cuda/kv_q8.cu (INT8 KV storage for the QSA layers).
//
// The CUDA source runs one block per 64-value group: 64 threads, one value each, the max|x| reduced with a warp
// XOR butterfly and the two warps combined through shared memory.  icpx 2026.1 exposes no subgroup-shuffle
// surface here, so the append becomes ONE work-item per (kv_head, group, K/V): it walks the 64 values serially,
// which is the same arithmetic in a different reduction order (fmax is associative and commutative, so the
// maximum itself is bit-identical).  The gather is a pure per-element map (one thread = four consecutive
// values as a char4) and needs no change at all: one work-item per 4-value run, a flat range<1>.
//
// The scale is quantized against the STORED fp16 scale (`sf = f32_from_f16(sbits)`), so the code the reader
// multiplies by is exactly the code the writer used; `__float2int_rn` is `std::rint` (round-to-nearest-even, the
// device default rounding).  `f16_from_f32`/`f32_from_f16` are the bit-exact converters from f16_bits.hpp.
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

void validate(const QsaShapes& s, const char* what) {
    if (s.head_dim % KV_Q8_GROUP != 0 || s.n_head_kv <= 0 || s.page_size <= 0) {
        std::fprintf(stderr, "kv_q8: %s: head_dim %lld must be a multiple of %d\n", what, (long long) s.head_dim,
                     KV_Q8_GROUP);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    validate(s, "kv_append_q8");
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int groups = head_dim / KV_Q8_GROUP;
    // The pointer members, unpacked so nothing cross-address-space or composite is captured into the lambda.
    int8_t* h_kq = host ? host->k_q : nullptr;
    int8_t* h_vq = host ? host->v_q : nullptr;
    uint16_t* h_kscale = host ? host->k_scale : nullptr;
    uint16_t* h_vscale = host ? host->v_scale : nullptr;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item per 64-value group of one KV head of K (is_v == 0) or V (is_v == 1); 64 serial values.
            h.parallel_for(sycl::range<1>((size_t) kv_heads * (size_t) groups * 2), [=](sycl::id<1> wid) {
                int w = (int) wid;
                const int is_v = w / (kv_heads * groups);
                w -= is_v * kv_heads * groups;
                const int hh = w / groups;
                const int g = w % groups;
                const long long pos = (long long) step[kStepPos];
                const float* cur = is_v ? vcur : kcur;
                const float* src = cur + (size_t) hh * head_dim + g * KV_Q8_GROUP;

                // max |x| over the 64 values; fmaxf is associative so the serial max equals the warp reduce's.
                float amax = 0.0f;
                for (int t = 0; t < KV_Q8_GROUP; ++t) amax = std::fmax(amax, std::fabs(src[t]));
                const uint16_t sbits = f16_from_f32(amax / 127.0f);
                const float sf = f32_from_f16(sbits);   // quantize against the STORED scale

                int8_t* codes = is_v ? v_q : k_q;
                uint16_t* scales = is_v ? v_scale : k_scale;
                const long long page = (long long) page_table[pos / page_size];
                if (page >= 0) {
                    const long long row = (page * kv_heads + hh) * page_size + (pos % page_size);
                    for (int t = 0; t < KV_Q8_GROUP; ++t) {
                        int qv = 0;
                        if (sf > 0.0f) {
                            qv = (int) std::rint(src[t] / sf);   // __float2int_rn: nearest, ties to even
                            qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);
                        }
                        codes[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) qv;
                    }
                    scales[row * groups + g] = sbits;
                }
                if (h_kq != nullptr) {
                    int8_t* hcodes = is_v ? h_vq : h_kq;
                    uint16_t* hscales = is_v ? h_vscale : h_kscale;
                    const long long row = ((pos / page_size) * kv_heads + hh) * page_size + (pos % page_size);
                    for (int t = 0; t < KV_Q8_GROUP; ++t) {
                        int qv = 0;
                        if (sf > 0.0f) {
                            qv = (int) std::rint(src[t] / sf);
                            qv = qv < -127 ? -127 : (qv > 127 ? 127 : qv);
                        }
                        hcodes[row * head_dim + g * KV_Q8_GROUP + t] = (int8_t) qv;
                    }
                    hscales[row * groups + g] = sbits;
                }
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_append_q8 launch: %s\n", e.what());
        std::exit(1);
    }
}

void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    validate(s, "kv_gather_q8");
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int per = head_dim / 4;
    const int groups = head_dim / KV_Q8_GROUP;
    const size_t total = (size_t) max_ids * (size_t) kv_heads * (size_t) per;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item = 4 consecutive values of one cell and head (the CUDA thread's char4 read).
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
                const long long i = (long long) wid;
                const long long n_ids = (long long) step[kStepWidth];
                const long long t_total = n_ids * kv_heads * per;
                if (i >= t_total) return;
                const long long id = i / ((long long) kv_heads * per);
                const int rem = (int) (i % ((long long) kv_heads * per));
                const int hh = rem / per, q4 = rem - hh * per;
                const int cell = ids[id];
                const long long page = (long long) page_table[cell / page_size];
                const long long row = (page * kv_heads + hh) * page_size + (cell % page_size);
                const int d = q4 * 4;
                const float ks = f32_from_f16(k_scale[row * groups + d / KV_Q8_GROUP]);
                const float vs = f32_from_f16(v_scale[row * groups + d / KV_Q8_GROUP]);
                const int8_t* kc = k_q + row * head_dim + q4 * 4;
                const int8_t* vc = v_q + row * head_dim + q4 * 4;
                const long long dst = (id * kv_heads + hh) * (long long) per + q4;
                for (int j = 0; j < 4; ++j) {
                    k_scratch[dst * 4 + j] = f16_from_f32((float) kc[j] * ks);
                    v_scratch[dst * 4 + j] = f16_from_f32((float) vc[j] * vs);
                }
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_gather_q8 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
