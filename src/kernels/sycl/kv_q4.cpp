// src/kernels/sycl/kv_q4.cpp - SYCL port of src/kernels/cuda/kv_q4.cu.  Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
//
// THE HADAMARD.  The CUDA `fwht256_kernel` runs one warp per row with the low 5 index bits across lanes
// (`__shfl_xor_sync`) and the high 3 across each lane's registers.  Join those two stages in index order and the
// whole thing is the standard in-place FWHT butterfly `for (len = 1; len < 256; len <<= 1)`, because the
// butterflies of distinct index bits commute (H_256 is a tensor product) - so one work-item per row, 256 floats
// in registers, reproduces the transform bit-for-bit.  The 1/16 (orthonormal) scale is applied on load as in the
// source.
//
// THE GROUP.  The CUDA `q4_group` reduces max|x| over a 32-lane warp while tie-breaking to the LARGER SIGNED
// VALUE so every lane picks the same `d`.  That is the maximum under lexicographic (|x|, x) order, which is
// associative and commutative, so one work-item per 32-value group walking it serially yields the SAME scale
// (and therefore the SAME codes) in every lane, with no shuffle.  `__float2int_rz` is a plain C++ truncation to
// int.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

// One 32-value group: ggml's q4_0, d = (the value of largest |x|) / -8, q = clamp(trunc(x/d + 8.5), 0, 15).
// Ties in |x| resolve to the larger value, so the maximum is a total order and the serial walk agrees with the
// warp XOR reduce.  Returns the scale bits; `bytes[t]` (t < 16) is element t in the low nibble, t+16 in the high.
inline uint16_t q4_group_dev(const float* x, uint8_t* bytes) {
    float amax = std::fabs(x[0]), mval = x[0];
    for (int t = 1; t < QK4_0; ++t) {
        const float a = std::fabs(x[t]), v = x[t];
        if (a > amax || (a == amax && v > mval)) {
            amax = a;
            mval = v;
        }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t qc[QK4_0];
    for (int t = 0; t < QK4_0; ++t) {
        const int q = (int) (x[t] * id + 8.5f);   // __float2int_rz: truncate toward zero
        qc[t] = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    }
    for (int t = 0; t < 16; ++t) bytes[t] = (uint8_t) (qc[t] | (qc[t + 16] << 4));
    return f16_from_f32(d);
}

inline void q4_store_dev(uint8_t* pool, long long row, int b, int blocks_per_head, uint16_t d, const uint8_t* bytes) {
    block_q4_0* blk =
        reinterpret_cast<block_q4_0*>(pool + row * (long long) blocks_per_head * (long long) sizeof(block_q4_0)) + b;
    blk->d = d;
    for (int t = 0; t < 16; ++t) blk->qs[t] = bytes[t];
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item per row: the standard in-order in-place FWHT, which the lane split plus register
            // stages of the CUDA kernel are exactly a reordering of.
            h.parallel_for(sycl::range<1>((size_t) n_rows), [=](sycl::id<1> rid) {
                const long long r = (long long) rid;
                const float* s = src + r * 256;
                float* d = dst + r * 256;
                float reg[256];
                for (int i = 0; i < 256; ++i) reg[i] = s[i] * (1.0f / 16.0f);
                for (int len = 1; len < 256; len <<= 1) {
                    for (int i = 0; i < 256; i += 2 * len) {
                        for (int j = 0; j < len; ++j) {
                            const float a = reg[i + j], b = reg[i + j + len];
                            reg[i + j] = a + b;
                            reg[i + j + len] = a - b;
                        }
                    }
                }
                for (int i = 0; i < 256; ++i) d[i] = reg[i];
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fwht256 launch: %s\n", e.what());
        std::exit(1);
    }
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    need_256(s, "kv_append_q4");
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int blocks_per_head = head_dim / QK4_0;
    uint8_t* h_kq4 = host ? host->k_q4 : nullptr;
    uint8_t* h_vq4 = host ? host->v_q4 : nullptr;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item = one 32-value group of one KV head of K (is_v == 0) or V (is_v == 1).
            h.parallel_for(sycl::range<1>((size_t) kv_heads * (size_t) blocks_per_head * 2), [=](sycl::id<1> wid) {
                int w = (int) wid;
                const int is_v = w / (kv_heads * blocks_per_head);
                w -= is_v * kv_heads * blocks_per_head;
                const int hh = w / blocks_per_head;
                const int b = w % blocks_per_head;
                const long long pos = (long long) step[kStepPos];
                const float* cur = is_v ? vcur : kcur;
                float x[QK4_0];
                for (int t = 0; t < QK4_0; ++t) x[t] = cur[hh * head_dim + b * QK4_0 + t];
                uint8_t bytes[16];
                const uint16_t d = q4_group_dev(x, bytes);
                const long long page = (long long) page_table[pos / page_size];
                if (page >= 0)
                    q4_store_dev(is_v ? v_q4 : k_q4, (page * kv_heads + hh) * page_size + (pos % page_size), b,
                                 blocks_per_head, d, bytes);
                if (h_kq4 != nullptr)
                    q4_store_dev(is_v ? h_vq4 : h_kq4,
                                 ((pos / page_size) * kv_heads + hh) * page_size + (pos % page_size), b,
                                 blocks_per_head, d, bytes);
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_append_q4 launch: %s\n", e.what());
        std::exit(1);
    }
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int blocks_per_head = head_dim / QK4_0;
    uint8_t* h_kq4 = host ? host->k_q4 : nullptr;
    uint8_t* h_vq4 = host ? host->v_q4 : nullptr;
    uint8_t* s_kq4 = stage ? stage->k_q4 : nullptr;
    uint8_t* s_vq4 = stage ? stage->v_q4 : nullptr;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        // Two ordered submits, one per K/V, exactly as the CUDA launch loop; they write disjoint buffers.
        for (int is_v = 0; is_v < 2; ++is_v) {
            const int iv = is_v;
            qp->submit([&, iv](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t) T * (size_t) kv_heads * (size_t) blocks_per_head),
                               [=](sycl::id<1> wid) {
                                   const long long w = (long long) wid;
                                   const long long t = w % T;
                                   const long long rest = w / T;
                                   const int hh = (int) (rest % kv_heads);
                                   const int b = (int) (rest / kv_heads);
                                   const long long pos = pos0 + t;
                                   const float* src = (iv ? V : K) + t * ((long long) kv_heads * head_dim) +
                                                      (long long) hh * head_dim + b * QK4_0;
                                   float x[QK4_0];
                                   for (int i = 0; i < QK4_0; ++i) x[i] = src[i];
                                   uint8_t bytes[16];
                                   const uint16_t d = q4_group_dev(x, bytes);
                                   const long long page = (long long) page_table[pos / page_size];
                                   const long long row_id =
                                       ((pos / page_size) * kv_heads + hh) * page_size + (pos % page_size);
                                   if (page >= 0)
                                       q4_store_dev(iv ? v_q4 : k_q4, (page * kv_heads + hh) * page_size + (pos % page_size),
                                                    b, blocks_per_head, d, bytes);
                                   if (h_kq4 != nullptr)
                                       q4_store_dev(iv ? h_vq4 : h_kq4, row_id, b, blocks_per_head, d, bytes);
                                   if (s_kq4 != nullptr)
                                       q4_store_dev(iv ? s_vq4 : s_kq4, row_id, b, blocks_per_head, d, bytes);
                               });
            });
        }
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_append_q4 batch launch: %s\n", e.what());
        std::exit(1);
    }
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int kv_heads = (int) s.n_head_kv;
    const int head_dim = (int) s.head_dim;
    const int page_size = (int) s.page_size;
    const int blocks_per_head = head_dim / QK4_0;
    const int bytes_per_head = blocks_per_head * (int) sizeof(block_q4_0);
    const size_t total = (size_t) max_ids * (size_t) kv_heads * (size_t) blocks_per_head;
    sycl::queue* qp = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        qp->submit([&](sycl::handler& h) {
            // One work-item = one (id, kv_head, 32-value block): decode the whole block into the fp16 scratch.
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> wid) {
                const long long i = (long long) wid;
                const long long n_ids = (long long) step[kStepWidth];
                const long long t_total = n_ids * kv_heads * blocks_per_head;
                if (i >= t_total) return;
                const long long id = i / ((long long) kv_heads * blocks_per_head);
                const int rem = (int) (i % ((long long) kv_heads * blocks_per_head));
                const int hh = rem / blocks_per_head;
                const int b = rem % blocks_per_head;
                const int cell = ids[id];
                const long long page = (long long) page_table[cell / page_size];
                const long long row = (page * kv_heads + hh) * page_size + (cell % page_size);
                const block_q4_0* k_blk =
                    reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
                const block_q4_0* v_blk =
                    reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
                const float kd = f32_from_f16(k_blk->d);
                const float vd = f32_from_f16(v_blk->d);
                for (int t = 0; t < QK4_0; ++t) {
                    const int j = t < 16 ? t : (t - 16);
                    const uint8_t k_byte = k_blk->qs[j];
                    const uint8_t v_byte = v_blk->qs[j];
                    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
                    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
                    const long long dst_offset = ((id * kv_heads + hh) * head_dim) + (b * QK4_0 + t);
                    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
                    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
                }
            });
        });
        if (stream == nullptr) qp->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "kv_gather_q4 launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
