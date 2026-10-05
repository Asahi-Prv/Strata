// src/kernels/sycl/verify_kernels.cpp - SYCL port of src/kernels/cuda/verify_kernels.cu.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
//
// The elementwise / gather / copy kernels keep the CUDA `<<<grid, block>>>` shape as a flat `range<1>` with the
// same bounds check (or the same fixed grid-stride loop for the large copies).  The four kernels that reduced with
// `__shfl_xor_sync` (`gdn_conv_l2_multi`, `gdn_ab_multi`, `gdn_step_norm_multi`, `row_top_prob`) go through the
// SYCL work-group reduction instead: `sycl::nd_range` + `sycl::local_accessor<float,1>` + `sycl::group_barrier`,
// which icpx 2026.1 accepts (see bf16_gemv.cpp).  The lane partials are still summed in lane order, but a warp
// butterfly is a tree so the summation ORDER differs from CUDA - every product is exact and only the order of the
// float additions changes, the same caveat native_gr_norm.cpp records, validated by max_rel.  There is no
// malloc_device scratch anywhere (icpx 2026.1 queues are out-of-order, so a two-submit partials buffer can race;
// see native_bf16.cpp).
#include "strata/kernels/verify_kernels.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

constexpr int kFetchBlocks = 48 * 8;   // fetch_blobs/gather_rows launch geometry from the CUDA launcher
constexpr int kFetchThreads = 256;
constexpr int kChunk = kFetchBlocks * kFetchThreads;

using uint4_t = sycl::vec<std::uint32_t, 4>;

template <typename E>
void gather_rows_launch(const E* src, long long row_e, const int32_t* ids, long long n, E* dst, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) kChunk), [=](sycl::id<1> gid) {
            const long long total = n * row_e;
            for (long long i = (long long) gid; i < total; i += kChunk) {
                const long long r = i / row_e, o = i - r * row_e;
                dst[i] = src[(long long) ids[r] * row_e + o];
            }
        });
    });
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    const long long per = (long long) (blob_bytes / 16);
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) kChunk), [=](sycl::id<1> gid) {
                const long long total = (long long) (*n) * per;
                uint4_t* out = reinterpret_cast<uint4_t*>(dst);
                for (long long i = (long long) gid; i < total; i += kChunk) {
                    const long long k = i / per, off = i - k * per;
                    out[i] = reinterpret_cast<const uint4_t*>(src[k])[off];
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "fetch_blobs launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const unsigned long long b = reinterpret_cast<unsigned long long>(base);
    const unsigned long long bytes = (unsigned long long) blob_bytes;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(128), [=](sycl::id<1> gid) {
                const int k = (int) gid;
                if (k < *n) ptr[k] = b + (unsigned long long) k * bytes;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "rebase_ptrs launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const size_t per_tok = (size_t) n_embd * (size_t) hc;
    const size_t total = (size_t) n_tok * per_tok;
    try {
        q->submit([&](sycl::handler& sh) {
            sh.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                const size_t t = (size_t) gid / per_tok;
                const size_t i = (size_t) gid % per_tok;
                R[t * per_tok + i] = h[t * per_tok + i] + e[t * (size_t) n_embd + i % (size_t) n_embd];
            });
        });
    } catch (const sycl::exception& e2) {
        std::fprintf(stderr, "add_streams_broadcast launch: %s\n", e2.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            sh.parallel_for(sycl::range<1>(1024), [=](sycl::id<1> gid) {
                const int i = (int) gid;
                if (i < n) { slot[i] = ids[i]; dst[i] = i; }
                if (i == 0) *count = n;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "ident_hits launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            sh.parallel_for(sycl::range<1>((size_t) 16 * 256), [=](sycl::id<1> gid) {
                const long long row = *row_dev;
                for (long long i = (long long) gid; i < R_stride; i += 16 * 256)
                    R_dst[i] = R_src[(size_t) row * R_stride + i];
                if (gid == 0) {
                    const int32_t tok = ids[row];
                    *tok_dst = tok;
                    if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
                    if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "mtp_select launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    sycl::queue& qref = *q;
    try {
        if (row_bytes % 16 == 0)
            gather_rows_launch<uint4_t>(reinterpret_cast<const uint4_t*>(src), row_bytes / 16, ids, n,
                                        reinterpret_cast<uint4_t*>(dst), qref);
        else if (row_bytes % 4 == 0)
            gather_rows_launch<uint32_t>(reinterpret_cast<const uint32_t*>(src), row_bytes / 4, ids, n,
                                         reinterpret_cast<uint32_t*>(dst), qref);
        else
            gather_rows_launch<uint8_t>(src, row_bytes, ids, n, dst, qref);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gather_rows launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) qref.wait();
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            sh.parallel_for(sycl::range<1>(64), [=](sycl::id<1> gid) {
                const int i = (int) gid;
                if (i < n) ids[i] = table[ids[i]];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "map_ids launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    if (n_rows <= 0) return;
    const int threads = 1024;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) threads), h);
            h.parallel_for(sycl::nd_range<1>((size_t) n_rows * (size_t) threads, (size_t) threads),
                           [=](sycl::nd_item<1> item) {
                const int t = (int) item.get_group(0);
                const int lane = (int) item.get_local_id(0);
                const float* l = logits + (size_t) t * n_vocab;
                const float m = l[ids[t]];
                float s = 0.0f;
                for (int i = lane; i < n_vocab; i += threads) s += sycl::exp(l[i] - m);
                part[lane] = s;
                sycl::group_barrier(item.get_group());
                if (lane == 0) {
                    float tot = 0.0f;
                    for (int w = 0; w < threads; ++w) tot += part[w];
                    probs[t] = 1.0f / tot;
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "row_top_prob launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    if (n <= 0) return;
    const int per_q = 8 * 256;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) n * per_q), [=](sycl::id<1> gid) {
                const int qq = (int) ((size_t) gid / per_q);
                const int local = (int) ((size_t) gid % per_q);
                int32_t* st = steps + (size_t) qq * 4;
                const int n_kv = st[1];
                const int start = n_kv > window ? n_kv - window : 0;
                const int width = n_kv - start;
                for (int j = local; j < width; j += per_q) ids[(size_t) qq * ids_stride + j] = start + j;
                if (local == 0) st[3] = width;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "window_ids launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(64), [=](sycl::id<1> gid) {
                const int i = (int) gid;
                if (i >= n) return;
                const int c = cells[i];
                steps[i * 4 + 0] = c;
                steps[i * 4 + 1] = c + 1;
                steps[i * 4 + 2] = (c + 1) / 4;
                steps[i * 4 + 3] = c + 1;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "dense_steps launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    const int grid_x = channels / S;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            // One work-group per (channel tile, token) - the CUDA dim3(channels/S, n_tok) grid - with the same
            // 128-wide lane reduction of y*y through a local accessor and a group barrier.
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) S), sh);
            sh.parallel_for(sycl::nd_range<1>((size_t) grid_x * (size_t) n_tok * (size_t) S, (size_t) S),
                            [=](sycl::nd_item<1> item) {
                const int gid = (int) item.get_group(0);
                const int bx = gid % grid_x;
                const int by = gid / grid_x;
                const int col = (int) item.get_local_id(0);
                const int t = t_begin + by;
                const int c = bx * S + col;
                // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
                float win[3];
                for (int j = 0; j < 3; ++j) {
                    const int src = t + j;          // index into [hist(3) | x...]
                    win[j] = src < 3 ? history[(size_t) c * 3 + src] : qkv[(size_t) (src - 3) * channels + c];
                }
                const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * channels + c];
                const float sum = v0 * conv_w[(size_t) c * 4] + v1 * conv_w[(size_t) c * 4 + 1] +
                                  v2 * conv_w[(size_t) c * 4 + 2] + x * conv_w[(size_t) c * 4 + 3];
                float y = sum / (1.0f + sycl::exp(-sum));
                if (bx < qk_heads) {
                    part[col] = y * y;
                    sycl::group_barrier(item.get_group());
                    float ss = 0.0f;
                    for (int l = 0; l < S; ++l) ss += part[l];
                    y *= 1.0f / sycl::sqrt(ss + eps);
                }
                h[(size_t) t * channels + c] = y;
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_conv_l2_multi launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const int nblocks = (channels + 255) / 256;
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) nblocks * 256), [=](sycl::id<1> gid) {
                const int c = (int) gid;
                if (c >= channels) return;
                const int n = *n_keep;
                if (n <= 0) return;
                float seq[3];
                for (int j = 0; j < 3; ++j) {
                    const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
                    seq[j] = src < 3 ? history[(size_t) c * 3 + src] : qkv[(size_t) (src - 3) * channels + c];
                }
                history[(size_t) c * 3] = seq[0];
                history[(size_t) c * 3 + 1] = seq[1];
                history[(size_t) c * 3 + 2] = seq[2];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_conv_commit launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const int rows = 2 * h_v;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            // One work-group (32 lanes) per row - the CUDA one-warp-per-row shape - with the same lane striding of
            // the packed bf16 weight words and the same ordered fmaf chain. The warp-XOR reduce becomes a
            // local-accessor reduce in lane order.
            sycl::local_accessor<float, 1> part(sycl::range<1>(32), sh);
            sh.parallel_for(sycl::nd_range<1>((size_t) rows * 32, 32), [=](sycl::nd_item<1> item) {
                const int row = (int) item.get_group(0);
                const int lane = (int) item.get_local_id(0);
                const bool is_beta = row >= h_v;
                const int r = is_beta ? row - h_v : row;
                const uint16_t* wbase = is_beta ? w_beta : w_alpha;
                const uint32_t* wrow = reinterpret_cast<const uint32_t*>(wbase + (size_t) r * n_embd);
                float acc[kVerifyMaxT];
                for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
                for (int j = lane; j < n_embd / 8; j += 32) {
                    const uint32_t w0 = wrow[4 * j + 0], w1 = wrow[4 * j + 1];
                    const uint32_t w2 = wrow[4 * j + 2], w3 = wrow[4 * j + 3];
                    for (int t = 0; t < n_tok; ++t) {
                        const float* xt = x + (size_t) t * n_embd;
                        const float* xa = xt + (size_t) j * 8;
                        float a = acc[t];
                        a = sycl::fma(f32_from_bf16((uint16_t) (w0 & 0xffffu)), xa[0], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w0 >> 16)), xa[1], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w1 & 0xffffu)), xa[2], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w1 >> 16)), xa[3], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w2 & 0xffffu)), xa[4], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w2 >> 16)), xa[5], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w3 & 0xffffu)), xa[6], a);
                        a = sycl::fma(f32_from_bf16((uint16_t) (w3 >> 16)), xa[7], a);
                        acc[t] = a;
                    }
                }
                for (int t = 0; t < n_tok; ++t) {
                    part[lane] = acc[t];
                    sycl::group_barrier(item.get_group());
                    if (lane == 0) {
                        float a = 0.0f;
                        for (int l = 0; l < 32; ++l) a += part[l];
                        if (is_beta) {
                            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + sycl::exp(-a));
                        } else {
                            const float v = a + dt[r];
                            const float sp = v > 20.0f ? v : sycl::log(1.0f + sycl::exp(v));
                            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
                        }
                    }
                    sycl::group_barrier(item.get_group());
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_ab_multi launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& sh) {
            // One work-group per head, 512 work-items laid out exactly like the CUDA dim3(S, RG): local id
            // rg*S + col.  `sk`/`sq`/`red`/`wsum` are the CUDA shared arrays; the two warp reductions (over col)
            // become local-accessor reductions in lane order, gated by group barriers.
            sycl::local_accessor<float, 1> sk(sycl::range<1>((size_t) S), sh);
            sycl::local_accessor<float, 1> sq(sycl::range<1>((size_t) S), sh);
            sycl::local_accessor<float, 1> red(sycl::range<1>((size_t) RG * S), sh);
            sycl::local_accessor<float, 1> part(sycl::range<1>((size_t) RG * S), sh);
            sycl::local_accessor<float, 1> wsum(sycl::range<1>((size_t) S * RG / 32), sh);
            sh.parallel_for(sycl::nd_range<1>((size_t) h_v * (size_t) S * (size_t) RG, (size_t) S * (size_t) RG),
                            [=](sycl::nd_item<1> item) {
                const int head = (int) item.get_group(0);
                const int tid = (int) item.get_local_id(0);
                const int col = tid % S;
                const int rg = tid / S;
                const int qh = head % h_k;
                const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
                const int value_dim = S * h_v;
                const int n = n_keep ? *n_keep : n_tok;
                float s[RPG];
                float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
                const size_t row_stride = (size_t) h_v * S;
                for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
                for (int t = 0; t < n; ++t) {
                    const float* ht = h + (size_t) t * conv_channels;
                    sycl::group_barrier(item.get_group());   // the previous token is done with sk/sq/red/part
                    if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                    sycl::group_barrier(item.get_group());
                    const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                    float kv = 0.0f;
                    for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                    red[rg * S + col] = kv;
                    sycl::group_barrier(item.get_group());
                    const float kv_col = red[0 * S + col] + red[1 * S + col] + red[2 * S + col] + red[3 * S + col];
                    const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                    float o = 0.0f;
                    for (int r = 0; r < RPG; ++r) {
                        s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                        o = sycl::fma(s[r], sq[rg * RPG + r], o);
                    }
                    sycl::group_barrier(item.get_group());
                    red[rg * S + col] = o;
                    sycl::group_barrier(item.get_group());
                    float oc = 0.0f, sq_part = 0.0f;
                    if (rg == 0) {
                        oc = (red[0 * S + col] + red[1 * S + col] + red[2 * S + col] + red[3 * S + col]) *
                             (1.0f / sycl::sqrt((float) S));
                        sq_part = oc * oc;
                    }
                    if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
                    part[tid] = sq_part;
                    sycl::group_barrier(item.get_group());
                    if ((tid & 31) == 0) {
                        const int wb = (tid >> 5) * 32;
                        float s2 = 0.0f;
                        for (int l = 0; l < 32; ++l) s2 += part[wb + l];
                        wsum[tid >> 5] = s2;
                    }
                    sycl::group_barrier(item.get_group());
                    if (rg == 0) {
                        const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                        const float scale = 1.0f / sycl::sqrt(ss / (float) S + eps);
                        const float zz = z[(size_t) t * value_dim + head * S + col];
                        y[(size_t) t * value_dim + head * S + col] =
                            oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
                    }
                }
                if (n_keep != nullptr && n > 0) {
                    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
                }
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "gdn_step_norm_multi launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

namespace {
void wait_flag_ge_kernel(const uint32_t* flag, uint32_t value, sycl::queue& qref) {
    qref.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            const volatile uint32_t* vf = flag;
            while (*vf < value) { }
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        });
    });
}
}  // namespace

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        wait_flag_ge_kernel(flag, value, *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "wait_flag_ge launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    if (n <= 0 || n_tok <= 0) return;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const size_t total = (size_t) n_tok * (size_t) n;
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                const size_t t = (size_t) gid / (size_t) n;
                const size_t i = (size_t) gid % (size_t) n;
                const unsigned long long token = (unsigned long long) tokens[t];
                const uint8_t* c = codes + (size_t) token * (size_t) row_codes;
                const float* sc = scales + (size_t) token * (size_t) row_groups;
                const float* of = offsets ? offsets + (size_t) token * (size_t) row_groups : nullptr;
                const int per_byte = 8 / code_bits;
                const unsigned mask = (1u << code_bits) - 1u;
                const int code = (c[i / per_byte] >> ((int) (i % per_byte) * code_bits)) & mask;
                const int64_t group = (int64_t) i / group_elems;
                const float product = (float) (code + code_bias) * sc[group];
                out[(size_t) t * n + i] = product + (of ? of[group] : 0.0f);
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "embedding_gather_dev launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    const size_t per_tok = (size_t) n_embd * (size_t) hc;
    const size_t total = (size_t) n_tok * per_tok;
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> gid) {
                const size_t t = (size_t) gid / per_tok;
                const size_t i = (size_t) gid % per_tok;
                R[t * per_tok + i] = x[t * (size_t) n_embd + i % (size_t) n_embd];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "broadcast_streams launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    if (n <= 0) return;
    const unsigned blocks = (unsigned) (((n + 255) / 256) < 64 ? ((n + 255) / 256) : 64);
    const long long chunk = (long long) blocks * 256;
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    try {
        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t) chunk), [=](sycl::id<1> gid) {
                const int idx = *index;
                if (idx < 0) return;
                for (long long i = (long long) gid; i < n; i += chunk) dst[i] = src[(size_t) idx * stride + i];
            });
        });
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "copy_indexed launch: %s\n", e.what());
        std::exit(1);
    }
    if (stream == nullptr) q->wait();
}

}  // namespace strata::kernels
