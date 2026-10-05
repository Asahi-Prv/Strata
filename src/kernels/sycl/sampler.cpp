// src/kernels/sycl/sampler.cpp - SYCL port of src/kernels/cuda/sampler.cu (P2.S2: the sampler chain).
//
//     penalties -> top_k -> min_p -> top_p -> temperature -> penalties -> pick
//
// THE ORDER IS THE WHOLE CONTENT OF THIS FILE.  `docs/sampling.md` transcribes it from llama.cpp's own chain
// (`common/sampling.cpp` L357/360/375/381/399) and the two facts that are easy to get backwards are that
// TEMPERATURE COMES AFTER THE TRUNCATION FILTERS and PENALTIES COME AFTER TEMPERATURE.  The intuitive order -
// scale first, then truncate, with penalties as pre-processing - is a different distribution.  Both produce a
// valid token, so only a comparison at the distribution level can tell them apart; the parity test does that
// explicitly by running the wrong order and requiring it to differ.
//
// Both kernels put ONE BLOCK per token over the vocabulary: `sampler_greedy_kernel` is the plain argmax,
// `sampler_kernel` runs the sampled chain as `top_k` block-argmax rounds followed by the top_p / temperature /
// draw chain (its header says why the selection must be parallel and why the tie rule keeps the semantics).
//
// THE SYCL PORT.  icpx 2026.1 exposes no warp-shuffle or block-reduction surface here, so the block-per-token
// reductions of the CUDA source become ONE WORK-ITEM PER TOKEN walking the vocabulary serially - the same shape
// `native_gr_norm` uses.  The CUDA source's own header insists "THE SEMANTICS ARE THE SERIAL ONES, EXACTLY": the
// each-round argmax resolves ties to the LOWEST index (the serial scan's strict `>` keeps the first maximum it
// meets), the top-p cut reads the descending selection order, and temperature and the Philox draw apply after
// the cut.  This port is therefore bit-for-bit the serial reference.  The shared-memory penalty BITMAP is not
// reproduced: it existed only to make the per-candidate `history_count` cheap, and the counts (and therefore
// every sampled value) are exactly what the per-candidate scan produces, so a direct scan is used.
//
// The generator stays the counter-based Philox 4x32-10 of the CUDA source (NOT curand): it is pure integer
// arithmetic, so the stream is a reproducible function of (seed, position) rather than of how many draws came
// before.  `__umulhi` is replaced by the portable 32x32->high-32 multiply, which is what it computes.
#include "strata/kernels/sampler.hpp"

#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// ---------------------------------------------------------------------------------------------------------
// The device helpers.  They take individual scalars rather than the `SamplerParams` struct: on icpx 2026.1 /
// Arc a kernel closure that captures a composite struct by value can fail with UR_RESULT_ERROR_OUT_OF_RESOURCES,
// and capturing a 64-bit scalar by value can fail with UR_RESULT_ERROR_DEVICE_LOST (see native_qsa_score.cpp),
// so the launch below captures only 32-bit ints/floats and pointer locals and reconstructs the seed/counter
// from their 32-bit halves.
// ---------------------------------------------------------------------------------------------------------

// `__umulhi`: the high 32 bits of the unsigned 32x32 product, as a 64-bit multiply shifted down.
inline uint32_t umulhi(uint32_t a, uint32_t b) {
    return (uint32_t) (((uint64_t) a * (uint64_t) b) >> 32);
}

// Philox 4x32-10, the counter-based generator the phase asks for.  Counter-based matters because it makes the
// stream a function of (seed, position) rather than of how many draws came before - so a batch can be sampled
// in any order and a run is reproducible.
inline uint32_t philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0,
                                 uint32_t k1) {
    const uint32_t hi0 = umulhi(0x9E3779B9u, c0);
    const uint32_t hi1 = umulhi(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0; c1 = n1; c2 = n2; c3 = n3;
    return 0;
}

inline float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    }
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// `count_in_history` and the penalty application, transcribed from `llama_sampler_penalties_apply`.
// The repeat penalty MULTIPLIES for non-positive logits and DIVIDES for positive ones - dividing
// unconditionally is the natural reading of the source paper and it INVERTS the penalty on half the
// vocabulary.  The presence penalty is `float(count > 0)`, a boolean, not the count.
inline int history_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i) if (h[i] == v) ++c;
    return c;
}

inline float apply_penalties(float logit, int count, float repeat, float freq, float present) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= repeat;
    else               logit /= repeat;
    logit -= (float) count * freq + (count > 0 ? 1.0f : 0.0f) * present;
    return logit;
}

/// **THE GREEDY ARGMAX, ONE WORK-ITEM PER TOKEN, COVERING THE VOCABULARY.**
///
/// **WHY THIS IS A SEPARATE KERNEL AND NOT A BRANCH.**  `sampler_kernel` is launched as a grid over TOKENS
/// with 64 threads and a `if (t >= n_tokens) return;` at the top.  The decode path has `n_tokens == 1`, so
/// that launch was `<<<1, 64>>>`, 63 threads exited on the first line, and ONE THREAD walked all 248,320
/// logits in a dependent loop on one SM of 48.  Measured in isolation (`bench/micro/sampler_cost.cu`):
/// **3.11 ms per token**, 5.7% of an ~54 ms token, and the whole of round 309's `sample` phase - the two
/// synchronisations around it are 0.03 ms each.
///
/// The serial scan is the semantics here: it walks `v` ascending with `if (s > bv)`, so the LOWEST index wins
/// a tie.  The port keeps that rule exactly (the CUDA parallel kernel's reduction existed only to reintroduce
/// it), so `sampler_parity` and C1 see no change.
void sampler_greedy_run(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                        int penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present,
                        int* out, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> tid) {
            const int t = (int) tid;
            const float* l = logits + (size_t) t * n_vocab;
            const int* hrow = history ? history + (size_t) t * history_len : nullptr;
            int hlen = 0;
            if (hrow) {
                hlen = penalty_last_n < history_len ? penalty_last_n : history_len;
                if (hlen < 0) hlen = 0;
                hrow += history_len - hlen;          // the window is the TAIL
            }

            // `n_vocab` is the "no candidate" index: it loses every comparison to a real one, so a work-item
            // with no elements contributes nothing rather than contributing a bogus zero.
            float bv = -INFINITY;
            int best = n_vocab;
            for (int v = 0; v < n_vocab; ++v) {
                const int count = hrow ? history_count(hrow, hlen, v) : 0;
                const float s = apply_penalties(l[v], count, penalty_repeat, penalty_freq, penalty_present);
                if (s > bv) { bv = s; best = v; }
            }
            // A tie between two `-inf` candidates leaves `best == n_vocab`, and the serial version answered 0.
            out[t] = (best < n_vocab) ? best : 0;
        });
    });
}

/// **THE SAMPLED PATH, ONE WORK-ITEM PER TOKEN.**  The kernel below replaced a version that ran the whole chain
/// in ONE THREAD per token (`<<<ceil(T/64), 64>>>`, so a 4-token window fielded four threads): `top_k` alone
/// was `k` sequential scans of the vocabulary with an inner sweep over the already-taken list - 20 x 248,320
/// iterations of dependent work on one SM - and a verify window measured **1.6 s in the sampler**, which made
/// every temperature-bearing request ~30x slower than a greedy one.  The CUDA source turned the selection into
/// `k` block argmaxes; the SYCL port returns it to the serial k-pass form, which is the order the CUDA
/// reduction was designed to reproduce exactly.
///
/// THE SEMANTICS ARE THE SERIAL ONES, EXACTLY.  The kept sequence - both its set and its order - is unchanged;
/// `top_p`'s cut reads that order in double arithmetic as before; temperature and the Philox draw apply after
/// the cut.  `sampler_parity` pins all of it against the host reference.
void sampler_run(const float* logits, int n_vocab, int n_tokens, const int* history, int history_len,
                 int top_k, float top_p, float min_p, float temperature, int min_keep, int penalty_last_n,
                 float penalty_repeat, float penalty_freq, float penalty_present, uint64_t seed,
                 uint64_t counter, int* out, sycl::queue* q) {
    q->submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> tid) {
            const int t = (int) tid;
            const float* l = logits + (size_t) t * n_vocab;

            // Temperature is needed by BOTH stages below, so it is computed here; the chain still APPLIES it
            // after the truncation filters - the survivors are chosen on the raw logits and only then scaled.
            const float inv_t = temperature > 0.0f ? 1.0f / temperature : 0.0f;

            // The penalty window is the last `penalty_last_n` entries of this row's history.
            const int* hrow = history ? history + (size_t) t * history_len : nullptr;
            int hlen = 0;
            if (hrow) {
                hlen = penalty_last_n < history_len ? penalty_last_n : history_len;
                if (hlen < 0) hlen = 0;
                hrow += history_len - hlen;          // the window is the TAIL
            }
            auto hit_count = [&](int v) -> int { return hrow ? history_count(hrow, hlen, v) : 0; };

            const int KMAX = 64;
            int k = top_k > 0 ? (top_k < KMAX ? top_k : KMAX) : 0;
            if (k <= 0) {
                // the CUDA kernel printed and left `out[t]` unwritten; the host wrapper refuses this case
                // before launching, so this is only a defence if called directly.
                return;
            }

            // ---- top_k: k rounds of argmax over the not-yet-taken.  `sel_*` holds the kept ids and their raw
            // logits in selection order: descending by value, ties to the lower index, which is the order the
            // top_p cut below is defined over.
            int sel_ids[KMAX];
            float sel_logit[KMAX];
            for (int i = 0; i < k; ++i) {
                // `n_vocab` is the "no candidate" index: it loses every comparison to a real one (same
                // convention as the greedy kernel, whose tie rule this selection shares).
                float bv = -INFINITY;
                int best = n_vocab;
                for (int v = 0; v < n_vocab; ++v) {
                    bool taken = false;
                    for (int j = 0; j < i; ++j) if (sel_ids[j] == v) { taken = true; break; }
                    if (taken) continue;
                    const float s = apply_penalties(l[v], hit_count(v), penalty_repeat, penalty_freq,
                                                    penalty_present);
                    if (s > bv) { bv = s; best = v; }
                }
                sel_ids[i] = (best < n_vocab) ? best : 0;
                sel_logit[i] = bv;
            }

            // ---- min_p: keep the descending prefix whose probability is at least `min_p` of the top token's.
            // The kept list is in selection order (descending), so the survivors are a PREFIX and the cut
            // composes with top_p's below.  In logit space the threshold is `sel_logit[0] + logf(min_p)` -
            // equivalent to `p >= min_p * p_max` without the overflow an exp of raw logits risks.  0 disables.
            int n_minp = k;
            if (min_p > 0.0f) {
                const float thresh = sel_logit[0] + sycl::log(min_p);
                for (int i = 0; i < k; ++i)
                    if (sel_logit[i] < thresh) { n_minp = i; break; }
            }

            // ---- top_p over the survivors, in descending order (which the selection produced), then
            // temperature and one Philox draw.  The arithmetic is the serial kernel's, instruction for
            // instruction.
            int n_keep = n_minp;
            float mx = sel_logit[0];
            for (int i = 1; i < n_minp; ++i) mx = sycl::fmax(mx, sel_logit[i]);
            if (top_p < 1.0f) {
                double sum = 0.0;
                for (int i = 0; i < n_minp; ++i) sum += sycl::exp((double) sel_logit[i] - (double) mx);
                double cum = 0.0;
                int cut = n_minp;
                for (int i = 0; i < n_minp; ++i) {
                    cum += sycl::exp((double) sel_logit[i] - (double) mx) / sum;
                    if (cum >= (double) top_p) { cut = i + 1; break; }
                }
                if (cut < min_keep) cut = min_keep < n_minp ? min_keep : n_minp;
                n_keep = cut;
            }
            auto scaled = [&](int i) {
                return apply_penalties(sel_logit[i] * inv_t, hit_count(sel_ids[i]), penalty_repeat,
                                       penalty_freq, penalty_present);
            };
            float smx = scaled(0);
            for (int i = 1; i < n_keep; ++i) smx = sycl::fmax(smx, scaled(i));
            double sum = 0.0;
            for (int i = 0; i < n_keep; ++i) sum += sycl::exp((double) scaled(i) - (double) smx);
            const float u = philox_uniform(seed, counter + (uint64_t) t);
            double cum = 0.0;
            int pick = sel_ids[n_keep - 1];
            for (int i = 0; i < n_keep; ++i) {
                cum += sycl::exp((double) scaled(i) - (double) smx) / sum;
                if ((double) u < cum) { pick = sel_ids[i]; break; }
            }
            out[t] = pick;
        });
    });
}

}  // namespace

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    sycl::queue* q = static_cast<sycl::queue*>(stream ? stream : strata::core::default_sycl_queue());
    // Unpack the parameter struct into scalars for the closure - see the note above on composite/64-bit
    // captures on Arc.  `seed`/`counter` are passed as their 32-bit halves.
    const float penalty_repeat = p.penalty_repeat, penalty_freq = p.penalty_freq,
                penalty_present = p.penalty_present;
    const uint32_t seed_lo = (uint32_t) p.seed, seed_hi = (uint32_t) (p.seed >> 32);
    const uint32_t counter_lo = (uint32_t) p.counter, counter_hi = (uint32_t) (p.counter >> 32);
    try {
        if (p.greedy || p.temperature <= 0.0f) {
            // One work-item per token over the vocabulary.  See `sampler_greedy_run`.
            sampler_greedy_run(logits, n_tokens, n_vocab, history, history_len, p.penalty_last_n, penalty_repeat,
                               penalty_freq, penalty_present, out, q);
        } else {
            if (p.top_k <= 0) {
                std::fprintf(stderr,
                             "sampler: the sampled path needs top_k in 1..64 (got %d); greedy needs no filters\n",
                             p.top_k);
                return;   // leave out[t] unwritten rather than returning an uninitialised token
            }
            const uint64_t seed = ((uint64_t) seed_hi << 32) | (uint64_t) seed_lo;
            const uint64_t counter = ((uint64_t) counter_hi << 32) | (uint64_t) counter_lo;
            sampler_run(logits, n_vocab, n_tokens, history, history_len, p.top_k, p.top_p, p.min_p,
                        p.temperature, p.min_keep, p.penalty_last_n, penalty_repeat, penalty_freq,
                        penalty_present, seed, counter, out, q);
        }
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "sample_tokens launch: %s\n", e.what());
        std::exit(1);
    }
}

}  // namespace strata::kernels
