// src/core/pinned_sycl.cpp - the SYCL backend for the pinned host arena (pinned.cu's counterpart).
//
// pinned.cu is the CUDA shape: it reserves OS pages with VirtualAlloc/mmap, then asks the driver to pin and map
// that EXISTING range with `cudaHostRegister(cudaHostRegisterPortable | cudaHostRegisterMapped)` (falling back
// to per-slice registration, then to a working-set lock), and hands the device a second address through
// `cudaHostGetDevicePointer`.  The engine needs exactly one property from all of that - a page-locked host
// buffer the GPU can DMA to and from - and USM gives that property directly.
//
// THE CENTRAL DIVERGENCE, stated once and relied on below:
//
//   `cudaHostRegister` pins memory the CALLER already owns.  USM has no equivalent operation: a USM allocation
//   (`sycl::malloc_host` / `malloc_device` / `malloc_shared`) must be created by the runtime, and only those
//   allocations are device-accessible.  There is no `sycl::register_host` for an arbitrary `new`/`malloc`/
//   `VirtualAlloc` range.  Therefore PinnedArena does NOT reserve + register; it allocates the arena as a USM
//   host allocation up front with `sycl::malloc_host`.  That is page-locked and device-accessible in the same
//   context, which is the property the copy engine wanted.  The large-page / working-set-lock probes in
//   pinned.cu have no role here: the runtime owns the backing and lock is not something the caller controls.
//
// The secondary divergences (mapped device pointer, and "register an existing region") are handled explicitly:
//
//   * `cudaHostGetDevicePointer` has no USM analog because there is only ONE address.  With USM the host
//     pointer returned by `sycl::malloc_host` is already valid in device kernels and copies; the "device
//     pointer" IS the host pointer.  host_get_device_pointer() below returns the same pointer.
//   * `cudaHostRegister` of an already-allocated non-USM range is emulated by copying that range into a
//     `sycl::malloc_host` shadow and recording the map.  That is only correct for read-mostly data and it is
//     NOT what PinnedArena does (PinnedArena would have to copy 30+ GiB, which defeats the point); it is kept
//     for the general call shape and is documented as best-effort.
//
// Compiled with -fsycl.  It includes only <sycl/sycl.hpp>, never a CUDA header.
#include "strata/core/pinned.hpp"
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

namespace {

// The one place the opaque `void* stream` wire becomes a queue.  A null stream is the default SYCL queue, the
// same convention device_sycl.cpp uses, so a call site that passed CUDA's null stream still lands on a real
// queue here.
sycl::queue* q_of(void* stream) {
    return static_cast<sycl::queue*>(stream ? stream : default_sycl_queue());
}

// ---- USM host allocation: this replaces cudaHostAlloc(cudaHostAllocMapped) ----------
//
// `sycl::malloc_host` returns page-locked memory that is accessible from both the host and the device within
// the allocation's context.  It is the direct equivalent of a portable, mapped cudaHostAlloc - except that the
// pointer is single, so no cudaHostGetDevicePointer step is needed.
void* host_alloc(uint64_t bytes, sycl::queue& q) {
    return sycl::malloc_host((size_t) bytes, q);
}

// ---- register/unregister of an EXISTING host region --------------------------------
//
// cudaHostRegisterMapped cannot be reproduced faithfully.  The closest correct behavior is a shadow copy into
// USM host memory: the caller's range stays where it is (the host reads it through `user`), and the device must
// use the returned `usm` pointer.  Writes by the device do NOT flow back into the caller's range; a caller that
// needs that must copy back.  PinnedArena does NOT go through this path - see the file header.
struct HostRegistration {
    const void* user = nullptr;
    size_t bytes = 0;
    void* usm = nullptr;
    sycl::context ctx;
};

std::mutex& registration_mutex() {
    static std::mutex m;
    return m;
}

std::vector<HostRegistration>& registrations() {
    static std::vector<HostRegistration> v;
    return v;
}

// ---- "get device pointer" for a USM host allocation --------------------------------
//
// Identity.  USM has one address space reachable from both sides.  This exists to keep the cudaHost* vocabulary
// visible, and returns the same pointer for a USM region and for a shadow created by host_register_mapped().
[[maybe_unused]] void* host_get_device_pointer(const void* user) {
    std::lock_guard<std::mutex> g(registration_mutex());
    for (const HostRegistration& r : registrations()) {
        if (r.user == user) return r.usm;
    }
    return const_cast<void*>(user);
}

[[maybe_unused]] void* host_register_mapped(const void* user, uint64_t bytes, sycl::queue& q) {
    if (user == nullptr || bytes == 0) return nullptr;
    void* shadow = sycl::malloc_host((size_t) bytes, q);
    if (shadow == nullptr) return nullptr;
    std::memcpy(shadow, user, (size_t) bytes);   // the one-time snapshot; writes do not track afterwards
    std::lock_guard<std::mutex> g(registration_mutex());
    registrations().push_back(HostRegistration{user, (size_t) bytes, shadow, q.get_context()});
    return shadow;
}

// cudaHostUnregister: drop the shadow.  The shadow is the device-visible object, so it is the thing freed.
[[maybe_unused]] void host_unregister_mapped(const void* user) {
    std::lock_guard<std::mutex> g(registration_mutex());
    auto& v = registrations();
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (it->user == user) {
            void* shadow = it->usm;
            sycl::context ctx = it->ctx;
            v.erase(it);
            sycl::free(shadow, ctx);
            return;
        }
    }
}

// ---- the doorbell fence ------------------------------------------------------------
//
// pinned.cu's doorbell is a payload write, `__threadfence_system()`, then a volatile store of the doorbell
// word; a host thread spins on the word and only then reads the payload.  USM host memory is system-scope
// coherent, so the same guarantee is a system-scope release fence before the volatile store (and a seq_cst
// system fence after it, so the store itself is ordered too).  pinned.hpp declares no doorbell symbol; this
// preserves the sequence for the layer that does own the doorbell.
[[maybe_unused]] void doorbell_publish_fence(void* d_seq, uint32_t value, void* stream) {
    sycl::queue* q = q_of(stream);
    try {
        q->submit([&](sycl::handler& h) {
            h.single_task([=]() {
                // Make everything the device wrote before this point visible system-wide, then publish.
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                *static_cast<volatile uint32_t*>(d_seq) = value;
                sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            });
        });
        if (stream == nullptr) q->wait();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "doorbell_publish_fence launch: %s\n", e.what());
        std::exit(1);
    }
}

// The CUDA constructor's per-slice registration partition.  Kept for signature parity; PinnedArena on SYCL
// pins the whole allocation in one shot, so the result is not used for slicing.
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// ---- the pinned arena -------------------------------------------------------------
//
// cudaHostAlloc(cudaHostAllocMapped | cudaHostAllocPortable) -> sycl::malloc_host: one page-locked, device-
// visible host allocation.  The whole allocation is "registered" in the sense pinned.hpp cares about
// (registered_bytes == capacity), because USM host memory is device-accessible without a separate step.
PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    (void) slice;   // slicing exists to work around refused registrations; USM never refuses per slice
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds) : capacity(bytes) {
    (void) bounds;   // no slice registration on SYCL
    if (bytes == 0) return;

    sycl::queue* q = q_of(nullptr);
    void* p = nullptr;
    try {
        p = host_alloc(bytes, *q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "PinnedArena: sycl::malloc_host(%llu B) failed: %s\n",
                     (unsigned long long) bytes, e.what());
        std::exit(1);
    }
    if (p == nullptr) {
        note = "sycl::malloc_host(" + std::to_string(bytes) +
               " B) FAILED - the arena is absent; copies will not be possible";
        capacity = 0;
        return;
    }

    base = p;
    // The header's enum still carries the CUDA name; PinnedByCuda means "the backend pinned it" and is the
    // honest tag for a USM host allocation.
    backing = PageBacking::PinnedByCuda;
    registered_bytes = bytes;   // the whole range is device-visible
    locked_bytes = bytes;       // malloc_host is page-locked by construction
    note = "sycl::malloc_host USM host allocation (page-locked, device-accessible; single address, no "
           "cudaHostGetDevicePointer). cudaHostRegister of an existing OS reservation has no USM equivalent, "
           "so the arena is allocated as USM host from the start; the large-page/working-set probes do not "
           "apply.";
}

PinnedArena::~PinnedArena() {
    if (base) {
        sycl::queue* q = q_of(nullptr);
        sycl::free(base, *q);
        base = nullptr;
    }
}

// ---- the parallel expert load (host-only; unchanged from pinned.cu) -----------------
LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // one handle per thread, seeked once per layer: a shared handle would need a lock around the seek
            // and would serialise the very thing the threads are here to parallelise
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "cannot open " + path;
                return;
            }
            f.seekg((std::streamoff) off);
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                f.read((char*) buf.data(), (std::streamsize) n);
                if ((uint64_t) f.gcount() != n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L);
                    return;
                }
                std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        return st;
    }
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

// ---- the stream bandwidth probe ---------------------------------------------------
//
// cudaMalloc + cudaStreamCreate become one malloc_device + the default queue; cudaMemcpyAsync(H2D) becomes
// handler::memcpy, and the two cudaStreamSynchronize calls become q.wait().  `src` is USM host memory, so the
// copy is a real pinned H2D transfer.
StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    if (chunk == 0) {
        std::fprintf(stderr, "stream_bandwidth: chunk is 0\n");
        st.seconds = -1.0;
        return st;
    }

    sycl::queue& q = *q_of(nullptr);
    void* dst = nullptr;
    try {
        dst = sycl::malloc_device((size_t) chunk, q);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "stream_bandwidth: sycl::malloc_device failed: %s\n", e.what());
        std::exit(1);
    }
    if (dst == nullptr) {
        std::fprintf(stderr, "stream_bandwidth: sycl::malloc_device failed for %llu B\n",
                     (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }

    auto submit_pass = [&]() {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            q.submit([&](sycl::handler& h) { h.memcpy(dst, src + off, (size_t) chunk); });
        }
    };

    try {
        // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
        submit_pass();
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) submit_pass();
        q.wait();
        st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "stream_bandwidth submit: %s\n", e.what());
        std::exit(1);
    }

    sycl::free(dst, q);
    return st;
}

}  // namespace strata::core
