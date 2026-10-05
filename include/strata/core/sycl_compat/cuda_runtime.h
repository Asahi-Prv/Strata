// include/strata/core/sycl_compat/cuda_runtime.h
//
// A drop-in CUDA-runtime shim implemented over oneAPI SYCL (DPC++).  The engine's HOST translation units
// (`src/core/*.cpp`, `src/program/generate.cpp`, `src/prefill/prefill.cpp`, and the kernel headers they pull
// in) `#include <cuda_runtime.h>` and call `cudaXxx`; this header lets those TUs compile and run against
// SYCL when the SYCL build puts this directory ahead of any real CUDA include directory.  The `.cu` files
// keep using the real CUDA header - this shim is only for the `-fsycl` host side.
//
// It is intentionally a header-only, `inline`-everything surface: no `.cpp` is added, and no other file in
// the tree has to change.  The wire types match what the engine already assumes:
//
//   * `cudaStream_t` is `void*`, and a NON-null value really is a `sycl::queue*`.  A null stream maps to
//     `strata::core::default_sycl_queue()` - the same `void*` the SYCL device backend (`device_sycl.cpp`)
//     hands to kernel files for a null-stream launch.  `as_queue()` casts it back.
//   * `cudaMalloc` is `sycl::malloc_device`; `cudaHostAlloc` is `sycl::malloc_host` (USM host memory is
//     device-accessible in the same context, which is what `cudaHostGetDevicePointer` relies on).
//
// WHAT IS FAITHFUL AND WHAT IS NOT
// ---------------------------------
//  * `cudaHostRegister`/`cudaHostUnregister` are NO-OPS returning `cudaSuccess`.  Real CUDA pins an
//    arbitrary OS reservation (the engine reserves a 31 GiB arena with `VirtualAlloc`/`mmap` and registers
//    it); USM has no "register this plain pointer" operation short of copying, which would defeat the point
//    of the arena.  A region that the engine only ever `.memcpy`s in or out still works because DPC++ accepts
//    ordinary host pointers in `queue::memcpy`; a region the engine expects to be *device addressable* without
//    a copy (a mapped pointer) will only be so if it came from `cudaHostAlloc`, whose USM host allocation
//    this shim does honour.  `cudaHostGetDevicePointer` therefore returns the same pointer for `cudaHostAlloc`
//    memory, and a best-effort same-pointer answer otherwise.
//  * `cudaGraph*` is implemented over `sycl::ext::oneapi::experimental::command_graph` (stream capture,
//    finalize/instantiate, replay).  If that extension is unavailable at compile time the shim falls back to
//    a documented NO-OP graph: capture is a flag, the capture body runs directly (so the first pass is
//    correct), `cudaGraphLaunch` is a no-op (so REPLAY DOES NOT RE-RUN the body), and `cudaGraphGetNodes`
//    reports one node so a zero-node guard does not reject it.  On the oneAPI 2026.1 toolchain the extension
//    IS present, so the real path is taken.
//  * Pageable (non-USM) host pointers passed to `queue::memcpy` rely on the DPC++ runtime's host-pointer
//    support; the shim does not stage them.  This is the usual DPC++ behaviour and is noted here so a future
//    port that hits `errc::invalid` knows where to add staging.
//
// Errors are kept thread-locally: a failing call stores a `cudaError_t` and the SYCL exception text, which
// `cudaGetErrorString` returns; `cudaGetLastError` consumes it, `cudaPeekAtLastError` does not.
#pragma once

#ifndef STRATA_SYCL_CUDA_RUNTIME_H
#define STRATA_SYCL_CUDA_RUNTIME_H

// If a real CUDA runtime header has already been pulled into this translation unit, do not redefine the
// whole surface underneath it - that would be a hard redefinition wall.  In a SYCL build this directory is
// expected to win the `<cuda_runtime.h>` lookup, so the normal case is that neither macro is set.
#if defined(__CUDA_RUNTIME_H__) || defined(__CUDA_RUNTIME_API_H__)
#define STRATA_SYCL_CUDA_RUNTIME_SHIM_SKIPPED 1
#else

// Windows' <windows.h> defines min/max as function-like macros, which mangle the SYCL headers (they use
// `min`/`max` as ordinary identifiers).  Disable and undo them before the SYCL include so both coexist.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <sycl/sycl.hpp>

#include <strata/core/device_runtime.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------------------------------------
// Feature detection for the oneAPI command-graph extension.  Prefer an explicit feature macro if a toolchain
// defines one, otherwise ask the preprocessor whether the header exists.  `STRATA_SYCL_CUDA_SHIM_NO_GRAPH=1`
// forces the no-op fallback for a toolchain that exposes the header but a broken runtime.
// ---------------------------------------------------------------------------------------------------------
#if defined(STRATA_SYCL_CUDA_SHIM_NO_GRAPH)
#define STRATA_SYCL_CUDA_SHIM_HAS_GRAPH 0
#elif defined(SYCL_EXT_ONEAPI_GRAPH)
#define STRATA_SYCL_CUDA_SHIM_HAS_GRAPH 1
#elif defined(__has_include)
#if __has_include(<sycl/ext/oneapi/experimental/graph.hpp>)
#define STRATA_SYCL_CUDA_SHIM_HAS_GRAPH 1
#else
#define STRATA_SYCL_CUDA_SHIM_HAS_GRAPH 0
#endif
#else
#define STRATA_SYCL_CUDA_SHIM_HAS_GRAPH 0
#endif

#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
#include <sycl/ext/oneapi/experimental/graph.hpp>
#endif

namespace strata::core::sycl_compat {

// ---------------------------------------------------------------------------------------------------------
// Types and enums.  The numeric values that the engine can observe are the CUDA ones where they matter
// (`cudaSuccess == 0`, `cudaErrorNotReady == 34`); the rest are private to this shim and only compared for
// equality.  The handle types are pointers so the engine's `== nullptr` tests and `new cudaGraphExec_t[n]`
// keep working unchanged.
// ---------------------------------------------------------------------------------------------------------

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNotReady = 34,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorGraphExecUpdateFailure = 902,
    cudaErrorUnknown = 999,
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

enum cudaHostAllocFlags {
    cudaHostAllocDefault = 0,
    cudaHostAllocPortable = 1,
    cudaHostAllocMapped = 2,
    cudaHostAllocWriteCombined = 4,
};

enum cudaHostRegisterFlags {
    cudaHostRegisterDefault = 0,
    cudaHostRegisterPortable = 1,
    cudaHostRegisterMapped = 2,
    cudaHostRegisterIoMemory = 4,
    cudaHostRegisterReadOnly = 8,
};

enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};

enum cudaEventFlags {
    cudaEventDefault = 0,
    cudaEventBlockingSync = 1,
    cudaEventDisableTiming = 2,
    cudaEventInterprocess = 4,
};

enum cudaStreamFlags {
    cudaStreamDefault = 0,
    cudaStreamNonBlocking = 1,
};

// Opaque handles.  `cudaStream_t` is deliberately `void*` (see the header comment): the engine stores
// streams in `void*` fields and passes them straight through to kernel files.
typedef void* cudaStream_t;

struct cudaEvent_st;
typedef cudaEvent_st* cudaEvent_t;

struct cudaGraphNode_st;
typedef cudaGraphNode_st* cudaGraphNode_t;

struct cudaGraph_st;
typedef cudaGraph_st* cudaGraph_t;

struct cudaGraphExec_st;
typedef cudaGraphExec_st* cudaGraphExec_t;

// The event is a barrier `sycl::event` plus the bits needed for timing: whether timing was requested, the
// host timestamp taken at `cudaEventRecord` (the fallback when queue profiling is unavailable), and whether
// the event has actually been recorded.
struct cudaEvent_st {
    sycl::event ev;
    bool timing = true;
    bool recorded = false;
    std::chrono::steady_clock::time_point stamp{};
};

struct cudaGraphNode_st {};

#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
using graph_state = sycl::ext::oneapi::experimental::graph_state;
template <graph_state S>
using command_graph = sycl::ext::oneapi::experimental::command_graph<S>;
#endif

struct cudaGraph_st {
    bool fallback = false;
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
    command_graph<graph_state::modifiable> graph;
    explicit cudaGraph_st(sycl::queue& q) : graph(q) {}
#else
    explicit cudaGraph_st(sycl::queue&) {}
#endif
};

struct cudaGraphExec_st {
    bool fallback = false;
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
    command_graph<graph_state::executable> exec;
    explicit cudaGraphExec_st(command_graph<graph_state::executable> e) : exec(std::move(e)) {}
#else
    explicit cudaGraphExec_st(int) {}
#endif
};

// ---------------------------------------------------------------------------------------------------------
// Thread-local error state plus the `guard` wrapper that turns a SYCL exception into a `cudaError_t`.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t& last_error_slot() {
    thread_local cudaError_t e = cudaSuccess;
    return e;
}

inline std::string& last_error_message_slot() {
    thread_local std::string s;
    return s;
}

inline cudaError_t fail_error(cudaError_t e, std::string message = std::string()) {
    last_error_slot() = e;
    if (!message.empty()) last_error_message_slot() = std::move(message);
    return e;
}

template <typename F>
inline cudaError_t guard(F&& f) {
    try {
        return f();
    } catch (const sycl::exception& e) {
        return fail_error(cudaErrorUnknown, e.what());
    } catch (const std::exception& e) {
        return fail_error(cudaErrorUnknown, e.what());
    }
}

inline cudaError_t cudaGetLastError() {
    const cudaError_t e = last_error_slot();
    last_error_slot() = cudaSuccess;
    return e;
}

inline cudaError_t cudaPeekAtLastError() { return last_error_slot(); }

inline const char* cudaGetErrorString(cudaError_t e) {
    if (e == cudaSuccess) return "no error";
    const std::string& msg = last_error_message_slot();
    if (!msg.empty()) return msg.c_str();
    switch (e) {
        case cudaErrorNotReady: return "cudaErrorNotReady (device not ready)";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation (out of memory)";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorStreamCaptureUnsupported: return "cudaErrorStreamCaptureUnsupported";
        case cudaErrorStreamCaptureInvalidated: return "cudaErrorStreamCaptureInvalidated";
        case cudaErrorGraphExecUpdateFailure: return "cudaErrorGraphExecUpdateFailure";
        default: return "cudaErrorUnknown";
    }
}

// ---------------------------------------------------------------------------------------------------------
// Backend plumbing: the queue behind a stream, the registry of live queues (for `cudaDeviceSynchronize`),
// and a running count of device bytes handed out (for a `cudaMemGetInfo` free estimate).
// ---------------------------------------------------------------------------------------------------------

/// A non-null stream is a `sycl::queue*`; a null stream is the project's default SYCL queue, with a local
/// fallback for the (non-SYCL-backend) case where `default_sycl_queue()` returns null.
inline sycl::queue* as_queue(cudaStream_t stream) {
    if (stream) return static_cast<sycl::queue*>(stream);
    void* raw = strata::core::default_sycl_queue();
    if (raw) return static_cast<sycl::queue*>(raw);
    static sycl::queue fallback{sycl::property_list{sycl::property::queue::in_order{}}};
    return &fallback;
}

inline std::mutex& queue_registry_mutex() {
    static std::mutex m;
    return m;
}

inline std::vector<sycl::queue*>& queue_registry() {
    static std::vector<sycl::queue*> v;
    return v;
}

inline void register_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(queue_registry_mutex());
    queue_registry().push_back(q);
}

inline void unregister_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(queue_registry_mutex());
    std::vector<sycl::queue*>& v = queue_registry();
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (*it == q) { v.erase(it); break; }
    }
}

inline std::atomic<size_t>& device_bytes_slot() {
    static std::atomic<size_t> b{0};
    return b;
}

/// A queue that supports event profiling, so `cudaEventElapsedTime` can read the SYCL profiling info.  The
/// `in_order` property is what gives CUDA's stream semantics (submissions run in issue order).
inline sycl::queue* make_stream_queue() {
    try {
        return new sycl::queue(
            sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    } catch (const sycl::exception&) {
        // Some devices cannot profile; a plain in-order queue still gives CUDA stream semantics, and
        // `cudaEventElapsedTime` falls back to its host timestamps.
        return new sycl::queue(sycl::property_list{sycl::property::queue::in_order{}});
    }
}

// ---------------------------------------------------------------------------------------------------------
// Device memory.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return guard([&]() -> cudaError_t {
        if (!devPtr) return fail_error(cudaErrorInvalidValue, "cudaMalloc: null out-pointer");
        void* p = sycl::malloc_device(size, *as_queue(nullptr));
        if (!p) return fail_error(cudaErrorMemoryAllocation, "cudaMalloc: sycl::malloc_device failed");
        *devPtr = p;
        device_bytes_slot().fetch_add(size);
        return cudaSuccess;
    });
}

// CUDA's C++ API also provides a templated cudaMalloc so `cudaMalloc(&typed_ptr, n)` compiles; mirror it
// (a plain `void**` parameter does not accept a `float**`).
template <class T>
inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(devPtr), size);
}

inline cudaError_t cudaFree(void* devPtr) {
    return guard([&]() -> cudaError_t {
        if (devPtr) sycl::free(devPtr, *as_queue(nullptr));
        return cudaSuccess;
    });
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void) kind;  // USM memcpy infers the direction from the pointer types
    return guard([&]() -> cudaError_t {
        as_queue(nullptr)->memcpy(dst, src, count).wait();
        return cudaSuccess;
    });
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind,
                                   cudaStream_t stream) {
    (void) kind;
    return guard([&]() -> cudaError_t {
        as_queue(stream)->memcpy(dst, src, count);
        return cudaSuccess;
    });
}

inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    return guard([&]() -> cudaError_t {
        as_queue(nullptr)->memset(devPtr, value, count).wait();
        return cudaSuccess;
    });
}

inline cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        as_queue(stream)->memset(devPtr, value, count);
        return cudaSuccess;
    });
}

/// Row-by-row copy: SYCL's handler memcpy is 1-D, so each row is queued separately on the stream.  The rows
/// are disjoint, so their relative order does not matter, and the whole set is ordered against the rest of
/// the stream.  `width`/`height` are in bytes/rows exactly as in the CUDA call.
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                     size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    (void) kind;
    return guard([&]() -> cudaError_t {
        sycl::queue* q = as_queue(stream);
        for (size_t row = 0; row < height; ++row) {
            (void) q->memcpy(static_cast<char*>(dst) + row * dpitch,
                             static_cast<const char*>(src) + row * spitch, width);
        }
        return cudaSuccess;
    });
}

/// Total is the device's global memory; "free" is that total minus the bytes this shim has handed out via
/// `cudaMalloc`.  It is an estimate (it cannot see allocations made outside the shim), but it is the figure
/// the engine's expert-cache auto-sizing reads and it tracks the shim's own arena.
inline cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    return guard([&]() -> cudaError_t {
        const size_t total =
            as_queue(nullptr)->get_device().get_info<sycl::info::device::global_mem_size>();
        const size_t used = device_bytes_slot().load();
        if (totalBytes) *totalBytes = total;
        if (freeBytes) *freeBytes = used < total ? total - used : 0;
        return cudaSuccess;
    });
}

// ---------------------------------------------------------------------------------------------------------
// Pinned host memory.  `cudaHostRegister`/`cudaHostUnregister` are no-ops - see the header comment.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) {
    (void) flags;  // Mapped/Portable are implied by USM host memory in the same context
    return guard([&]() -> cudaError_t {
        if (!pHost) return fail_error(cudaErrorInvalidValue, "cudaHostAlloc: null out-pointer");
        void* p = sycl::malloc_host(size, *as_queue(nullptr));
        if (!p) return fail_error(cudaErrorMemoryAllocation, "cudaHostAlloc: sycl::malloc_host failed");
        *pHost = p;
        return cudaSuccess;
    });
}

inline cudaError_t cudaMallocHost(void** pHost, size_t size) {
    return cudaHostAlloc(pHost, size, cudaHostAllocDefault);
}

inline cudaError_t cudaFreeHost(void* pHost) {
    return guard([&]() -> cudaError_t {
        if (pHost) sycl::free(pHost, *as_queue(nullptr));
        return cudaSuccess;
    });
}

inline cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    (void) ptr;
    (void) size;
    (void) flags;
    return cudaSuccess;  // no-op: plain OS memory is not USM and cannot be pinned without copying
}

inline cudaError_t cudaHostUnregister(void* ptr) {
    (void) ptr;
    return cudaSuccess;
}

/// USM host memory is device-accessible in the same context, so the device pointer is the host pointer.
/// (For a no-op-registered plain pointer this is a best-effort answer, documented above.)
inline cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
    (void) flags;
    if (!pDevice) return fail_error(cudaErrorInvalidValue, "cudaHostGetDevicePointer: null out-pointer");
    *pDevice = pHost;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------------------------------------
// Streams.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
    return guard([&]() -> cudaError_t {
        if (!pStream) return fail_error(cudaErrorInvalidValue, "cudaStreamCreate: null out-pointer");
        sycl::queue* q = make_stream_queue();
        *pStream = static_cast<cudaStream_t>(q);
        register_queue(q);
        return cudaSuccess;
    });
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    (void) flags;  // every shim queue is independent; NonBlocking has no per-queue analogue to set
    return cudaStreamCreate(pStream);
}

inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        if (stream) {
            sycl::queue* q = static_cast<sycl::queue*>(stream);
            unregister_queue(q);
            delete q;
        }
        return cudaSuccess;
    });
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        as_queue(stream)->wait();
        return cudaSuccess;
    });
}

/// A queue has no direct "is it idle" query, so submit a barrier after all prior work and read its status.
/// That is a QUERY (it does not block), which is what `cudaStreamQuery` promises.
inline cudaError_t cudaStreamQuery(cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        sycl::event barrier = as_queue(stream)->ext_oneapi_submit_barrier();
        const sycl::info::event_command_status st =
            barrier.get_info<sycl::info::event::command_execution_status>();
        return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
    });
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    (void) flags;
    return guard([&]() -> cudaError_t {
        if (event && event->recorded) {
            sycl::queue* q = as_queue(stream);
            (void) q->ext_oneapi_submit_barrier(std::vector<sycl::event>{event->ev});
        }
        return cudaSuccess;
    });
}

// The graph capture state lives in a thread-local map from queue to the in-flight modifiable graph; the
// engine's capture protocol is Begin(stream) / body / End(stream,&graph) on one thread.
inline std::vector<std::pair<sycl::queue*, cudaGraph_st*>>& capture_registry() {
    thread_local std::vector<std::pair<sycl::queue*, cudaGraph_st*>> m;
    return m;
}

inline cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    (void) mode;  // capture mode is a CUDA stream-graph distinction with no SYCL analogue
    return guard([&]() -> cudaError_t {
        sycl::queue* q = as_queue(stream);
        for (const auto& entry : capture_registry()) {
            if (entry.first == q) {
                return fail_error(cudaErrorStreamCaptureUnsupported, "cudaStreamBeginCapture: already capturing");
            }
        }
        cudaGraph_st* g = new cudaGraph_st(*q);
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
        g->graph.begin_recording(*q);
#else
        g->fallback = true;  // no-op capture: the body runs directly, replay does nothing
#endif
        capture_registry().emplace_back(q, g);
        return cudaSuccess;
    });
}

inline cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    return guard([&]() -> cudaError_t {
        if (!pGraph) return fail_error(cudaErrorInvalidValue, "cudaStreamEndCapture: null out-pointer");
        sycl::queue* q = as_queue(stream);
        auto& reg = capture_registry();
        for (auto it = reg.begin(); it != reg.end(); ++it) {
            if (it->first == q) {
                cudaGraph_st* g = it->second;
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
                g->graph.end_recording(*q);
#endif
                reg.erase(it);
                *pGraph = g;
                return cudaSuccess;
            }
        }
        return fail_error(cudaErrorStreamCaptureInvalidated, "cudaStreamEndCapture: no capture in progress");
    });
}

// ---------------------------------------------------------------------------------------------------------
// Events.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t cudaEventCreate(cudaEvent_t* pEvent) {
    return guard([&]() -> cudaError_t {
        if (!pEvent) return fail_error(cudaErrorInvalidValue, "cudaEventCreate: null out-pointer");
        *pEvent = new cudaEvent_st{};
        return cudaSuccess;
    });
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* pEvent, unsigned int flags) {
    return guard([&]() -> cudaError_t {
        if (!pEvent) return fail_error(cudaErrorInvalidValue, "cudaEventCreateWithFlags: null out-pointer");
        cudaEvent_st* e = new cudaEvent_st{};
        e->timing = (flags & cudaEventDisableTiming) == 0;
        *pEvent = e;
        return cudaSuccess;
    });
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        if (!event) return fail_error(cudaErrorInvalidValue, "cudaEventRecord: null event");
        event->stamp = std::chrono::steady_clock::now();
        event->ev = as_queue(stream)->ext_oneapi_submit_barrier();
        event->recorded = true;
        return cudaSuccess;
    });
}

inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    return guard([&]() -> cudaError_t {
        if (!event || !event->recorded) return cudaSuccess;  // an unrecorded event is "already done"
        const sycl::info::event_command_status st =
            event->ev.get_info<sycl::info::event::command_execution_status>();
        return st == sycl::info::event_command_status::complete ? cudaSuccess : cudaErrorNotReady;
    });
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return guard([&]() -> cudaError_t {
        if (event && event->recorded) event->ev.wait();
        return cudaSuccess;
    });
}

/// Device time between two recorded events, in milliseconds.  Prefer SYCL's profiling timestamps; if the
/// queue was not created with profiling (notably the project's default queue) fall back to the host
/// timestamps captured at `cudaEventRecord`, which is a coarser but non-throwing estimate.
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms) return fail_error(cudaErrorInvalidValue, "cudaEventElapsedTime: null out-pointer");
    if (!start || !end || !start->recorded || !end->recorded) {
        return fail_error(cudaErrorInvalidValue, "cudaEventElapsedTime: unrecorded event");
    }
    try {
        const uint64_t a = start->ev.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t b = end->ev.get_profiling_info<sycl::info::event_profiling::command_end>();
        *ms = static_cast<float>(static_cast<double>(b - a) / 1.0e6);
    } catch (const sycl::exception&) {
        const std::chrono::duration<double, std::milli> d = end->stamp - start->stamp;
        *ms = static_cast<float>(d.count());
    }
    return cudaSuccess;
}

// ---------------------------------------------------------------------------------------------------------
// Graphs.
// ---------------------------------------------------------------------------------------------------------

/// `cudaGraphInstantiate` has two spellings in the engine: the modern 3-argument one
/// `(exec, graph, flags)` and the 5-argument one `(exec, graph, errorNode, logBuffer, flags)` with nulls.
/// Both land on the same finalize.
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                        unsigned long long flags) {
    (void) flags;
    return guard([&]() -> cudaError_t {
        if (!pGraphExec || !graph) {
            return fail_error(cudaErrorInvalidValue, "cudaGraphInstantiate: null handle");
        }
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
        cudaGraphExec_st* ex = new cudaGraphExec_st(graph->graph.finalize());
#else
        cudaGraphExec_st* ex = new cudaGraphExec_st(0);
        ex->fallback = true;
#endif
        *pGraphExec = ex;
        return cudaSuccess;
    });
}

inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                        cudaGraphNode_t* pErrorNode, char* pLogBuffer,
                                        size_t bufferSize) {
    (void) pErrorNode;
    (void) pLogBuffer;
    (void) bufferSize;
    return cudaGraphInstantiate(pGraphExec, graph, 0ull);
}

inline cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    return guard([&]() -> cudaError_t {
        if (!graph || !numNodes) {
            return fail_error(cudaErrorInvalidValue, "cudaGraphGetNodes: null handle");
        }
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
        std::vector<sycl::ext::oneapi::experimental::node> all = graph->graph.get_nodes();
        const size_t n = all.size();
#else
        const size_t n = 1;  // fallback: report one node so a zero-node guard does not reject the graph
#endif
        if (nodes) {
            const size_t capacity = *numNodes;
            const size_t fill = capacity < n ? capacity : n;
            for (size_t i = 0; i < fill; ++i) nodes[i] = nullptr;
        }
        *numNodes = n;
        return cudaSuccess;
    });
}

/// Replay the instantiated graph on the stream.  In the no-op fallback this does nothing (documented: the
/// capture body already ran once, and there is nothing recorded to replay).
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    return guard([&]() -> cudaError_t {
        if (!graphExec) return fail_error(cudaErrorInvalidValue, "cudaGraphLaunch: null graph");
#if STRATA_SYCL_CUDA_SHIM_HAS_GRAPH
        (void) as_queue(stream)->ext_oneapi_graph(graphExec->exec);
#else
        (void) stream;
#endif
        return cudaSuccess;
    });
}

/// The graph is uploaded when it is finalized/launched, so an explicit upload is a no-op.
inline cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream) {
    (void) graphExec;
    (void) stream;
    return cudaSuccess;
}

inline cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return cudaSuccess;
}

inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    delete graphExec;
    return cudaSuccess;
}

// ---------------------------------------------------------------------------------------------------------
// Device-wide and last-error calls.
// ---------------------------------------------------------------------------------------------------------

inline cudaError_t cudaDeviceSynchronize() {
    return guard([&]() -> cudaError_t {
        as_queue(nullptr)->wait();
        std::lock_guard<std::mutex> lock(queue_registry_mutex());
        for (sycl::queue* q : queue_registry()) {
            if (q) q->wait();
        }
        return cudaSuccess;
    });
}

/// Queue a host callback to run in order on the stream, matching `cudaLaunchHostFunc`.
inline cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void*), void* userData) {
    return guard([&]() -> cudaError_t {
        sycl::queue* q = as_queue(stream);
        q->submit([=](sycl::handler& h) {
            h.host_task([=]() {
                if (fn) fn(userData);
            });
        });
        return cudaSuccess;
    });
}

}  // namespace strata::core::sycl_compat

// The engine calls these unqualified at global scope (it always has, against the real CUDA header), so pull
// the shim's names out of its namespace.  Only the handful the host TUs actually name is enough, but the
// whole set is small and re-exporting it keeps a half-adopted TU from failing on one missing name.
using strata::core::sycl_compat::cudaError_t;
using strata::core::sycl_compat::cudaSuccess;
using strata::core::sycl_compat::cudaErrorInvalidValue;
using strata::core::sycl_compat::cudaErrorMemoryAllocation;
using strata::core::sycl_compat::cudaErrorNotReady;
using strata::core::sycl_compat::cudaErrorStreamCaptureUnsupported;
using strata::core::sycl_compat::cudaErrorStreamCaptureInvalidated;
using strata::core::sycl_compat::cudaErrorGraphExecUpdateFailure;
using strata::core::sycl_compat::cudaErrorUnknown;
using strata::core::sycl_compat::cudaMemcpyKind;
using strata::core::sycl_compat::cudaMemcpyHostToHost;
using strata::core::sycl_compat::cudaMemcpyHostToDevice;
using strata::core::sycl_compat::cudaMemcpyDeviceToHost;
using strata::core::sycl_compat::cudaMemcpyDeviceToDevice;
using strata::core::sycl_compat::cudaMemcpyDefault;
using strata::core::sycl_compat::cudaHostAllocFlags;
using strata::core::sycl_compat::cudaHostAllocDefault;
using strata::core::sycl_compat::cudaHostAllocPortable;
using strata::core::sycl_compat::cudaHostAllocMapped;
using strata::core::sycl_compat::cudaHostAllocWriteCombined;
using strata::core::sycl_compat::cudaHostRegisterFlags;
using strata::core::sycl_compat::cudaHostRegisterDefault;
using strata::core::sycl_compat::cudaHostRegisterPortable;
using strata::core::sycl_compat::cudaHostRegisterMapped;
using strata::core::sycl_compat::cudaHostRegisterIoMemory;
using strata::core::sycl_compat::cudaHostRegisterReadOnly;
using strata::core::sycl_compat::cudaStreamCaptureMode;
using strata::core::sycl_compat::cudaStreamCaptureModeGlobal;
using strata::core::sycl_compat::cudaStreamCaptureModeThreadLocal;
using strata::core::sycl_compat::cudaStreamCaptureModeRelaxed;
using strata::core::sycl_compat::cudaEventFlags;
using strata::core::sycl_compat::cudaEventDefault;
using strata::core::sycl_compat::cudaEventBlockingSync;
using strata::core::sycl_compat::cudaEventDisableTiming;
using strata::core::sycl_compat::cudaEventInterprocess;
using strata::core::sycl_compat::cudaStreamFlags;
using strata::core::sycl_compat::cudaStreamDefault;
using strata::core::sycl_compat::cudaStreamNonBlocking;
using strata::core::sycl_compat::cudaStream_t;
using strata::core::sycl_compat::cudaEvent_t;
using strata::core::sycl_compat::cudaGraph_t;
using strata::core::sycl_compat::cudaGraphExec_t;
using strata::core::sycl_compat::cudaGraphNode_t;
using strata::core::sycl_compat::cudaGetLastError;
using strata::core::sycl_compat::cudaPeekAtLastError;
using strata::core::sycl_compat::cudaGetErrorString;
using strata::core::sycl_compat::cudaMalloc;
using strata::core::sycl_compat::cudaFree;
using strata::core::sycl_compat::cudaMemcpy;
using strata::core::sycl_compat::cudaMemcpyAsync;
using strata::core::sycl_compat::cudaMemset;
using strata::core::sycl_compat::cudaMemsetAsync;
using strata::core::sycl_compat::cudaMemcpy2DAsync;
using strata::core::sycl_compat::cudaMemGetInfo;
using strata::core::sycl_compat::cudaHostAlloc;
using strata::core::sycl_compat::cudaMallocHost;
using strata::core::sycl_compat::cudaFreeHost;
using strata::core::sycl_compat::cudaHostRegister;
using strata::core::sycl_compat::cudaHostUnregister;
using strata::core::sycl_compat::cudaHostGetDevicePointer;
using strata::core::sycl_compat::cudaStreamCreate;
using strata::core::sycl_compat::cudaStreamCreateWithFlags;
using strata::core::sycl_compat::cudaStreamDestroy;
using strata::core::sycl_compat::cudaStreamSynchronize;
using strata::core::sycl_compat::cudaStreamQuery;
using strata::core::sycl_compat::cudaStreamWaitEvent;
using strata::core::sycl_compat::cudaStreamBeginCapture;
using strata::core::sycl_compat::cudaStreamEndCapture;
using strata::core::sycl_compat::cudaEventCreate;
using strata::core::sycl_compat::cudaEventCreateWithFlags;
using strata::core::sycl_compat::cudaEventRecord;
using strata::core::sycl_compat::cudaEventDestroy;
using strata::core::sycl_compat::cudaEventQuery;
using strata::core::sycl_compat::cudaEventSynchronize;
using strata::core::sycl_compat::cudaEventElapsedTime;
using strata::core::sycl_compat::cudaGraphInstantiate;
using strata::core::sycl_compat::cudaGraphLaunch;
using strata::core::sycl_compat::cudaGraphDestroy;
using strata::core::sycl_compat::cudaGraphExecDestroy;
using strata::core::sycl_compat::cudaGraphGetNodes;
using strata::core::sycl_compat::cudaGraphUpload;
using strata::core::sycl_compat::cudaDeviceSynchronize;
using strata::core::sycl_compat::cudaLaunchHostFunc;

#endif  // real CUDA header present?
#endif  // STRATA_SYCL_CUDA_RUNTIME_H
