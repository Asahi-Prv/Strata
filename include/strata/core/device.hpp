// include/strata/core/device.hpp - backend-agnostic device surface (SYCLomatic port: the device layer).
//
// This header names NO CUDA AND NO SYCL symbol.  The whole engine already talks to the device through opaque
// wire formats - `void* stream` (a cudaStream_t on the CUDA build, a sycl::queue* on the SYCL build) and raw
// device pointers (`float*`, `uint16_t*`, ...) passed as kernel arguments.  Everything that DOES name a
// concrete backend lives behind DeviceRuntime in src/core/device_{cuda,sycl,runtime}.cpp, so a new backend is
// one implementation of this one interface and nothing above it has to change.
//
// `DeviceArena` is ONE backend allocation per planner region with bump sub-allocation below it and no frees.
// The memory plan from P1.S9 is fixed at startup, so the set of regions and their sizes is known before
// anything is allocated; an allocator that could free would be solving a problem the engine does not have
// while adding fragmentation and failure modes it does.  The VRAM/VRAM-equivalent budget is the binding
// constraint of the whole design, and the plan is printed against it at startup so the discrepancy is visible
// immediately.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace strata::core {

/// Which backend the process is running.  `select_runtime` fixes this once at startup; the default is CUDA,
/// which is what the published binary builds and runs against today.
enum class DeviceKind : int { CUDA = 0, SYCL = 1 };
inline const char* to_string(DeviceKind k) { return k == DeviceKind::CUDA ? "cuda" : "sycl"; }

/// Backend-neutral device facts.  `arch_name` / `compute_units` replace the CUDA-only compute capability so a
/// single struct prints on an RTX (sm_120) and on an Arc (EU count) with one field set per backend.  A
/// binary carried to another machine still says what it found here, which is the whole point of the struct.
struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    std::string arch_name;          // "sm_120" on CUDA, an architecture string on SYCL
    int compute_units = 0;          // multiprocessors / EU count, one backend-neutral name
    uint64_t total_bytes = 0;       // as reported by the backend at query time
    uint64_t free_bytes = 0;
    int driver_version = 0;         // backend driver version, or 0 if the backend does not report one
};

/// A backend-independent error.  `CudaError` becomes this: the code carries the backend it came from, so a
/// caller that cares about the kind (or the parity harnesses that catch it) can still branch on it.
class DeviceError : public std::runtime_error {
public:
    DeviceError(const std::string& what, DeviceKind kind, int code)
        : std::runtime_error(what), kind_(kind), code_(code) {}
    DeviceKind kind() const { return kind_; }
    int code() const { return code_; }

private:
    DeviceKind kind_;
    int code_;
};

/// A launch stream, opaque on the wire.  `raw()` is a cudaStream_t on the CUDA build and a sycl::queue* on the
/// SYCL build; the null stream is the default stream (CUDA) / current queue (SYCL), represented the same way
/// so no call site special-cases "is this the default".  Keeping `void*` on the wire is what lets SYCLomatic's
/// output and every one of the ~100 kernel entry points stay untouched.
class DeviceStream {
public:
    DeviceStream() = default;
    DeviceStream(void* raw, DeviceKind kind) : raw_(raw), kind_(kind) {}
    void* raw() const { return raw_; }
    DeviceKind kind() const { return kind_; }
    bool is_default() const { return raw_ == nullptr; }
    explicit operator bool() const { return raw_ != nullptr; }

private:
    void* raw_ = nullptr;
    DeviceKind kind_ = DeviceKind::CUDA;
};

/// A host<->device copy direction.  The values are backend-neutral; each backend maps them onto its own
/// enum (cudaMemcpyKind / the SYCL memcpy overload) so no kernel or parity harness carries a CUDA enum.
enum class MemcpyKind : int { HostToDevice, DeviceToHost, DeviceToDevice, Default };

/// One backend.  `make_cuda_runtime` / `make_sycl_runtime` build one and return nullptr when that backend is
/// not available here; `active_runtime()` is what DeviceArena, the copy helpers and the sync helper dispatch
/// through, so the engine never names a backend directly.
class DeviceRuntime {
public:
    virtual ~DeviceRuntime() = default;
    virtual DeviceKind kind() const = 0;
    virtual const char* name() const = 0;
    virtual DeviceInfo device_info(int ordinal) = 0;
    virtual void* device_alloc(uint64_t bytes) = 0;     // device-side memory (cudaMalloc / malloc_device)
    virtual void device_free(void* p) = 0;
    virtual void* stream_create() = 0;                  // a non-default launch stream
    virtual void stream_destroy(void* s) = 0;
    virtual void memcpy(void* stream, void* dst, const void* src, size_t bytes, MemcpyKind kind) = 0;
    virtual void synchronize(void* stream) = 0;         // null stream == sync everything
    virtual void poison(void* p, size_t bytes) = 0;     // fill with a NaN-ish pattern (see DeviceArena)
};

/// Pick the backend.  Called once at startup (the engine entry point does it); absent that the first use of
/// the surface falls back to CUDA.  Returns the runtime, or nullptr if the requested backend is unavailable.
std::unique_ptr<DeviceRuntime> select_runtime(DeviceKind kind);
void set_active_runtime(std::unique_ptr<DeviceRuntime> rt);   // override the startup choice (tests, --selftest)
DeviceRuntime* active_runtime();

/// The public device surface the engine already calls.  `device_info` dispatches through `active_runtime()`.
DeviceInfo device_info(int ordinal = 0);

/// One backend allocation, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a
/// debug build so that reading uninitialised device memory gives a NaN rather than a plausible number - the
/// same reasoning as the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

/// A copy the engine and the parity harnesses can call without a CUDA or SYCL include.
void device_memcpy(void* stream, void* dst, const void* src, size_t bytes, MemcpyKind kind);
/// Sync the stream (or everything, when `stream` is null).
void device_synchronize(void* stream);

}  // namespace strata::core
