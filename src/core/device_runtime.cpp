// src/core/device_runtime.cpp - the backend dispatch (SYCLomatic port).
//
// This translation unit owns the one runtime the process is on and the backend-agnostic shapes of the device
// surface.  It names no backend directly: every device allocation, copy, sync and poison goes through
// `active_runtime()->...`, so the CUDA build links device_cuda.cpp and the SYCL build links device_sycl.cpp
// and nothing here changes between them.
#include "strata/core/device_runtime.hpp"

#include <mutex>

namespace strata::core {

namespace {
// One mutex guards BOTH the lazy construction and an explicit override, because the accessor and
// set_active_runtime touch the same (owned, ptr) pair.  Construction is a one-time event; the mutex cost
// is paid on that one path and on the rare set_active_runtime call, never on a token-path allocation.
std::mutex g_runtime_mutex;
std::unique_ptr<DeviceRuntime> g_owned;
DeviceRuntime* g_ptr = nullptr;

DeviceRuntime* current_locked() {
    if (!g_ptr) {
        if (!g_owned) g_owned = select_runtime(DeviceKind::CUDA);
        g_ptr = g_owned.get();
    }
    return g_ptr;
}
}  // namespace

DeviceRuntime* active_runtime() {
    std::lock_guard<std::mutex> g(g_runtime_mutex);
    DeviceRuntime* rt = current_locked();
    if (!rt) throw DeviceError("no device backend is available (neither CUDA nor SYCL ran select_runtime)",
                               DeviceKind::CUDA, -1);
    return rt;
}

void set_active_runtime(std::unique_ptr<DeviceRuntime> rt) {
    std::lock_guard<std::mutex> g(g_runtime_mutex);
    g_owned = std::move(rt);   // drop the previous runtime before the new one is observable
    g_ptr = g_owned.get();
}

DeviceInfo device_info(int ordinal) {
    std::lock_guard<std::mutex> g(g_runtime_mutex);
    return current_locked()->device_info(ordinal);
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison)
    : capacity_(bytes), ordinal_(ordinal), poison_(poison) {
    DeviceRuntime* rt = active_runtime();
    if (bytes == 0) throw DeviceError("DeviceArena of 0 bytes", rt->kind(), -1);
    base_ = rt->device_alloc(bytes);
    if (base_ == nullptr) throw DeviceError("device_alloc failed for a DeviceArena of " +
                                                std::to_string(bytes) + " B", rt->kind(), -1);
    if (poison_) rt->poison(base_, bytes);
}

DeviceArena::~DeviceArena() {
    if (base_) active_runtime()->device_free(base_);   // best effort: a destructor must not throw
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    DeviceRuntime* rt = active_runtime();
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) {
        throw DeviceError("DeviceArena::alloc alignment must be a power of two", rt->kind(), -1);
    }
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw DeviceError(msg, rt->kind(), -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

void device_memcpy(void* stream, void* dst, const void* src, size_t bytes, MemcpyKind kind) {
    active_runtime()->memcpy(stream, dst, src, bytes, kind);
}

void device_synchronize(void* stream) { active_runtime()->synchronize(stream); }

}  // namespace strata::core
