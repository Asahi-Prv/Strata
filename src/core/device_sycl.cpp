// src/core/device_sycl.cpp - the SYCL backend for Intel GPUs (SYCLomatic port).
//
// Implements DeviceRuntime on top of the oneAPI SYCL runtime, so the whole engine runs on an Intel Arc with
// only this file replacing device_cuda.cpp in the link line.  It mirrors CudaRuntime one-to-one: every method
// is the SYCL shape of the CUDA method above, and the interface (device_runtime.hpp) is identical, so
// device_runtime.cpp and the engine never change between the two.
//
// Compiled with -fsycl by the SYCL build; it includes only <sycl/sycl.hpp>, never a CUDA header.  It targets
// the SYCL 2020 API as shipped by the oneAPI DPC++/C++ compiler (2026.1): `sycl::` (not the retired `cl::sycl`),
// `sycl::exception` (not `sycl::Error`), a scalar `global_mem_size` and a `driver_version` that comes back as a
// string, which is why the shape looks like the original but the names have been modernized.
#include "strata/core/device_runtime.hpp"

#include <sycl/sycl.hpp>

#include <bit>
#include <cstdint>
#include <string>

namespace strata::core {

// Forward-declared at enclosing-scope so both queue_of() (a member, defined before it) and default_sycl_queue()
// (defined after the anonymous namespace closes) can name _default_queue(); a definition inside the anonymous
// namespace below would not be visible at those two call sites in this compiler.
static sycl::queue& _default_queue();

namespace {

    // Re-tag a SYCL failure with a prefix.  The old `Error(what, code)` spelling had the message first; the SYCL
    // 2020 `exception` takes (code, what), so the two arguments are swapped.
    sycl::exception to_sycl(const std::string& what, const sycl::exception& e) {
        return sycl::exception(e.code(), what + ": " + e.what());
    }

    // Forward declaration: queue_of() calls this after the class body.
    void qmemcpy_impl(sycl::queue& q, void* dst, const void* src, size_t bytes);

class SyclRuntime : public DeviceRuntime {
public:
    SyclRuntime() : q_(sycl::device{}) {}

    DeviceKind kind() const override { return DeviceKind::SYCL; }
    const char* name() const override { return "sycl"; }

    DeviceInfo device_info(int ordinal) override {
        // ORDINAL is a CUDA concept (a driver-side device number).  On SYCL the queue already picked its
        // device, so ordinal just selects WHICH queue to hand back a report for: the default one here, the
        // ordinal-th enumerated Intel device otherwise.  A fuller port enumerates them; this is the shape.
        const auto all = sycl::device::get_devices();
        sycl::device dev = (ordinal > 0 && ordinal < (int) all.size()) ? all[ordinal] : sycl::device{};
        sycl::queue q(dev);

        DeviceInfo d;
        d.ordinal = ordinal;
        d.name = dev.get_info<sycl::info::device::name>();
        // "Arwin64" / "Arwin" mirrors the CUDA build's sm_120: the address width is what parallels the
        // multiprocessor/EU count in the plan's arithmetic, and the Arc cards we target are 64-bit.
        d.arch_name = dev.get_info<sycl::info::device::address_bits>() == 64 ? "Arwin64" : "Arwin";
        // compute_units: EU count is what parallels the CUDA multiprocessor count in the plan's arithmetic.
        d.compute_units = dev.get_info<sycl::info::device::max_compute_units>();
        // global_mem_size is now a single total (not the old allocation range); there is no clean free-space
        // query, so report 0 and let the caller read the total.
        d.total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
        d.free_bytes = 0;
        // SYCL exposes driver version as a string (e.g. "2026.1"); the struct keeps driver_version an int,
        // so take the leading dotted number's major*100 - it is only a report field, not used in any math.
        const auto ver = dev.get_info<sycl::info::device::driver_version>();
        int major = 0;
        for (char c : ver) {
            if (c >= '0' && c <= '9') { major = major * 10 + (c - '0'); }
            else if (major != 0) { break; }
        }
        d.driver_version = major * 100;
        return d;
    }

    void* device_alloc(uint64_t bytes) override {
        void* p = sycl::malloc_device((size_t) bytes, q_);
        if (!p) throw DeviceError("sycl::malloc_device failed for " + std::to_string(bytes) + " B", DeviceKind::SYCL, -1);
        return p;
    }
    void device_free(void* p) override { if (p) sycl::free(p, q_); }

    void* stream_create() override { return new sycl::queue(); }
    void stream_destroy(void* s) override { if (s) delete static_cast<sycl::queue*>(s); }

    void memcpy(void* stream, void* dst, const void* src, size_t bytes, MemcpyKind kind) override {
        (void) kind;
        qmemcpy_impl(queue_of(stream), dst, src, bytes);
    }
    void synchronize(void* stream) override { queue_of(stream).wait(); }

    void poison(void* p, size_t bytes) override {
        const uint64_t n = bytes / sizeof(float);
        // One element per work-item with a bounds check: the same shape as the CUDA poison loop and correct
        // for any size.  0x7fc00000 is the bit pattern of +inf-as-NaN-ish filler, not zero.
        try {
            q_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    static_cast<float*>(p)[i] = sycl::bit_cast<float, unsigned int>(0x7fc00000u);
                });
            });
            q_.wait();
        } catch (const sycl::exception& e) { throw to_sycl("poison_kernel", e); }
    }

private:
    // The null stream is the default queue (CUDA) / current queue (SYCL), both represented as nullptr on the
    // wire.  Anything non-null is a queue we created, so cast it back.
    static sycl::queue& queue_of(void* stream) {
        return stream ? *static_cast<sycl::queue*>(stream) : _default_queue();
    }
    sycl::queue q_;
};

void qmemcpy_impl(sycl::queue& q, void* dst, const void* src, size_t bytes) {
    q.submit([&](sycl::handler& h) { h.memcpy(dst, src, bytes); });
}

}  // namespace

static sycl::queue& _default_queue() {
    static sycl::queue q_(sycl::device{});
    return q_;
}

std::unique_ptr<DeviceRuntime> make_sycl_runtime() {
    return std::make_unique<SyclRuntime>();
}

/// The process-wide backend selector (`device.hpp`).  `device_runtime.cpp` is backend-agnostic and calls
/// `select_runtime` to obtain the one runtime; the CUDA backend would answer `make_cuda_runtime()`.  In a SYCL
/// build only this backend is linked, so any requested kind resolves to the SYCL runtime.
std::unique_ptr<DeviceRuntime> select_runtime(DeviceKind) {
    return make_sycl_runtime();
}

/// The default queue for kernels launched on a null stream.  Defined outside the anonymous namespace so it has
/// external linkage (SYCL kernel files link to it); it just hands back the same static queue the backend uses.
void* default_sycl_queue() {
    sycl::queue& q = _default_queue();
    return static_cast<void*>(&q);
}

}  // namespace strata::core
