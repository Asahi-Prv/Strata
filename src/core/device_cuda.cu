// src/core/device_cuda.cpp - the CUDA backend (SYCLomatic port).
//
// Implements DeviceRuntime for CUDA.  This is what device.cu used to be, trimmed to the interface in
// device_runtime.hpp: no DeviceArena body lives here (that is backend-agnostic and stays in
// device_runtime.cpp), so swapping in device_sycl.cpp changes nothing above.  The poison kernel is the one
// piece of device code that has to stay in the backend, and it does.
#include "strata/core/device_runtime.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>

namespace strata::core {

namespace {

__global__ void poison_kernel(float* p, uint64_t n_floats) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_floats) p[i] = __int_as_float(0x7fc00000);   // NaN pattern, not zero
}

void check_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw DeviceError(std::string(what) + ": " + cudaGetErrorString(e), DeviceKind::CUDA, (int) e);
}

// A large region has to be poisoned in a loop because gridDim.x is 32-bit: 12 GB of floats is 3e9 elements,
// which is 1.2e7 blocks and fits, but the loop keeps it correct for any size rather than for today's sizes.
class CudaRuntime : public DeviceRuntime {
public:
    DeviceKind kind() const override { return DeviceKind::CUDA; }
    const char* name() const override { return "cuda"; }

    DeviceInfo device_info(int ordinal) override {
        int count = 0;
        check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count == 0) throw DeviceError("no CUDA device is present; Strata targets sm_120 (RTX 5000 series)", DeviceKind::CUDA, -1);
        if (ordinal < 0 || ordinal >= count) {
            throw DeviceError("device ordinal " + std::to_string(ordinal) + " is out of range (have " +
                                  std::to_string(count) + ")",
                              DeviceKind::CUDA, -1);
        }
        check_cuda(cudaSetDevice(ordinal), "cudaSetDevice");

        DeviceInfo d;
        d.ordinal = ordinal;
        cudaDeviceProp p{};
        check_cuda(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
        d.name = p.name;
        d.arch_name = "sm_" + std::to_string(p.major) + std::to_string(p.minor);
        d.compute_units = p.multiProcessorCount;
        check_cuda(cudaMemGetInfo(&d.free_bytes, &d.total_bytes), "cudaMemGetInfo");
        check_cuda(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
        return d;
    }

    void* device_alloc(uint64_t bytes) override {
        void* p = nullptr;
        check_cuda(cudaMalloc(&p, (size_t) bytes), "cudaMalloc");
        return p;
    }
    void device_free(void* p) override { if (p) cudaFree(p); }

    void* stream_create() override {
        cudaStream_t s = nullptr;
        check_cuda(cudaStreamCreate(&s), "cudaStreamCreate");
        return s;
    }
    void stream_destroy(void* s) override { if (s) cudaStreamDestroy((cudaStream_t) s); }

    void memcpy(void* stream, void* dst, const void* src, size_t bytes, MemcpyKind kind) override {
        cudaMemcpyKind k = cudaMemcpyDefault;
        switch (kind) {
            case MemcpyKind::HostToDevice: k = cudaMemcpyHostToDevice; break;
            case MemcpyKind::DeviceToHost: k = cudaMemcpyDeviceToHost; break;
            case MemcpyKind::DeviceToDevice: k = cudaMemcpyDeviceToDevice; break;
            default: k = cudaMemcpyDefault; break;
        }
        check_cuda(cudaMemcpyAsync(dst, src, bytes, k, (cudaStream_t) stream), "cudaMemcpyAsync");
    }
    void synchronize(void* stream) override { check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

    void poison(void* p, size_t bytes) override {
        const uint64_t n = bytes / sizeof(float);
        const int threads = 256;
        uint64_t blocks = (n + threads - 1) / threads;
        const uint64_t max_blocks = 0x7FFFFFFFull;
        for (uint64_t b = 0; b < blocks; b += max_blocks) {
            const uint64_t chunk = (blocks - b < max_blocks) ? (blocks - b) : max_blocks;
            check_cuda(cudaGetLastError(), "poison_kernel capture");
            poison_kernel<<<(unsigned) chunk, threads>>>((float*) p + b * threads, n - b * threads);
            check_cuda(cudaGetLastError(), "poison_kernel");
        }
        check_cuda(cudaDeviceSynchronize(), "poison sync");
    }
};

}  // namespace

std::unique_ptr<DeviceRuntime> make_cuda_runtime() {
    return std::make_unique<CudaRuntime>();
}

}  // namespace strata::core
