// src/kernels/sycl/mrope_table.cpp - host-side storage for the multimodal RoPE table (SYCLomatic port).
//
// native_rope.cu keeps an atomic const int32_t* that the engine sets (mrope_table_set) and every rope kernel
// reads (mrope_table). It is host bookkeeping - a device pointer stored on the host - so this is plain C++ and
// takes no -fsycl: there is no device work here. rope.cpp (and any later rope kernel) calls mrope_table() once
// per launch and captures the returned pointer into the parallel_for lambda, where the kernel reads it.
#include "strata/kernels/mrope.hpp"

#include <atomic>

namespace strata::kernels {

namespace {
std::atomic<const int32_t*> mrope_tab{nullptr};
}  // namespace

void mrope_table_set(const int32_t* device_table) { mrope_tab.store(device_table, std::memory_order_relaxed); }
const int32_t* mrope_table() { return mrope_tab.load(std::memory_order_relaxed); }

}  // namespace strata::kernels
