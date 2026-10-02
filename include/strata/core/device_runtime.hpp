// src/core/device_runtime.hpp - the backend dispatch the device surface goes through (SYCLomatic port).
//
// device.hpp is a pure interface that names no backend.  This file is the matching implementation side: it
// builds the backends (`make_cuda_runtime` / `make_sycl_runtime` declared there), holds the one `active_runtime`
// the process chose, and contains the backend-agnostic bodies (the DeviceArena bump allocator, the copy and
// sync helpers) that read the active backend rather than naming it.
#pragma once

#include "strata/core/device.hpp"

namespace strata::core {

/// The backend the process is using.  Lazily created on first use (CUDA unless `select_runtime` set one), so
/// the many call sites that never touch a backend still get a working one.
DeviceRuntime* active_runtime();

/// The active backend's default/current launch queue, or nullptr when the active backend is not SYCL.
/// SYCL kernel files cast it back to `sycl::queue*` themselves, so keeping the return type `void*` means this
/// header still names no backend and `device_runtime.cpp` stays free of `<sycl.hpp>`.  It is what a kernel that
/// was launched on CUDA's null stream (default stream) uses on SYCL, so a `void* stream` that is null on one
/// build still finds a real queue on the other.
void* default_sycl_queue();

}  // namespace strata::core
