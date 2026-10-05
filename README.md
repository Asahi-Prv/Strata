This project(Strata CUDA-to-SYCL) owner is not have strata required resource.
My test scope is compile passed by icpx.

## How to build (SYCL / Intel GPU)

This fork ports the CUDA device layer to **oneAPI SYCL** for Intel GPUs. The ported
kernels live in `src/kernels/sycl/`; the CMake build selects the backend with a flag.

### Requirements

- **Intel oneAPI Base Toolkit + DPC++/C++ Compiler** (`icpx`) — tested with **2026.1**
- **CMake 3.24+** (the Ninja generator is recommended)
- An **Intel GPU** is only needed to *run*; **building does not need one**
- Roughly 10 GB of free disk for the build

### 1. Open a oneAPI environment (puts `icpx` and the SYCL runtime on `PATH`)

- **Windows:** run `setvars.bat` from the oneAPI install, or open the
  **"Intel oneAPI command prompt for Intel 64 Visual Studio"**.
- **Linux:** `source /opt/intel/oneapi/setvars.sh`

### 2. Configure — SYCL backend on, CUDA off (the two are mutually exclusive)

```
cmake -S . -B build -G Ninja -DSTRATA_ENABLE_SYCL=ON -DSTRATA_ENABLE_CUDA=OFF
```

`-G Ninja` is optional (any CMake generator works). Add `-DSTRATA_NATIVE_EXPERTS=OFF`
to skip fetching llama.cpp/ggml when the CPU i-quant experts are not needed.

### 3. Build

```
cmake --build build
```

On success the configure step prints

```
-- Strata: SYCL kernels built (Intel GPU); add each ported kernel to this library as it lands
```

and the build produces the ported kernel library:

- Windows: `build/strata_kernels_sycl.lib`
- Linux:   `build/libstrata_kernels_sycl.a`

Every ported kernel in `src/kernels/sycl/` is compiled with `-fsycl` by `icpx`.

### Status

- **Kernels:** every CUDA kernel has a SYCL port under `src/kernels/sycl/`, and
  the whole kernel library `strata_kernels_sycl` compiles and links with `icpx`.
- **Engine:** the full engine now **builds** on the SYCL backend. A
  `STRATA_ENABLE_SYCL=ON` configure produces `strata.exe` (plus `strata_core`,
  `strata_engine` and `strata_prefill` on oneMKL) through a CUDA-to-SYCL
  compatibility shim (`include/strata/core/sycl_compat/cuda_runtime.h`);
  `strata.exe` starts and runs its host setup.
- **Not verified end to end:** a full token generation needs the model pack
  (~38 GB) and a CPU with AVX-512 VNNI/VBMI (the engine's expert kernel requires
  them) - neither was available here - so generation on the Intel GPU has not been
  exercised. The runtime numerical correctness of the ported kernels is likewise
  not yet validated against the CUDA oracle.
- **Device backend (verified):** a SYCL build's `strata-device` runs and
  enumerates the real GPU through oneAPI SYCL -
  `device 0: Intel(R) Arc(TM) B570 Graphics, 160 compute units, 9.641 GiB, driver 100`.

## Credits

- Model: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team; compressed versions by
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI. Their licenses apply
  to the model files.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). Ideas from
  [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen). More in the [details](docs/DETAILS.md#credits-and-licenses).
