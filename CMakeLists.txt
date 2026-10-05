# Strata Engine - top-level build (P1.S1).
#
# Targets (phase-1 Ă‚Â§P1.S1): `strata_artifact` (no CUDA dependency), `tests`.  `strata_plan` and the C++
# `strata-pack` are added when their sources exist - a target with no sources would make the build graph
# claim something the tree does not have, which is worse than an absent target.
#
# CUDA targets arrive in Phase 2 and this build REJECTS any architecture other than 120: the engine is
# written against sm_120 (RTX 5000 / Blackwell) and silently building for an older arch would produce a
# binary that runs and is wrong about its own performance, which is the worst of both outcomes.
cmake_minimum_required(VERSION 3.24)
project(strata VERSION 0.1.0 LANGUAGES CXX)

# The prebuilt release (setup.py downloads it, so users need no compiler): ggml's CPU backend for any AVX2 CPU
# instead of this machine's (Strata's own AVX-512 kernels are per-file and chosen at run time either way), and
# the MSVC runtime linked in, so no Visual C++ redistributable is needed.
option(STRATA_PORTABLE "build for other PCs: AVX2 baseline, static MSVC runtime" OFF)
if(STRATA_PORTABLE AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(STRATA_ENABLE_CUDA "Build the CUDA targets (Phase 2+)" OFF)
# SYCLomatic port: the device layer has a second backend (Intel GPU).  The build graph is identical on both
# sides - only the two device-layer implementations swap - so this is mutually exclusive with CUDA, not a
# parallel target.  When ON it is selected even where the default would be CUDA.
option(STRATA_ENABLE_SYCL "Build the device layer on SYCL (Intel GPU) instead of CUDA" OFF)
# The test suite and the micro benchmarks live in tests/ and bench/micro/, which the published source leaves out.
set(_strata_tests_default OFF)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/micro)
  set(_strata_tests_default ON)
endif()
option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

# Two backends, one device layer: choosing both would link device_cuda.cu AND device_sycl.cpp into the same
# strata_core, which is two DeviceRuntime implementations and duplicate symbols.  Fail early and loudly.
if(STRATA_ENABLE_SYCL AND STRATA_ENABLE_CUDA)
  message(FATAL_ERROR "STRATA_ENABLE_SYCL and STRATA_ENABLE_CUDA are mutually exclusive - pick the device backend once.")
endif()

if(STRATA_ENABLE_CUDA)
  enable_language(CUDA)
  # `enable_language(CUDA)` gives the .cu compile rules but NOT the imported targets, so `CUDA::cudart` has to
  # be brought in explicitly for the host-side tests that call cudaMalloc themselves.
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
  foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "-.*$" "" _base "${_a}")     # 120-real / 120-virtual are still 120
    # Plan v0.3 P6: developed and measured on sm_120 (RTX 50); the kernels need sm_80 or newer (tf32 mma in
    # the attention scorer, bf16 math), so RTX 30 (86) and RTX 40 (89) build too.  Older cards are refused.
    if(_base STREQUAL "native" OR _base STREQUAL "all" OR _base STREQUAL "all-major")
      continue()
    endif()
    if(_base LESS 80)
      message(FATAL_ERROR
        "Strata needs an NVIDIA GPU of compute capability 8.0 or newer (RTX 30 / 40 / 50 series), but "
        "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}'.")
    endif()
  endforeach()
  message(STATUS "Strata: CUDA enabled, arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Warnings live on an INTERFACE target the project's own code links, NOT in add_compile_options: applied
# globally they would also hit FetchContent'd Catch2, whose -W4 warnings are not this project's to fix and
# whose /WX failures would be indistinguishable from our own.
#
# Every flag is guarded by COMPILE_LANGUAGE:CXX.  Without that guard nvcc receives bare `/W4 /utf-8` and dies
# with "A single input file is required for a non-link phase" - it parses them as inputs rather than host
# flags.  Host flags for CUDA must go through -Xcompiler, which is what COMPILE_LANGUAGE:CUDA does below.
add_library(strata_warnings INTERFACE)
target_compile_options(strata_warnings INTERFACE
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wextra>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/W4>)
if(STRATA_WERROR)
  target_compile_options(strata_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/WX>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Werror>)
endif()

# ---- strata_artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(strata_artifact INTERFACE)
target_include_directories(strata_artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_artifact INTERFACE strata_warnings)

add_executable(strata-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(strata-gguf PRIVATE strata_artifact)

add_executable(strata-dequant src/artifact/dequant.cpp)
target_link_libraries(strata-dequant PRIVATE strata_artifact)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(strata-dequant PRIVATE STRATA_DEQUANT_SELFTEST)

# ---- strata_plan ----------------------------------------------------------
# Also header-only.  A "planner" that hides state would be one whose decision is not inspectable, and the
# whole point of P1.S9 is that the decision is arithmetic you can check by hand.
add_library(strata_plan INTERFACE)
target_include_directories(strata_plan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_plan INTERFACE strata_warnings)

add_executable(strata-plan src/plan/plan_main.cpp)
target_link_libraries(strata-plan PRIVATE strata_plan)

# ---- strata_core (Phase 2) --------------------------------------------------
# Only built when a backend is on, so the Phase 1 targets keep working on a machine with no toolkit.  The arch
# guard above has already refused anything but sm_120 by the time this runs (CUDA); on SYCL the device is
# validated in device_sycl::SyclRuntime::device_info instead.
#
# The device layer has TWO backends that share one interface (device_runtime.hpp):
#   * device_cuda.cu  - the original backend, used by the default CUDA build,
#   * device_sycl.cpp - the Intel-GPU backend, selected when STRATA_ENABLE_SYCL is ON and compiled with -fsycl.
# The backend-agnostic body (the DeviceArena bump allocator, the copy/sync helpers) lives in device_runtime.cpp
# and names NEITHER backend, so the only thing this block swaps between the two is the single backend source.
#
# NOTE (SYCLomatic port): pinned.cu and graph.cpp still name CUDA (cudaHostRegister / cudaGraph).  They are
# ported in the same way - one backend implementation each behind the same wire - but that is the follow-on
# work; until it lands a SYCL build must still pass the CUDA toolkit so these two stay .cu.  The device layer
# itself is already backend-exchangeable.
if(STRATA_ENABLE_CUDA)
  if(STRATA_ENABLE_SYCL)
    set(_strata_core_backend src/core/device_sycl.cpp)
    set_source_files_properties(src/core/device_sycl.cpp PROPERTIES COMPILE_OPTIONS "-fsycl")
    message(STATUS "Strata: device layer on SYCL (Intel GPU)")
  else()
    set(_strata_core_backend src/core/device_cuda.cu)
  endif()
  add_library(strata_core STATIC ${_strata_core_backend} src/core/device_runtime.cpp
              src/core/pinned.cu src/platform/memory.cpp src/core/graph.cpp src/core/weights.cpp
              src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # CUDA::cudart is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
  # cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
  # linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings CUDA::cudart)

  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  add_executable(strata-load src/core/load_main.cpp)
  target_link_libraries(strata-load PRIVATE strata_core strata_kernels)

  add_executable(strata-concurrent src/core/concurrent_main.cpp)
  target_include_directories(strata-concurrent PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata-concurrent PRIVATE strata_core strata_kernels CUDA::cudart)

  add_executable(strata-overlap src/core/overlap_main.cpp)
  # a .cpp that calls cudaMemcpyAsync itself needs the toolkit headers and cudart explicitly
  target_include_directories(strata-overlap PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata-overlap PRIVATE strata_core strata_kernels CUDA::cudart)

  # The load test needs the 34 GB artifact, so it is registered ONLY where that artifact exists.  A test that
  # silently skips is a test that stops running; an absent registration is visible in the configure output.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin)
    add_test(NAME expert_arena_load
             COMMAND strata-load --file ${CMAKE_CURRENT_SOURCE_DIR}/pack/full/experts.bin --layers 1)
    set_tests_properties(expert_arena_load PROPERTIES PASS_REGULAR_EXPRESSION "per-layer checksums")
  else()
    message(STATUS "strata: pack/full/experts.bin absent - the expert arena load test is NOT registered")
  endif()

  # ---- strata_kernels (P2.S2) ----
  # Naive first, each with a parity test against a reference that was itself validated against ggml.  The
  # kernels link `strata_artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
  # reference is the whole point - a kernel checked against another copy of itself proves nothing.
  add_library(strata_kernels STATIC src/kernels/cuda/dequant_s2.cu src/kernels/cuda/s2_gemv.cu
                          src/kernels/cuda/s_gemv.cu
                          src/kernels/cuda/s2_gemv_quads.cu
                          src/kernels/cuda/s2_gemv_fast.cu
                          src/kernels/cuda/router_top10.cu
                          src/kernels/cuda/quantize_act.cu
                          src/kernels/cuda/rope.cu
                          src/kernels/cuda/sampler.cu
                          src/kernels/cuda/s2_gemv_q8.cu
    src/kernels/cuda/s2_expert_grouped.cu
                          src/kernels/cuda/shared_expert.cu
                          src/kernels/cuda/gr.cu
                          src/kernels/cuda/gdn.cu
                          src/kernels/cuda/qsa.cu
                          src/kernels/cuda/elementwise.cu
                          src/kernels/cuda/bf16_gemv.cu
                          src/kernels/cuda/native_mmvq.cu
                          src/kernels/cuda/native_bf16.cu
                          src/kernels/cuda/native_gr_norm.cu
                          src/kernels/cuda/native_moe.cu
                          src/kernels/cuda/native_gdn.cu
                          src/kernels/cuda/native_gr_postops.cu
                          src/kernels/cuda/native_router.cu
                          src/kernels/cuda/native_gdn_preprocess.cu
                          src/kernels/cuda/native_qsa.cu
                          src/kernels/cuda/native_rope.cu
                          src/kernels/cuda/native_ple_postops.cu
                          src/kernels/cuda/native_flash_attn.cu
                          src/kernels/cuda/native_qsa_indexer.cu
                          src/kernels/cuda/native_qsa_score.cu
                          src/kernels/cuda/ple.cu
                          src/kernels/cuda/kv_q8.cu
                          src/kernels/cuda/kv_stream.cu
                          src/kernels/cuda/kv_q4.cu
                          src/kernels/cuda/fused_gr.cu
                          src/kernels/cuda/qsa_decode_attn.cu
                          src/kernels/cuda/fused_gdn.cu
                          src/kernels/cuda/dequant_bf16.cu
                          src/kernels/cuda/qsa_select.cu
                          src/kernels/cuda/verify_kernels.cu
                          src/kernels/cuda/iq_kernels.cu
                          src/kernels/ngram.cpp
                          src/ngram/ple_reader.cpp
                          src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels PUBLIC strata_core strata_artifact)
  # plan v0.3 P6: llama.cpp's block layouts and codebook grids for the i-quant formats (MIT, third_party/ggml)
  target_include_directories(strata_kernels PRIVATE third_party/ggml)
  # Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
  set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/native_gr_norm.cu
    src/kernels/cuda/native_moe.cu
    src/kernels/cuda/native_gdn.cu
    src/kernels/cuda/native_gr_postops.cu
    src/kernels/cuda/native_router.cu
    src/kernels/cuda/native_gdn_preprocess.cu
    src/kernels/cuda/native_qsa.cu
    src/kernels/cuda/native_rope.cu
    src/kernels/cuda/native_ple_postops.cu
    src/kernels/cuda/native_flash_attn.cu
    src/kernels/cuda/native_qsa_indexer.cu
    src/kernels/cuda/native_qsa_score.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")

  set(STRATA_ORACLE_SOURCE_DIR "" CACHE PATH "Pinned llama.cpp source for independent CUDA parity tests")
  set(STRATA_ORACLE_BUILD_DIR "" CACHE PATH "Matching existing shared llama.cpp build")
  if(STRATA_ORACLE_SOURCE_DIR AND STRATA_ORACLE_BUILD_DIR)
    foreach(parity IN ITEMS native_mmvq native_bf16 native_shared native_gr_norm native_moe native_gdn native_gr_postops native_router native_gdn_preprocess native_gdn_pipeline native_qsa native_rope native_ple_postops native_flash_attn native_qsa_indexer native_qsa_score)
    add_executable(${parity}_parity bench/micro/${parity}_parity.cpp)
    target_include_directories(${parity}_parity PRIVATE
      "${STRATA_ORACLE_SOURCE_DIR}/ggml/include" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(${parity}_parity PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${parity}_parity PRIVATE strata_kernels CUDA::cudart)
    foreach(component IN ITEMS ggml ggml-base ggml-cpu ggml-cuda)
      string(REPLACE "-" "_" variable "STRATA_ORACLE_${component}")
      find_library(${variable} NAMES ${component} PATHS
        "${STRATA_ORACLE_BUILD_DIR}/ggml/src" "${STRATA_ORACLE_BUILD_DIR}/ggml/src/${component}"
        "${STRATA_ORACLE_BUILD_DIR}/bin" NO_DEFAULT_PATH REQUIRED)
      target_link_libraries(${parity}_parity PRIVATE "${${variable}}")
    endforeach()
    add_test(NAME ${parity}_parity COMMAND ${parity}_parity --selftest)
    if(WIN32)
      set_tests_properties(${parity}_parity PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${STRATA_ORACLE_BUILD_DIR}/bin")
    endif()
    endforeach()
  endif()

  # ---- strata_engine (P2.S5) -----------------------------------------------------
  # The COMPOSED layer lives here and not in `strata_core`, because it calls kernels and `strata_kernels`
  # already links `strata_core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
  # library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
  # about arithmetic, and this one knows the ORDER the arithmetic happens in.
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu CUDA::cudart)

  # `strata generate` (P2.S6).  The FIRST user-facing program: everything under it is a component, and this is
  # what composes them into a token.  No tokenizer - the prompt arrives as ids - because Gate C1 compares logits
  # against llama.cpp on the SAME ids and a tokenizer on one side of that comparison is a second variable.
  # plan v0.3 P5: batched prompt processing (tensor-core GEMMs through cuBLAS).
  add_library(strata_prefill STATIC src/prefill/gemm.cu src/prefill/kernels.cu src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata_prefill PUBLIC strata_engine CUDA::cublas CUDA::cudart)

  add_executable(strata src/program/generate.cpp)
  target_include_directories(strata PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec CUDA::cudart)

  add_executable(dequant_s2_parity src/kernels/dequant_s2_parity.cpp)
  # The tests are HOST C++ files that call cudaMalloc/cudaMemcpy themselves, so they need the toolkit headers
  # and cudart explicitly: enabling the CUDA language gives those to .cu sources automatically, but a .cpp
  # target is just a C++ target and gets neither.
  target_include_directories(dequant_s2_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(dequant_s2_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME dequant_s2_parity COMMAND dequant_s2_parity --selftest)

  add_executable(s2_gemv_parity src/kernels/s2_gemv_parity.cpp)
  target_include_directories(s2_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s2_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s2_gemv_parity COMMAND s2_gemv_parity --selftest)

  add_executable(iq_parity src/kernels/iq_parity.cpp)
  target_include_directories(iq_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(iq_parity PRIVATE strata_kernels CUDA::cudart)
  add_executable(shared_expert_parity src/kernels/shared_expert_parity.cpp)
  target_include_directories(shared_expert_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(shared_expert_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME shared_expert_parity COMMAND shared_expert_parity --selftest)

  add_executable(gr_parity src/kernels/gr_parity.cpp)
  target_include_directories(gr_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(gr_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME gr_parity COMMAND gr_parity --selftest)

  add_executable(gdn_parity src/kernels/gdn_parity.cpp)
  target_include_directories(gdn_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(gdn_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME gdn_parity COMMAND gdn_parity --selftest)

  add_executable(s2_gemv_q8_parity src/kernels/s2_gemv_q8_parity.cpp)
  target_include_directories(s2_gemv_q8_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s2_gemv_q8_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s2_gemv_q8_parity COMMAND s2_gemv_q8_parity --selftest)

  add_executable(sampler_parity src/kernels/sampler_parity.cpp)
  target_include_directories(sampler_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(sampler_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME sampler_parity COMMAND sampler_parity --selftest)

  add_executable(rope_parity src/kernels/rope_parity.cpp)
  target_include_directories(rope_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(rope_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME rope_parity COMMAND rope_parity --selftest)

  add_executable(quantize_act_parity src/kernels/quantize_act_parity.cpp)
  target_include_directories(quantize_act_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(quantize_act_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

  add_executable(router_top10_parity src/kernels/router_top10_parity.cpp)
  target_include_directories(router_top10_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(router_top10_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME router_top10_parity COMMAND router_top10_parity --selftest)

  add_executable(s_gemv_parity src/kernels/s_gemv_parity.cpp)
  target_include_directories(s_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s_gemv_parity COMMAND s_gemv_parity --selftest)

  add_executable(elementwise_parity src/kernels/elementwise_parity.cpp)
  target_include_directories(elementwise_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(elementwise_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

  add_executable(bf16_gemv_parity src/kernels/bf16_gemv_parity.cpp)
  target_include_directories(bf16_gemv_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(bf16_gemv_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

  add_executable(s_gemv_q8k_parity src/kernels/s_gemv_q8k_parity.cpp)
  target_include_directories(s_gemv_q8k_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(s_gemv_q8k_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME s_gemv_q8k_parity COMMAND s_gemv_q8k_parity --selftest)

  add_executable(qsa_parity src/kernels/qsa_parity.cpp)
  target_include_directories(qsa_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(qsa_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME qsa_parity COMMAND qsa_parity --selftest)

  # ---- P2.S4: the PLE n-gram hash, the IQ4_NL table read, and the PLE block ----
  # `ngram.cpp` is host-only but lives in `strata_kernels` because `ple.cu` shares its geometry constants, and
  # splitting them across two libraries would only add a layer.
  #
  # The pack, the original GGUF and the ggml capture are all REQUIRED inputs: this test's oracles are real
  # artifacts, so the working directory is pinned to the source tree and a missing input is a hard error
  # INSIDE the test.  A test that passes by finding nothing to compare is the failure mode this project keeps
  # re-learning.
  add_executable(ple_parity src/kernels/ple_parity.cpp)
  target_include_directories(ple_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(ple_parity PRIVATE strata_kernels CUDA::cudart)
  set(STRATA_PLE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bench/micro" CACHE PATH
      "Directory containing required ple_in.bin and ple_out.bin parity fixtures")
  add_test(NAME ple_parity COMMAND ple_parity --selftest
    --in "${STRATA_PLE_FIXTURE_DIR}/ple_in.bin" --out "${STRATA_PLE_FIXTURE_DIR}/ple_out.bin"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  # Plan v0.3 P7: INT8 KV storage vs a host reference and vs the FP16 path (GPU, synthetic, no model).
  add_executable(kv_q8_parity src/kernels/kv_q8_parity.cpp)
  target_include_directories(kv_q8_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(kv_q8_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME kv_q8_parity COMMAND kv_q8_parity)

  # KV streaming (docs/kv-streaming-design.md): streamed vs resident attention, bitwise (GPU, synthetic, no model).
  add_executable(kv_stream_parity src/kernels/kv_stream_parity.cpp)
  target_include_directories(kv_stream_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(kv_stream_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME kv_stream_parity COMMAND kv_stream_parity)

  # PR #21: Q4_0 KV with Walsh-Hadamard rotation vs a host reference (GPU, synthetic, no model).
  add_executable(kv_q4_parity src/kernels/kv_q4_parity.cpp)
  target_include_directories(kv_q4_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(kv_q4_parity PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME kv_q4_parity COMMAND kv_q4_parity)

  # Plan v0.3 P3: native MMVQ for 1..8 tokens (bitwise selftest + --bench timing); synthetic data, no model.
  if(STRATA_BUILD_TESTS)
  add_executable(native_mmvq_multi bench/micro/native_mmvq_multi.cpp)
  target_include_directories(native_mmvq_multi PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(native_mmvq_multi PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME native_mmvq_multi COMMAND native_mmvq_multi --selftest)
  endif()

  # Plan v0.3 P0.1: working-set lock of a host region (CPU only).
  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_link_libraries(platform_memory_test PRIVATE strata_core CUDA::cudart)
  add_test(NAME platform_memory_test COMMAND platform_memory_test)

  # Plan v0.3 P2: the n-gram table read straight from the SSD. The selftest needs only the CPU and a temp file.
  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_include_directories(ple_reader_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(ple_reader_test PRIVATE strata_kernels CUDA::cudart)
  add_test(NAME ple_reader_selftest COMMAND ple_reader_test --selftest)

  # The GPU test is registered only when CUDA is on, and it runs ON THE TARGET PC.  `--selftest` allocates
  # and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
  # which is correct: an engine that targets one GPU should not report a pass for having no GPU.
  add_test(NAME cuda_device_selftest COMMAND strata-device --selftest)

  # ---- P2.S5: the GraphRegistry --------------------------------------------------
  # A .cu because the test needs real kernels to capture; it links `strata_core` and not `strata_kernels`,
  # because what it exercises is the capture machinery and not any kernel in it.
  if(STRATA_BUILD_TESTS)
  add_executable(graph_registry_test bench/micro/graph_registry_test.cu)
  target_include_directories(graph_registry_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(graph_registry_test PRIVATE strata_core CUDA::cudart)
  add_test(NAME graph_registry_test COMMAND graph_registry_test)

  # ---- P2.S5's GATE: is a device<->pinned-memory handoff graph-capturable? --------------------
  # The doorbell protocol puts the D2H of `x_f` and the miss list INSIDE G[l] - that is the whole overlap.  The
  # phase doc says to verify this "before building on it", and this project has already lost two rounds to
  # building first (a memory-only spin that DEADLOCKS on Windows, and a `cudaEventRecord` inside a capture that
  # is SILENTLY DROPPED).  It links only cudart: it is about the capture machinery, not about any kernel.
  add_executable(pinned_capture bench/micro/pinned_capture.cu)
  target_include_directories(pinned_capture PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(pinned_capture PRIVATE CUDA::cudart)
  add_test(NAME pinned_capture COMMAND pinned_capture)
  endif()

  # Tells tests/CMakeLists.txt to add the CUDA-dependent Catch2 target.  It cannot be declared there
  # unconditionally: tests/ is added at the end of this file, but the flag has to be known before it runs.
  set(STRATA_CUDA_TESTS ON)
endif()

# ---- strata_kernels_cpu (P2.S3) ----
#
# A SEPARATE LIBRARY from `strata_kernels`, and `/arch:AVX512` is set on the SOURCE rather than on the target
# so that the flag applies only to the TU that needs it.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI); without the flag MSVC reports undeclared intrinsics and never mentions AVX-512;
#   * a TU compiled WITH `/arch:AVX512` may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests, exactly as P2.S3 specifies.
# Plan v0.3 P6: speculation components that need no GPU.
add_library(strata_spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(strata_spec PUBLIC include)
if(STRATA_BUILD_TESTS)
  add_executable(suffix_drafter_test src/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE strata_spec)
  add_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test src/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE strata_spec)
  add_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test src/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE strata_spec)
  add_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()

add_library(strata_kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
target_include_directories(strata_kernels_cpu PUBLIC include)
# Plan v0.3 P6: only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# MSVC spells the AVX-512 target `/arch:AVX512`, but the oneAPI icx driver is an LLVM/Clang front end and
# rejects it (it wants the GNU `-mavx512*` form) even though it is cl-compatible enough for CMake's MSVC
# variable to be set on Windows.  Route IntelLLVM/Clang/GNU to the `else()` arm below.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang|GNU")
  set_source_files_properties(src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  # GCC/Clang spell it differently, and `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
  set_source_files_properties(src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()
find_package(Threads REQUIRED)
target_link_libraries(strata_kernels_cpu PUBLIC Threads::Threads)

# ---- plan v0.3 P6: the i-quant models' CPU experts (IQ2_XS / IQ3_XXS files) ----
#
# The CPU computes a native expert with ggml-cpu's own dot products (`ggml_get_type_traits_cpu`), so an IQ
# weight means exactly what it means in llama.cpp.  ggml is built static from a llama.cpp checkout: a local one
# with -DSTRATA_GGML_DIR=..., otherwise the pinned upstream commit (third_party/ggml/VERSION.txt).
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")
if(STRATA_NATIVE_EXPERTS)
  enable_language(C)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_STATIC ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  if(STRATA_PORTABLE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
      set(${_f} ON CACHE BOOL "" FORCE)
    endforeach()
  else()
    set(GGML_NATIVE ON CACHE BOOL "" FORCE)
  endif()
  if(STRATA_GGML_DIR)
    add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  else()
    include(FetchContent)
    FetchContent_Declare(strata_llamacpp
      GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
      GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
      GIT_SHALLOW FALSE)
    FetchContent_GetProperties(strata_llamacpp)
    if(NOT strata_llamacpp_POPULATED)
      FetchContent_Populate(strata_llamacpp)
    endif()
    add_subdirectory(${strata_llamacpp_SOURCE_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
  endif()
  # part of strata_kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
  # one-pass static linking has no cycle to resolve)
  target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp)
  target_include_directories(strata_kernels_cpu PRIVATE third_party/ggml)
  if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM|Clang|GNU")
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  endif()
  target_link_libraries(strata_kernels_cpu PUBLIC ggml-cpu ggml-base)
  target_compile_definitions(strata_kernels_cpu PUBLIC STRATA_NATIVE_EXPERTS=1)
  if(STRATA_ENABLE_CUDA)
    add_executable(native_expert_parity src/kernels/native_expert_parity.cpp)
    target_include_directories(native_expert_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(native_expert_parity PRIVATE strata_kernels strata_kernels_cpu strata_artifact CUDA::cudart)
  endif()
endif()

if(STRATA_ENABLE_CUDA AND STRATA_BUILD_TESTS)
  add_executable(hit_cpu_order_parity bench/micro/hit_cpu_order_parity.cu)
  target_link_libraries(hit_cpu_order_parity PRIVATE strata_kernels strata_kernels_cpu CUDA::cudart)
endif()

if(STRATA_BUILD_TESTS)
  # Plan v0.3 P6: one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # plan v0.3 P5: the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the 38 GB model; run it by hand or from tools/gpu_session.py)
  add_executable(dequant_bf16_test src/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE strata_kernels strata_artifact CUDA::cudart)

  add_executable(expert_multi_test src/kernels/cpu/expert_multi_test.cpp)
  target_link_libraries(expert_multi_test PRIVATE strata_kernels_cpu)
  add_test(NAME expert_multi_test COMMAND expert_multi_test)

  add_executable(expert_parity src/kernels/cpu/expert_parity.cpp)
  target_link_libraries(expert_parity PRIVATE strata_kernels_cpu)
  # The pack is a REQUIRED input, not an optional one: this test compares against a real expert blob, and a
  # missing file would otherwise be reported as a pass by a test that skipped everything.
  add_test(NAME expert_parity COMMAND expert_parity --selftest
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

   add_executable(pool_test src/kernels/cpu/pool_test.cpp)
   target_link_libraries(pool_test PRIVATE strata_kernels_cpu)
   add_test(NAME pool_test COMMAND pool_test --selftest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
 endif()

# ---- SYCLomatic port (Intel GPU): the kernels, backend-swapped under STRATA_ENABLE_SYCL -----------------
#
# The mirror of strata_kernels.  Only built on a SYCL configuration (STRATA_ENABLE_SYCL ON, CUDA OFF): each
# ported .cpp here replaces its .cu in the engine's kernel library.  It starts with the EASIEST kernel (rope -
# one thread per row, no warp ops, no shared memory) and grows as each .cu is ported down the list.  The
# engine link target that picks this up instead of strata_kernels is the follow-on work (strata_engine /
# strata_prefill / the parity harnesses all need the same swap), so this library stands on its own for now:
# it compiles and its parity test can be registered there.
if(STRATA_ENABLE_SYCL AND NOT STRATA_ENABLE_CUDA)
  add_library(strata_kernels_sycl STATIC
              src/kernels/sycl/rope.cpp
              src/kernels/sycl/dequant_s2.cpp
              src/kernels/sycl/dequant_q8_0.cpp
               src/kernels/sycl/elementwise.cpp
               src/kernels/sycl/quantize_q8_0.cpp
                src/kernels/sycl/bf16_gemv.cpp
                src/kernels/sycl/s_gemv.cpp
                 src/kernels/sycl/s2_gemv.cpp
                 src/kernels/sycl/quantize_act.cpp
                 src/kernels/sycl/dequant_bf16.cpp
                 src/kernels/sycl/native_moe.cpp
                  src/kernels/sycl/native_rope.cpp
                  src/kernels/sycl/native_gr_postops.cpp
                  src/kernels/sycl/native_gr_norm.cpp
                  src/kernels/sycl/s2_gemv_quads.cpp
                  src/kernels/sycl/s2_gemv_q8.cpp
                  src/kernels/sycl/s2_gemv_fast.cpp
                   src/kernels/sycl/native_bf16.cpp
                   src/kernels/sycl/native_qsa.cpp
                   src/kernels/sycl/native_qsa_indexer.cpp
                   src/kernels/sycl/native_qsa_score.cpp
                   src/kernels/sycl/native_ple_postops.cpp
                   src/kernels/sycl/fused_gdn.cpp
                   src/kernels/sycl/fused_gr.cpp
                   src/kernels/sycl/gdn.cpp
                   src/kernels/sycl/gr.cpp
                   src/kernels/sycl/iq_kernels.cpp
                   src/kernels/sycl/kv_q4.cpp
                   src/kernels/sycl/kv_q8.cpp
                   src/kernels/sycl/kv_stream.cpp
                   src/kernels/sycl/native_flash_attn.cpp
                   src/kernels/sycl/native_gdn.cpp
                   src/kernels/sycl/native_gdn_preprocess.cpp
                   src/kernels/sycl/native_mmvq.cpp
                   src/kernels/sycl/native_router.cpp
                   src/kernels/sycl/ple.cpp
                   src/kernels/sycl/qsa.cpp
                   src/kernels/sycl/qsa_decode_attn.cpp
                   src/kernels/sycl/qsa_select.cpp
                   src/kernels/sycl/router_top10.cpp
                   src/kernels/sycl/s2_expert_grouped.cpp
                   src/kernels/sycl/sampler.cpp
                   src/kernels/sycl/shared_expert.cpp
                   src/kernels/sycl/verify_kernels.cpp
                   src/kernels/sycl/mrope_table.cpp)
  target_include_directories(strata_kernels_sycl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(strata_kernels_sycl PRIVATE third_party/ggml)
  # -fsycl is the whole SYCL build: the whole strata_kernels_sycl and strata_core get it, per-file so it does
  # not leak onto the host-only TUs that stay plain C++.
   set_source_files_properties(src/kernels/sycl/rope.cpp src/kernels/sycl/dequant_s2.cpp src/kernels/sycl/dequant_q8_0.cpp src/kernels/sycl/elementwise.cpp src/kernels/sycl/quantize_q8_0.cpp src/kernels/sycl/bf16_gemv.cpp src/kernels/sycl/s_gemv.cpp src/kernels/sycl/s2_gemv.cpp src/kernels/sycl/quantize_act.cpp src/kernels/sycl/dequant_bf16.cpp src/kernels/sycl/native_moe.cpp src/kernels/sycl/native_rope.cpp src/kernels/sycl/native_gr_postops.cpp src/kernels/sycl/native_gr_norm.cpp src/kernels/sycl/s2_gemv_quads.cpp src/kernels/sycl/s2_gemv_q8.cpp src/kernels/sycl/s2_gemv_fast.cpp src/kernels/sycl/native_bf16.cpp src/kernels/sycl/native_qsa.cpp src/kernels/sycl/native_qsa_indexer.cpp src/kernels/sycl/native_qsa_score.cpp src/kernels/sycl/native_ple_postops.cpp src/kernels/sycl/fused_gdn.cpp src/kernels/sycl/fused_gr.cpp src/kernels/sycl/gdn.cpp src/kernels/sycl/gr.cpp src/kernels/sycl/iq_kernels.cpp src/kernels/sycl/kv_q4.cpp src/kernels/sycl/kv_q8.cpp src/kernels/sycl/kv_stream.cpp src/kernels/sycl/native_flash_attn.cpp src/kernels/sycl/native_gdn.cpp src/kernels/sycl/native_gdn_preprocess.cpp src/kernels/sycl/native_mmvq.cpp src/kernels/sycl/native_router.cpp src/kernels/sycl/ple.cpp src/kernels/sycl/qsa.cpp src/kernels/sycl/qsa_decode_attn.cpp src/kernels/sycl/qsa_select.cpp src/kernels/sycl/router_top10.cpp src/kernels/sycl/s2_expert_grouped.cpp src/kernels/sycl/sampler.cpp src/kernels/sycl/shared_expert.cpp src/kernels/sycl/verify_kernels.cpp PROPERTIES COMPILE_OPTIONS "-fsycl")
  message(STATUS "Strata: SYCL kernels built (Intel GPU); add each ported kernel to this library as it lands")
endif()

# ============================================================================
# ---- SYCLomatic port (Intel GPU): the FULL engine on the SYCL backend ------
# ============================================================================
#
# The mirror of the CUDA block above, selected when STRATA_ENABLE_SYCL=ON.  Only
# two things actually change with the backend:
#
#   * the device layer source swaps (device_sycl.cpp + pinned_sycl.cpp instead of
#     device_cuda.cu + pinned.cu), and
#   * the kernel library swaps (strata_kernels_sycl instead of the .cu sources).
#
# Everything above them - strata_engine, strata_prefill, the `strata` program and
# the host parity harnesses - is built from the SAME host sources, compiled with
# -fsycl, with include/strata/core/sycl_compat on the include path so their
# `#include <cuda_runtime.h>` resolves to the SYCL shim.  The shim directory is
# listed BEFORE include/ so it wins over any CUDA header that may be on the box.
# The CUDA branch above is deliberately untouched: when STRATA_ENABLE_CUDA=ON
# none of this runs.
if(STRATA_ENABLE_SYCL AND NOT STRATA_ENABLE_CUDA)
  set(STRATA_SYCL_COMPAT ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/core/sycl_compat)

  # ---- strata_core: the SYCL device layer and the backend-neutral body ----
  # graph.cpp still names cudaGraph*; it is compiled against the shim until its
  # GraphRegistry gets its own SYCL port.  pinned.cu is a .cu and cannot be built
  # without nvcc, so the SYCL-suffixed pinned_sycl.cpp takes its place.
  add_library(strata_core STATIC
              src/core/device_sycl.cpp
              src/core/device_runtime.cpp
              src/core/pinned_sycl.cpp
              src/platform/memory.cpp
              src/core/graph.cpp
              src/core/weights.cpp
              src/core/layout.cpp)
  target_include_directories(strata_core PUBLIC
    ${STRATA_SYCL_COMPAT}
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_core PUBLIC strata_plan strata_warnings)
  # -fsycl on the device layer and on every host TU that calls through the shim.
  set_source_files_properties(src/core/device_sycl.cpp src/core/pinned_sycl.cpp
                              src/core/graph.cpp src/core/weights.cpp
                              PROPERTIES COMPILE_OPTIONS "-fsycl")
  # The link step needs -fsycl as well: it pulls in the SYCL runtime and the
  # offloaded objects.  PUBLIC so the rule reaches the `strata` link line too.
  target_link_options(strata_core PUBLIC "-fsycl")

  # `strata-device` names no backend: it prints what the active runtime reports.
  add_executable(strata-device src/core/device_main.cpp)
  target_link_libraries(strata-device PRIVATE strata_core)

  # ---- strata_kernels: the ported SYCL library replaces the .cu kernels ----
  # ngram.cpp / ple_reader.cpp / direct_file.cpp are backend-neutral host helpers
  # that lived in strata_kernels; they ride along in the SYCL kernel library.
  target_sources(strata_kernels_sycl PRIVATE
                 src/kernels/ngram.cpp src/ngram/ple_reader.cpp src/platform/direct_file.cpp)
  target_link_libraries(strata_kernels_sycl PUBLIC strata_core strata_artifact)

  # The name the engine links is unchanged; it now re-exports the SYCL kernels.
  add_library(strata_kernels INTERFACE)
  target_link_libraries(strata_kernels INTERFACE strata_kernels_sycl strata_core strata_artifact)

  # ---- strata_engine: identical sources to the CUDA branch, built with -fsycl ----
  add_library(strata_engine STATIC src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp
    src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp src/core/mtp.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(strata_engine PRIVATE "-fsycl")
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu)

  # ---- strata_prefill: the SYCL ports of gemm.cu / kernels.cu replace them ----
  # prefill.cpp is backend-neutral host code that reaches the device through the
  # shim; gemm_sycl.cpp (oneMKL) and kernels_sycl.cpp are the ported device half.
  add_library(strata_prefill STATIC
              src/prefill/gemm_sycl.cpp
              src/prefill/kernels_sycl.cpp
              src/prefill/prefill.cpp)
  target_include_directories(strata_prefill PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_source_files_properties(src/prefill/gemm_sycl.cpp src/prefill/kernels_sycl.cpp
                              PROPERTIES COMPILE_OPTIONS "-fsycl")
  target_compile_options(strata_prefill PRIVATE "-fsycl")

  # ---- oneMKL: gemm_sycl.cpp calls oneapi::mkl::blas::row_major::gemm ----
  # oneMKL's CMake package is named `MKL` (MKLConfig.cmake) and exports the
  # imported target `MKL::MKL_SYCL`, which carries both the oneMKL include dir
  # and the SYCL oneMKL libraries.  It is linked PUBLIC because strata_prefill
  # is a STATIC library: a private dependency's symbols would still be needed by
  # the `strata` exe that consumes it, so the link line must propagate.
  # ENABLE_SYCL_COMPILER is pinned ON because the config's AUTO probe only looks
  # for Intel driver names, and this build runs the SYCL host code through icx.
  set(MKL_INTERFACE lp64 CACHE STRING "" FORCE)
  set(MKL_THREADING intel_thread CACHE STRING "" FORCE)
  set(ENABLE_SYCL_COMPILER ON CACHE STRING "" FORCE)
  find_package(MKL CONFIG QUIET)
  if(TARGET MKL::MKL_SYCL)
    set(STRATA_ONEMKL_LIBS MKL::MKL_SYCL)
    message(STATUS "Strata: oneMKL found via MKLConfig.cmake -> MKL::MKL_SYCL")
  else()
    # Fallback for a toolchain without the MKL CMake package: take the raw
    # libs and the include dir straight from MKLROOT (oneAPI's mkl/latest).
    if(DEFINED ENV{MKLROOT})
      set(_strata_mklroot "$ENV{MKLROOT}")
    else()
      set(_strata_mklroot "C:/Program Files (x86)/Intel/oneAPI/mkl/latest")
    endif()
    find_path(STRATA_ONEMKL_INCLUDE oneapi/mkl.hpp HINTS "${_strata_mklroot}/include")
    find_library(STRATA_MKL_SYCL NAMES mkl_sycl mkl_sycl_dll HINTS "${_strata_mklroot}/lib")
    find_library(STRATA_MKL_INTEL_LP64 NAMES mkl_intel_lp64 HINTS "${_strata_mklroot}/lib")
    find_library(STRATA_MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS "${_strata_mklroot}/lib")
    find_library(STRATA_MKL_CORE NAMES mkl_core HINTS "${_strata_mklroot}/lib")
    find_library(STRATA_LIBIOMP5 NAMES libiomp5md
                 HINTS "${_strata_mklroot}/../compiler/latest/lib" "${_strata_mklroot}/../compiler/latest/lib/intel64")
    if(STRATA_ONEMKL_INCLUDE AND STRATA_MKL_SYCL AND STRATA_MKL_CORE)
      target_include_directories(strata_prefill PUBLIC "${STRATA_ONEMKL_INCLUDE}")
      target_link_directories(strata_prefill PUBLIC "${_strata_mklroot}/lib")
      set(STRATA_ONEMKL_LIBS "${STRATA_MKL_SYCL}" "${STRATA_MKL_INTEL_LP64}"
          "${STRATA_MKL_INTEL_THREAD}" "${STRATA_MKL_CORE}")
      if(STRATA_LIBIOMP5)
        list(APPEND STRATA_ONEMKL_LIBS "${STRATA_LIBIOMP5}")
      endif()
      message(STATUS "Strata: oneMKL found via raw libs under ${_strata_mklroot}")
    else()
      message(FATAL_ERROR
        "SYCL prefill needs oneMKL: neither MKL::MKL_SYCL nor the raw libs under '${_strata_mklroot}' were found.")
    endif()
  endif()
  target_link_libraries(strata_prefill PUBLIC strata_engine ${STRATA_ONEMKL_LIBS})

  # ---- `strata` ----
  add_executable(strata src/program/generate.cpp)
  target_compile_options(strata PRIVATE "-fsycl")
  target_link_libraries(strata PRIVATE strata_engine strata_prefill strata_spec)

  # ---- host parity harnesses: CUDA::cudart is replaced by the shim ----
  set(STRATA_SYCL_PARITY_TESTS
      dequant_s2_parity s2_gemv_parity iq_parity shared_expert_parity gr_parity gdn_parity
      s2_gemv_q8_parity sampler_parity rope_parity quantize_act_parity router_top10_parity
      s_gemv_parity elementwise_parity bf16_gemv_parity s_gemv_q8k_parity qsa_parity ple_parity
      kv_q8_parity kv_stream_parity kv_q4_parity)
  foreach(_t IN LISTS STRATA_SYCL_PARITY_TESTS)
    add_executable(${_t} src/kernels/${_t}.cpp)
    target_compile_options(${_t} PRIVATE "-fsycl")
    target_link_libraries(${_t} PRIVATE strata_kernels)
  endforeach()

  add_executable(platform_memory_test src/platform/memory_test.cpp)
  target_compile_options(platform_memory_test PRIVATE "-fsycl")
  target_link_libraries(platform_memory_test PRIVATE strata_core)

  add_executable(ple_reader_test src/ngram/ple_reader_test.cpp)
  target_compile_options(ple_reader_test PRIVATE "-fsycl")
  target_link_libraries(ple_reader_test PRIVATE strata_kernels)

  message(STATUS "Strata: SYCL engine built (Intel GPU) - core/engine/prefill/strata + host parity via the shim")
endif()


if(STRATA_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
