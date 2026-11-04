# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # A100/A30, RTX 30xx/A10/A40, Jetson Orin, RTX 40xx/L4/L40, H100
  set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89;90" CACHE STRING "CUDA architectures (sm_XX numbers)")
endif()

project(bonsai_ternary_fast LANGUAGES CXX)

option(BTF_CUDA "Build the CUDA decoder (needs nvcc)" ON)
set(BTF_MODEL_GGUF "" CACHE FILEPATH "Ternary-Bonsai-2-27B-PTQ1_0.gguf for the real-model tests")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
find_package(Threads REQUIRED)

set(BTF_HAVE_CUDA OFF)
if(BTF_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(BTF_HAVE_CUDA ON)
  else()
    message(STATUS "nvcc not found: building the CPU emulator, tools and tests only")
  endif()
endif()

# Reference decoder copied from bonsai-decoder @ 7bcdb88 (GGUF reader, model
# binding, CPU fp32 reference, tokenizer, synthetic model). See ref/README.md.
set(REF ref/bonsai-decoder)
add_library(bd_ref STATIC
  ${REF}/src/mapped_file.cpp ${REF}/src/gguf.cpp ${REF}/src/types.cpp ${REF}/src/model.cpp
  ${REF}/src/cpu_ops.cpp ${REF}/src/cpu_decoder.cpp ${REF}/src/trace.cpp ${REF}/src/tokenizer.cpp
  ${REF}/src/sampler.cpp ${REF}/src/synth.cpp ${REF}/src/cuda_stub.cpp)
target_include_directories(bd_ref PUBLIC ${REF}/include)
target_link_libraries(bd_ref PUBLIC Threads::Threads)

add_library(btf STATIC src/repack.cpp src/emu_decoder.cpp)
target_include_directories(btf PUBLIC include PRIVATE src)
target_link_libraries(btf PUBLIC bd_ref)

set(BTF_KERNEL_SOURCES src/cuda/kernels.cu)
if(BTF_HAVE_CUDA)
  target_sources(btf PRIVATE ${BTF_KERNEL_SOURCES} src/cuda/gpu_decoder.cu)
  target_link_libraries(btf PUBLIC CUDA::cudart)
  target_compile_definitions(btf PUBLIC BTF_HAVE_CUDA=1)
  set_source_files_properties(${BTF_KERNEL_SOURCES} PROPERTIES COMPILE_OPTIONS "-Xptxas=-v")
else()
  target_sources(btf PRIVATE src/gpu_stub.cpp)
endif()

foreach(tool btf_plan btf_bench btf_golden)
  add_executable(${tool} tools/${tool}.cpp)
  target_include_directories(${tool} PRIVATE ${REF}/tools)
  target_link_libraries(${tool} PRIVATE btf)
endforeach()

enable_testing()
foreach(t test_packed test_emu_vs_cpu test_plan)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE btf)
  add_test(NAME ${t} COMMAND ${t})
  set_tests_properties(${t} PROPERTIES LABELS cpu SKIP_RETURN_CODE 77)
endforeach()
set_tests_properties(test_plan PROPERTIES ENVIRONMENT "BTF_MODEL_GGUF=${BTF_MODEL_GGUF}")

if(BTF_HAVE_CUDA)
  foreach(t test_gpu_kernels test_gpu_decoder)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE src)
    target_link_libraries(${t} PRIVATE btf)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
  endforeach()
  add_test(NAME kernel_resources
    COMMAND ${CMAKE_COMMAND} -DNVCC=${CMAKE_CUDA_COMPILER} "-DARCHS=${CMAKE_CUDA_ARCHITECTURES}"
            -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR} "-DINCLUDES=include;${REF}/include;src"
            "-DSOURCES=${BTF_KERNEL_SOURCES}" -DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/kernel_resources
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_kernel_resources.cmake)
  set_tests_properties(kernel_resources PROPERTIES LABELS cpu)
  if(BTF_MODEL_GGUF)
    add_test(NAME golden_greeting COMMAND btf_golden -m ${BTF_MODEL_GGUF} --ref-tokens
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/greeting_ref.txt)
    set_tests_properties(golden_greeting PROPERTIES LABELS "gpu;golden" SKIP_RETURN_CODE 77)
  endif()
endif()
