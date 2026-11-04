# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright (C) 2026 Ahmad Ali Parr / SnapKitty Research Lab
# Copied from bonsai-decoder (cmake/check_kernel_resources.cmake), which copied it
# from bonsai-weight-loader. Changed here: INCLUDES is a list of include dirs.
# For every kernel source and architecture: emit PTX and fail on ld.local /
# st.local, then assemble with -Xptxas=-v and fail on any non-zero spill.
# The ptxas report is written to OUT_DIR/ptxas_report.txt.
file(MAKE_DIRECTORY ${OUT_DIR})
set(incs "")
foreach(i IN LISTS INCLUDES)
  list(APPEND incs -I${SRC_DIR}/${i})
endforeach()
set(report "")
set(failed OFF)
foreach(arch IN LISTS ARCHS)
  string(REGEX REPLACE "-.*" "" arch "${arch}")
  foreach(src IN LISTS SOURCES)
    get_filename_component(stem ${src} NAME_WE)
    set(ptx ${OUT_DIR}/${stem}_sm${arch}.ptx)
    execute_process(
      COMMAND ${NVCC} -std=c++17 -O3 ${incs} -arch=sm_${arch} -ptx ${SRC_DIR}/${src} -o ${ptx}
      RESULT_VARIABLE rc ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "nvcc -ptx ${src} sm_${arch} failed:\n${err}")
    endif()
    file(STRINGS ${ptx} local_ops REGEX "(ld|st)\\.local")
    list(LENGTH local_ops n_local)
    execute_process(
      COMMAND ${NVCC} -std=c++17 -O3 ${incs} -gencode arch=compute_${arch},code=sm_${arch}
              -Xptxas=-v -c ${SRC_DIR}/${src} -o ${OUT_DIR}/${stem}_sm${arch}.o
      RESULT_VARIABLE rc ERROR_VARIABLE ptxas)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "nvcc -c ${src} sm_${arch} failed:\n${ptxas}")
    endif()
    string(APPEND report "==== ${src} sm_${arch}: ${n_local} ld.local/st.local in PTX\n${ptxas}\n")
    if(n_local GREATER 0)
      set(failed ON)
      message(SEND_ERROR "${src} sm_${arch}: local memory in PTX: ${local_ops}")
    endif()
    string(REGEX MATCHALL "[0-9]+ bytes spill (stores|loads)" spills "${ptxas}")
    foreach(s IN LISTS spills)
      if(NOT s MATCHES "^0 bytes")
        set(failed ON)
        message(SEND_ERROR "${src} sm_${arch}: ${s}")
      endif()
    endforeach()
    string(REGEX MATCHALL "[0-9]+ bytes stack frame" frames "${ptxas}")
    foreach(s IN LISTS frames)
      if(NOT s MATCHES "^0 bytes")
        set(failed ON)
        message(SEND_ERROR "${src} sm_${arch}: ${s}")
      endif()
    endforeach()
  endforeach()
endforeach()
file(WRITE ${OUT_DIR}/ptxas_report.txt "${report}")
message("${report}")
if(failed)
  message(FATAL_ERROR "kernel resource check failed")
endif()
message("kernel resource check passed: no ld.local/st.local, 0 spill bytes, 0 stack frame")
