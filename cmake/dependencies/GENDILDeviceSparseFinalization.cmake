include_guard(GLOBAL)

function(gendil_validate_device_platform)
  if(USE_CUDA AND USE_HIP)
    message(
      FATAL_ERROR
      "GenDiL supports one device platform per build. Disable either "
      "USE_CUDA or USE_HIP."
    )
  endif()
endfunction()

function(gendil_sparse_finalization_unavailable reason)
  set(reason_text "${reason}")
  foreach(reason_fragment IN LISTS ARGN)
    string(APPEND reason_text "${reason_fragment}")
  endforeach()

  if(GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "ON")
    message(
      FATAL_ERROR
      "Device sparse finalization was requested but is unavailable: "
      "${reason_text}"
    )
  elseif(GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "AUTO")
    message(
      WARNING
      "Device sparse finalization is unavailable: ${reason_text} Raw "
      "triplets will move to host for finalization and the canonical matrix "
      "may later move back to device. Set "
      "GENDIL_DEVICE_SPARSE_FINALIZATION=OFF to select this fallback "
      "intentionally without this warning."
    )
  endif()
endfunction()

function(gendil_finalize_device_sparse_finalization target enabled)
  if(GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "ON" AND
     NOT (USE_CUDA OR USE_HIP))
    message(
      FATAL_ERROR
      "GENDIL_DEVICE_SPARSE_FINALIZATION=ON requires USE_CUDA=ON or USE_HIP=ON."
    )
  endif()

  if(enabled)
    target_compile_definitions(
      ${target}
      INTERFACE GENDIL_HAS_DEVICE_SPARSE_FINALIZATION
    )
    message(STATUS "GPU sparse finalization enabled")
  elseif(GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "OFF")
    message(STATUS "GPU sparse finalization disabled by request")
  endif()
endfunction()
