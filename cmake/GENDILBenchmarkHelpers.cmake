include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/GENDILTargetHelpers.cmake")

function(gendil_add_benchmark target source)
  add_executable(${target} "${source}")
  target_link_libraries(${target} PRIVATE GENDIL::GENDIL)
  target_include_directories(
    ${target}
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )
  gendil_set_device_source_language(${target} "${source}")
  add_dependencies(benchmarks ${target})
endfunction()

function(gendil_add_gpu_benchmark target source)
  if(NOT (USE_CUDA OR USE_HIP))
    return()
  endif()
  gendil_add_benchmark(${target} "${source}")
endfunction()

function(gendil_add_non_cuda_benchmark target source)
  if(USE_CUDA)
    return()
  endif()
  gendil_add_benchmark(${target} "${source}")
endfunction()
