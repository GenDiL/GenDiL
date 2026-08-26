include_guard(GLOBAL)

function(gendil_configure_mfem target output_enable_cuda)
  set(enable_cuda OFF)
  if(USE_MFEM)
    if(USE_HIP)
      find_package(HIP REQUIRED)
      find_package(HIPBLAS REQUIRED)
      find_package(HIPSPARSE REQUIRED)
    endif()

    find_package(
      MFEM REQUIRED NAMES MFEM
      HINTS "${MFEM_DIR}" "${MFEM_DIR}/lib/cmake/mfem"
      NO_DEFAULT_PATH
    )

    if(USE_CUDA AND NOT MFEM_USE_CUDA)
      message(
        FATAL_ERROR
        "USE_MFEM=ON with USE_CUDA=ON requires a CUDA-enabled MFEM, but "
        "the found MFEM was built without CUDA support. Rebuild MFEM with "
        "MFEM_USE_CUDA=ON or disable GenDiL CUDA support."
      )
    endif()
    if(USE_HIP AND NOT MFEM_USE_HIP)
      message(
        FATAL_ERROR
        "USE_MFEM=ON with USE_HIP=ON requires a HIP-enabled MFEM, but "
        "the found MFEM was built without HIP support. Rebuild MFEM with "
        "MFEM_USE_HIP=ON or disable GenDiL HIP support."
      )
    endif()

    message(STATUS "MFEM found!")
    target_include_directories(${target} INTERFACE ${MFEM_INCLUDE_DIRS})
    target_link_libraries(${target} INTERFACE ${MFEM_LIBRARIES})
    target_compile_definitions(
      ${target}
      INTERFACE
        GENDIL_USE_MFEM
        MFEM_DIR="${MFEM_LIBRARY_DIR}/.."
    )
    message(STATUS "mfem include directories: ${MFEM_INCLUDE_DIRS}")
    message(STATUS "mfem libraries: ${MFEM_LIBRARIES}")
    message(STATUS "MFEM_DIR is set to: ${MFEM_LIBRARY_DIR}/..")

    if(MFEM_USE_CUDA)
      set(enable_cuda ON)
    endif()
  else()
    message(STATUS "MFEM support is not enabled.")
  endif()

  set(${output_enable_cuda} ${enable_cuda} PARENT_SCOPE)
endfunction()
