include_guard(GLOBAL)

function(gendil_configure_hypre target output_device_enabled)
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)

  set(device_enabled OFF)
  if(USE_HYPRE)
    find_package(HYPRE REQUIRED CONFIG)
    message(STATUS "Hypre found!")
    target_link_libraries(${target} INTERFACE HYPRE::HYPRE)
    target_compile_definitions(${target} INTERFACE GENDIL_USE_HYPRE)

    if(HYPRE_ENABLE_SYCL)
      message(
        FATAL_ERROR
        "USE_HYPRE=ON found a SYCL-enabled Hypre, but GenDiL's Hypre device "
        "backend supports only CUDA and HIP in this version."
      )
    endif()

    if(USE_CUDA)
      if(HYPRE_ENABLE_CUDA)
        set(device_enabled ON)
        target_compile_definitions(
          ${target} INTERFACE GENDIL_USE_HYPRE_DEVICE
        )
        message(
          STATUS
          "GenDiL HypreCSR default backend: HypreCSRDeviceBackend (CUDA)"
        )
      elseif(HYPRE_ENABLE_HIP)
        message(
          FATAL_ERROR
          "USE_CUDA=ON requires a CUDA-enabled Hypre for "
          "HypreCSRDeviceBackend, but the found Hypre was built with HIP "
          "support."
        )
      else()
        message(
          STATUS
          "GenDiL device support is enabled, but the found Hypre is "
          "host-only; HypreCSR will use HypreCSRHostBackend by default."
        )
      endif()
    elseif(USE_HIP)
      if(HYPRE_ENABLE_HIP)
        set(device_enabled ON)
        target_compile_definitions(
          ${target} INTERFACE GENDIL_USE_HYPRE_DEVICE
        )
        message(
          STATUS
          "GenDiL HypreCSR default backend: HypreCSRDeviceBackend (HIP)"
        )
      elseif(HYPRE_ENABLE_CUDA)
        message(
          FATAL_ERROR
          "USE_HIP=ON requires a HIP-enabled Hypre for "
          "HypreCSRDeviceBackend, but the found Hypre was built with CUDA "
          "support."
        )
      else()
        message(
          STATUS
          "GenDiL device support is enabled, but the found Hypre is "
          "host-only; HypreCSR will use HypreCSRHostBackend by default."
        )
      endif()
    elseif(HYPRE_ENABLE_CUDA OR HYPRE_ENABLE_HIP)
      message(
        STATUS
        "The found Hypre has device support, but GenDiL device support is "
        "off; HypreCSR will use HypreCSRHostBackend by default."
      )
    endif()

    get_target_property(
      HYPRE_INCLUDE_DIRS HYPRE::HYPRE INTERFACE_INCLUDE_DIRECTORIES
    )
    set(hypre_internal_include_dirs ${HYPRE_INCLUDE_DIRS})
    foreach(hypre_include_dir IN LISTS HYPRE_INCLUDE_DIRS)
      if(EXISTS "${hypre_include_dir}/seq_block_mv/_hypre_seq_block_mv.h")
        list(
          APPEND hypre_internal_include_dirs
          "${hypre_include_dir}/seq_block_mv"
        )
      endif()
    endforeach()
    target_include_directories(
      ${target} INTERFACE ${hypre_internal_include_dirs}
    )

    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${hypre_internal_include_dirs})
    set(CMAKE_REQUIRED_LIBRARIES HYPRE::HYPRE)
    check_cxx_source_compiles(
      "
      #include <HYPRE.h>
      #include <HYPRE_parcsr_mv.h>
      #include <HYPRE_parcsr_ls.h>
      #include <HYPRE_utilities.h>
      #include \"_hypre_utilities.h\"
      #include \"_hypre_seq_mv.h\"
      #include \"_hypre_parcsr_mv.h\"
      int main()
      {
        HYPRE_Initialized();
        (void) &hypre_SeqVectorSetData;
        (void) &hypre_ParVectorInitializeShell;
        (void) &hypre_ParVectorInitialize_v2;
        (void) &hypre_ParCSRMatrixInitialize_v2;
        return 0;
      }
      "
      GENDIL_HYPRE_HAS_INTERNAL_HEADERS
    )
    cmake_pop_check_state()
    if(NOT GENDIL_HYPRE_HAS_INTERNAL_HEADERS)
      message(
        FATAL_ERROR
        "USE_HYPRE=ON requires Hypre internal headers "
        "(_hypre_utilities.h, _hypre_seq_mv.h, _hypre_parcsr_mv.h) and the "
        "corresponding internal aliasing APIs for zero-copy "
        "HYPRE_ParCSRMatrix/HYPRE_ParVector views."
      )
    endif()
  else()
    message(STATUS "Hypre support is not enabled.")
  endif()

  set(${output_device_enabled} ${device_enabled} PARENT_SCOPE)
endfunction()
