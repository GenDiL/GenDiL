foreach(gendil_sparse_mode IN ITEMS AUTO OFF)
  string(TOLOWER "${gendil_sparse_mode}" gendil_sparse_mode_lower)
  add_test(
    NAME configure-device-sparse-finalization-${gendil_sparse_mode_lower}
    COMMAND
      ${CMAKE_COMMAND}
      -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
      -DBINARY_DIR=${PROJECT_BINARY_DIR}/configure-tests/${gendil_sparse_mode_lower}
      -DMODE=${gendil_sparse_mode}
      -DEXPECT_SUCCESS=TRUE
      -P
      ${PROJECT_SOURCE_DIR}/tests/cmake/configure-device-sparse-finalization.cmake
  )
endforeach()

add_test(
  NAME configure-device-sparse-finalization-on-requires-device
  COMMAND
    ${CMAKE_COMMAND}
    -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
    -DBINARY_DIR=${PROJECT_BINARY_DIR}/configure-tests/on-no-device
    -DMODE=ON
    -DEXPECT_SUCCESS=FALSE
    "-DEXPECTED_TEXT=requires USE_CUDA=ON or USE_HIP=ON"
    -P
    ${PROJECT_SOURCE_DIR}/tests/cmake/configure-device-sparse-finalization.cmake
)

if(USE_HIP)
  foreach(gendil_missing_rocprim_mode IN ITEMS AUTO ON)
    string(
      TOLOWER
      "${gendil_missing_rocprim_mode}"
      gendil_missing_rocprim_mode_lower
    )
    if(gendil_missing_rocprim_mode STREQUAL "AUTO")
      set(gendil_missing_rocprim_success TRUE)
      set(
        gendil_missing_rocprim_text
        "Raw triplets will move to host for finalization"
      )
    else()
      set(gendil_missing_rocprim_success FALSE)
      set(gendil_missing_rocprim_text "rocPRIM was not found")
    endif()
    add_test(
      NAME configure-device-sparse-finalization-hip-missing-rocprim-${gendil_missing_rocprim_mode_lower}
      COMMAND
        ${CMAKE_COMMAND}
        -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
        -DBINARY_DIR=${PROJECT_BINARY_DIR}/configure-tests/hip-missing-rocprim-${gendil_missing_rocprim_mode_lower}
        -DMODE=${gendil_missing_rocprim_mode}
        -DDEVICE_BACKEND=HIP
        -DHIP_COMPILER=${CMAKE_HIP_COMPILER}
        -DHIP_DIR=${HIP_DIR}
        -Drocsparse_DIR=${rocsparse_DIR}
        -DEXPECT_SUCCESS=${gendil_missing_rocprim_success}
        "-DEXPECTED_TEXT=${gendil_missing_rocprim_text}"
        -P
        ${PROJECT_SOURCE_DIR}/tests/cmake/configure-device-sparse-finalization.cmake
    )
  endforeach()

  if(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
    add_test(
      NAME configure-device-sparse-finalization-exports-rocprim
      COMMAND
        ${CMAKE_COMMAND}
        -DPACKAGE_CONFIG=${PROJECT_BINARY_DIR}/GENDILConfig.cmake
        -P ${PROJECT_SOURCE_DIR}/tests/cmake/verify-rocprim-export.cmake
    )
  endif()
endif()
