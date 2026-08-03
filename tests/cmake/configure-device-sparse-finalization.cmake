if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR OR NOT DEFINED MODE)
  message(FATAL_ERROR "SOURCE_DIR, BINARY_DIR, and MODE are required")
endif()

set(configure_command
  "${CMAKE_COMMAND}"
  -S "${SOURCE_DIR}"
  -B "${BINARY_DIR}"
  -DUSE_OPENMP=OFF
  -DUSE_CUDA=OFF
  -DUSE_HIP=OFF
  -DGENDIL_ENABLE_BENCHMARKS=OFF
  "-DGENDIL_DEVICE_SPARSE_FINALIZATION=${MODE}"
)

if(DEFINED DEVICE_BACKEND AND DEVICE_BACKEND STREQUAL "HIP")
  list(APPEND configure_command
    -DUSE_HIP=ON
    "-DCMAKE_HIP_COMPILER=${HIP_COMPILER}"
    "-DHIP_DIR=${HIP_DIR}"
    "-Drocsparse_DIR=${rocsparse_DIR}"
    -DCMAKE_DISABLE_FIND_PACKAGE_rocprim=TRUE
    -DGENDIL_FETCH_ROCPRIM=OFF
  )
endif()

execute_process(
  COMMAND ${configure_command}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_stdout
  ERROR_VARIABLE configure_stderr
)
set(configure_output "${configure_stdout}\n${configure_stderr}")

if(EXPECT_SUCCESS AND NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "Expected configuration to succeed, but it failed:\n${configure_output}")
elseif(NOT EXPECT_SUCCESS AND configure_result EQUAL 0)
  message(FATAL_ERROR
    "Expected configuration to fail, but it succeeded:\n${configure_output}")
endif()

if(DEFINED EXPECTED_TEXT)
  string(FIND "${configure_output}" "${EXPECTED_TEXT}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "Configuration output did not contain '${EXPECTED_TEXT}':\n"
      "${configure_output}")
  endif()
endif()
