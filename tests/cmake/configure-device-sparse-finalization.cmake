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
  -DUSE_MFEM=OFF
  -DUSE_HYPRE=OFF
  -DUSE_CALIPER=OFF
  -DUSE_RAJA=OFF
  -DBUILD_TESTING=OFF
  -DGENDIL_ENABLE_BENCHMARKS=OFF
  "-DGENDIL_DEVICE_SPARSE_FINALIZATION=${MODE}"
)

foreach(language IN ITEMS C CXX)
  if(DEFINED ${language}_COMPILER AND NOT "${${language}_COMPILER}" STREQUAL "")
    list(APPEND configure_command
      "-DCMAKE_${language}_COMPILER=${${language}_COMPILER}")
  endif()
endforeach()
if(DEFINED GENERATOR AND NOT GENERATOR STREQUAL "")
  list(APPEND configure_command -G "${GENERATOR}")
endif()
if(DEFINED MAKE_PROGRAM AND NOT MAKE_PROGRAM STREQUAL "")
  list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()

if(DEFINED DEVICE_BACKEND AND DEVICE_BACKEND STREQUAL "HIP")
  if(NOT DEFINED HIP_ARCHITECTURES OR HIP_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "HIP_ARCHITECTURES is required for a HIP configuration check")
  endif()
  # Keep multiple architectures in one CMake argument.
  string(REPLACE ";" "\\;" hip_architectures "${HIP_ARCHITECTURES}")
  list(APPEND configure_command
    -DUSE_HIP=ON
    "-DCMAKE_HIP_COMPILER=${HIP_COMPILER}"
    "-DCMAKE_HIP_ARCHITECTURES=${hip_architectures}"
    "-DAMDGPU_TARGETS=${hip_architectures}"
    "-DHIP_DIR=${HIP_DIR}"
    # GenDiL finds HIP; rocSPARSE's dependency uses the lowercase package name.
    "-Dhip_DIR=${hip_DIR}"
    "-Drocsparse_DIR=${rocsparse_DIR}"
    -DCMAKE_DISABLE_FIND_PACKAGE_rocprim=TRUE
    -DGENDIL_FETCH_ROCPRIM=OFF
  )
  if(DEFINED HIP_ROCM_ROOT AND NOT HIP_ROCM_ROOT STREQUAL "")
    list(APPEND configure_command
      "-DCMAKE_HIP_COMPILER_ROCM_ROOT=${HIP_ROCM_ROOT}")
  endif()
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
  # Normalize whitespace in both output and expected text for robust matching
  # CMake can wrap long messages across lines, breaking exact matches
  string(REGEX REPLACE "[\r\n\t ]+" " " normalized_output "${configure_output}")
  string(REGEX REPLACE "[\r\n\t ]+" " " normalized_expected "${EXPECTED_TEXT}")
  string(FIND "${normalized_output}" "${normalized_expected}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "Configuration output did not contain '${EXPECTED_TEXT}':\n"
      "${configure_output}")
  endif()
endif()
