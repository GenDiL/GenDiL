# Check CMake admission and argument forwarding only; this does not compile
# CUDA/HIP code or require a device toolchain.
if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()
include("${SOURCE_DIR}/cmake/GENDILTestHelpers.cmake")

function(gendil_add_test target source)
  if(NOT source STREQUAL "probe.cpp" OR
     NOT "${ARGN}" STREQUAL "EXCLUDE_FROM_ALL;NO_CTEST;TIMEOUT;42")
    message(FATAL_ERROR "The compiler gate did not preserve test arguments")
  endif()
  set_property(GLOBAL APPEND PROPERTY admitted_tests "${target}")
endfunction()

function(check_admission name cuda hip compiler version expected_count)
  set(USE_CUDA "${cuda}")
  set(USE_HIP "${hip}")
  set(CMAKE_CUDA_COMPILER_ID "${compiler}")
  set(CMAKE_CUDA_COMPILER_VERSION "${version}")
  set_property(GLOBAL PROPERTY admitted_tests "")
  foreach(index RANGE 1 2)
    gendil_add_non_nvcc_test("${name}-${index}" probe.cpp
      EXCLUDE_FROM_ALL NO_CTEST TIMEOUT 42)
  endforeach()
  get_property(admitted GLOBAL PROPERTY admitted_tests)
  list(LENGTH admitted count)
  if(NOT count EQUAL expected_count)
    message(FATAL_ERROR
      "${name}: admitted ${count} tests, expected ${expected_count}")
  endif()

  if(cuda)
    execute_process(
      COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${SOURCE_DIR}"
        "-DCOMPILER_ID=${compiler}" "-DCOMPILER_VERSION=${version}"
        -P "${CMAKE_CURRENT_LIST_DIR}/nvcc-warning-fixture.cmake"
      RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "${name}: warning fixture failed:\n${stdout}\n${stderr}")
    endif()
    string(REGEX MATCHALL "CMake Warning" warnings "${stderr}")
    list(LENGTH warnings warning_count)
    if(expected_count EQUAL 0)
      string(REGEX REPLACE "[\r\n\t ]+" " " diagnostic "${stderr}")
      set(expected "Weak-Form DSL is not supported with NVCC. Detected NVCC ${version}; tests requiring this support will be skipped.")
      string(FIND "${diagnostic}" "${expected}" position)
      if(NOT warning_count EQUAL 1 OR position EQUAL -1)
        message(FATAL_ERROR "${name}: expected one unsupported-NVCC warning:\n${stderr}")
      endif()
    elseif(NOT warning_count EQUAL 0)
      message(FATAL_ERROR "${name}: unexpected warning:\n${stderr}")
    endif()
  endif()
endfunction()

check_admission(cpu OFF OFF "" "" 2)
check_admission(hip OFF ON "" "" 2)
check_admission(clang-cuda ON OFF Clang 19.1.7 2)
check_admission(nvcc-12 ON OFF NVIDIA 12.4.131 0)
check_admission(nvcc-13-1 ON OFF NVIDIA 13.1.115 0)
check_admission(nvcc-13-2 ON OFF NVIDIA 13.2.51 0)
check_admission(nvcc-13-3 ON OFF NVIDIA 13.3.33 0)
check_admission(nvcc-previous-ci ON OFF NVIDIA 13.4.92 0)
check_admission(nvcc-ci ON OFF NVIDIA 13.5.37 0)
# A future version must not enable these tests without complete-build validation.
check_admission(nvcc-unverified-future ON OFF NVIDIA 13.6.0 0)
