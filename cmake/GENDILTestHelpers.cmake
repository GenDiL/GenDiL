include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/GENDILTargetHelpers.cmake")

function(gendil_add_test target source)
  cmake_parse_arguments(
    PARSE_ARGV 2
    GENDIL_TEST
    "EXCLUDE_FROM_ALL;NO_CTEST"
    "TIMEOUT"
    ""
  )

  if(GENDIL_TEST_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "gendil_add_test received unexpected arguments: "
      "${GENDIL_TEST_UNPARSED_ARGUMENTS}"
    )
  endif()

  if(GENDIL_TEST_EXCLUDE_FROM_ALL)
    add_executable(${target} EXCLUDE_FROM_ALL "${source}")
  else()
    add_executable(${target} "${source}")
  endif()
  target_link_libraries(${target} PRIVATE GENDIL::GENDIL)
  gendil_set_device_source_language(${target} "${source}")

  if(NOT GENDIL_TEST_NO_CTEST)
    add_test(NAME ${target} COMMAND ${target})
    if(DEFINED GENDIL_TEST_TIMEOUT)
      set_tests_properties(${target} PROPERTIES TIMEOUT ${GENDIL_TEST_TIMEOUT})
    endif()
  endif()
endfunction()

function(gendil_add_non_nvcc_test target source)
  if(USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_ID)
      message(
        FATAL_ERROR
        "USE_CUDA is ON but CMAKE_CUDA_COMPILER_ID is undefined. Ensure "
        "enable_language(CUDA) has been called before adding tests."
      )
    endif()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      message(STATUS "Skipping test ${target}: excluded for NVCC compiler")
      return()
    endif()
  endif()

  gendil_add_test(${target} "${source}" ${ARGN})
endfunction()

function(gendil_add_compile_failure_test target source expected_text)
  add_test(
    NAME ${target}
    COMMAND
      ${CMAKE_COMMAND}
      -DCXX=${CMAKE_CXX_COMPILER}
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${source}
      -DINCLUDE_DIR=${PROJECT_SOURCE_DIR}/include
      -DOBJECT_FILE=${CMAKE_CURRENT_BINARY_DIR}/${target}.o
      -DEXPECTED_TEXT=${expected_text}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_compile_failure.cmake
  )
endfunction()

function(gendil_add_target_compile_failure_test target source expected_text)
  set(probe_target "${target}-probe")
  add_executable(${probe_target} EXCLUDE_FROM_ALL "${source}")
  target_link_libraries(${probe_target} PRIVATE GENDIL::GENDIL)
  gendil_set_device_source_language(${probe_target} "${source}")

  add_test(
    NAME ${target}
    COMMAND
      ${CMAKE_COMMAND}
      -DBUILD_DIR=${CMAKE_BINARY_DIR}
      -DPROBE_TARGET=${probe_target}
      -DBUILD_CONFIG=$<CONFIG>
      "-DEXPECTED_TEXT=${expected_text}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_target_compile_failure.cmake
  )
  set_tests_properties(${target} PROPERTIES RUN_SERIAL TRUE)
endfunction()
