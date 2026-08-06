if(NOT DEFINED BUILD_DIR OR
   NOT DEFINED PROBE_TARGET OR
   NOT DEFINED EXPECTED_TEXT)
  message(FATAL_ERROR "Target compile-failure test is missing inputs.")
endif()

set(
  build_command
  "${CMAKE_COMMAND}"
  --build
  "${BUILD_DIR}"
  --target
  "${PROBE_TARGET}"
)
if(DEFINED BUILD_CONFIG AND NOT BUILD_CONFIG STREQUAL "")
  list(APPEND build_command --config "${BUILD_CONFIG}")
endif()

execute_process(
  COMMAND ${build_command}
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_stdout
  ERROR_VARIABLE build_stderr
)

if(build_result EQUAL 0)
  message(FATAL_ERROR
    "Expected target compilation to fail: ${PROBE_TARGET}")
endif()

set(build_output "${build_stdout}\n${build_stderr}")
string(FIND "${build_output}" "${EXPECTED_TEXT}" expected_index)
if(expected_index EQUAL -1)
  message(
    FATAL_ERROR
    "Target compilation failed without expected diagnostic "
    "'${EXPECTED_TEXT}'.\n${build_output}"
  )
endif()
