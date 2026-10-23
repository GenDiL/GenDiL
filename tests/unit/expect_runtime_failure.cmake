if(NOT DEFINED TEST_PROGRAM OR NOT DEFINED MODE OR NOT DEFINED EXPECTED_TEXT)
  message(FATAL_ERROR "Missing runtime-failure test configuration")
endif()

execute_process(
  COMMAND "${TEST_PROGRAM}" "${MODE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

if(result STREQUAL "0")
  message(FATAL_ERROR "Expected ${TEST_PROGRAM} ${MODE} to fail")
endif()

set(output "${stdout}${stderr}")
string(FIND "${output}" "${EXPECTED_TEXT}" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(
    FATAL_ERROR
    "Runtime failure did not contain expected text '${EXPECTED_TEXT}'.\n${output}"
  )
endif()
