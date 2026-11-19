if(NOT DEFINED PRODUCER_SETTINGS)
  message(FATAL_ERROR "PRODUCER_SETTINGS is required")
endif()
include("${PRODUCER_SETTINGS}")

# Absolute install destinations cannot be safely redirected into the test prefix.
if(IS_ABSOLUTE "${GENDIL_PRODUCER_INCLUDE_DIR}" OR
   IS_ABSOLUTE "${GENDIL_PRODUCER_DATA_DIR}")
  message(FATAL_ERROR
    "The installed-package relocation test requires relative include and data install directories.")
endif()

if(NOT BUILD_CONFIG)
  set(BUILD_CONFIG "${CMAKE_BUILD_TYPE}")
endif()
if(NOT BUILD_CONFIG)
  set(BUILD_CONFIG Release)
endif()
set(test_root "${GENDIL_TEST_ROOT}/${BUILD_CONFIG}")
file(REMOVE_RECURSE "${test_root}")
file(MAKE_DIRECTORY "${test_root}")

function(run_step name)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
  )
  file(WRITE "${test_root}/${name}.log" "${stdout}\n${stderr}")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "Installed-package ${name} failed (${result}):\n${stdout}\n${stderr}")
  endif()
endfunction()

set(original_prefix "${test_root}/original-prefix")
set(relocated_prefix "${test_root}/relocated-prefix")
run_step(install "${CMAKE_COMMAND}" --install "${GENDIL_PRODUCER_BUILD}"
  --prefix "${original_prefix}" --config "${BUILD_CONFIG}")
file(RENAME "${original_prefix}" "${relocated_prefix}")

# A cache file preserves list-valued settings without shell or CLI re-quoting.
set(consumer_cache "${test_root}/consumer-cache.cmake")
file(WRITE "${consumer_cache}"
  "include([==[${PRODUCER_SETTINGS}]==])\n"
  "list(PREPEND CMAKE_PREFIX_PATH [==[${relocated_prefix}]==])\n"
  "set(CMAKE_PREFIX_PATH \"\${CMAKE_PREFIX_PATH}\" CACHE STRING \"Dependency search paths\" FORCE)\n")
set(consumer_build "${test_root}/consumer-build")
set(configure_command "${CMAKE_COMMAND}"
  -S "${GENDIL_PRODUCER_SOURCE}/tests/cmake/installed-consumer"
  -B "${consumer_build}"
  -G "${GENDIL_PRODUCER_GENERATOR}"
  -C "${consumer_cache}"
  "-DCMAKE_BUILD_TYPE=${BUILD_CONFIG}"
  "-DCMAKE_INSTALL_PREFIX=${test_root}/consumer-install"
  "-DGENDIL_EXPECTED_PREFIX=${relocated_prefix}"
  "-DGENDIL_EXPECTED_INCLUDE_DIR=${relocated_prefix}/${GENDIL_PRODUCER_INCLUDE_DIR}"
  "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"
  "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"
  "-DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=OFF"
)
if(GENDIL_PRODUCER_PLATFORM)
  list(APPEND configure_command -A "${GENDIL_PRODUCER_PLATFORM}")
endif()
if(GENDIL_PRODUCER_TOOLSET)
  list(APPEND configure_command -T "${GENDIL_PRODUCER_TOOLSET}")
endif()
run_step(configure ${configure_command})
run_step(build "${CMAKE_COMMAND}" --build "${consumer_build}"
  --config "${BUILD_CONFIG}" --parallel 2)

if(GENDIL_TEST_BACKEND STREQUAL "CPU")
  run_step(run "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
    --build-config "${BUILD_CONFIG}" --output-on-failure --no-tests=error)
else()
  message(STATUS "${GENDIL_TEST_BACKEND} installed consumer compiled; device execution is deferred.")
endif()
