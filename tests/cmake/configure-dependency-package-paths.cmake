# Test package discovery/export only. These interface-only dependency fixtures
# do not provide MFEM, RAJA, or Caliper implementations or numerical validation.
if(NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR)
  message(FATAL_ERROR "SOURCE_DIR and BINARY_DIR are required")
endif()

file(REMOVE_RECURSE "${BINARY_DIR}")
file(MAKE_DIRECTORY "${BINARY_DIR}")

function(run_step name)
  execute_process(COMMAND ${ARGN}
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  file(WRITE "${BINARY_DIR}/${name}.log" "${stdout}\n${stderr}")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Dependency-path ${name} failed (${result}):\n${stdout}\n${stderr}")
  endif()
endfunction()

function(require_text file text description)
  file(READ "${file}" contents)
  string(FIND "${contents}" "${text}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "${description}: '${text}' is missing from ${file}")
  endif()
endfunction()

set(config_subdir_MFEM lib/cmake/mfem)
set(config_subdir_RAJA share/raja/cmake)
set(config_subdir_caliper share/cmake/caliper)
set(target_MFEM mfem)
set(target_RAJA RAJA)
set(target_caliper caliper)

# Each package records its exact selected configuration directory, so another
# installation in the host environment cannot make a broken search pass.
foreach(origin IN ITEMS producer override)
  foreach(package IN ITEMS MFEM RAJA caliper)
    set(config_dir "${BINARY_DIR}/packages/${origin}/${package}/${config_subdir_${package}}")
    file(MAKE_DIRECTORY "${config_dir}/include")
    file(WRITE "${config_dir}/${package}Config.cmake"
      "set(GENDIL_FIXTURE_${package}_DIR \"\${CMAKE_CURRENT_LIST_DIR}\")\n"
      "if(NOT TARGET ${target_${package}})\n"
      "  add_library(${target_${package}} INTERFACE IMPORTED)\n"
      "  set_target_properties(${target_${package}} PROPERTIES\n"
      "    INTERFACE_INCLUDE_DIRECTORIES \"\${CMAKE_CURRENT_LIST_DIR}/include\")\n"
      "endif()\n")
    if(package STREQUAL "MFEM")
      # The imported target supplies includes. Copying a legacy include path
      # onto GenDiL would break export when the test build is under its source.
      file(APPEND "${config_dir}/${package}Config.cmake"
        "set(MFEM_INCLUDE_DIRS \"\")\n"
        "set(MFEM_LIBRARY_DIR \"\${CMAKE_CURRENT_LIST_DIR}\")\n"
        "set(MFEM_LIBRARIES mfem)\n"
        "set(MFEM_USE_CUDA OFF)\n"
        "set(MFEM_USE_HIP OFF)\n")
    endif()
  endforeach()
endforeach()

set(consumer_source "${BINARY_DIR}/consumer-source")
file(MAKE_DIRECTORY "${consumer_source}")
file(WRITE "${consumer_source}/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.25)
project(dependency_path_consumer LANGUAGES CXX)
find_package(GENDIL REQUIRED CONFIG)
if(NOT TARGET GENDIL::GENDIL)
  message(FATAL_ERROR "The installed GenDiL target is missing")
endif()
set(config_subdir_MFEM lib/cmake/mfem)
set(config_subdir_RAJA share/raja/cmake)
set(config_subdir_caliper share/cmake/caliper)
foreach(package IN ITEMS MFEM RAJA caliper)
  set(expected "${EXPECTED_DEPENDENCIES}/${package}/${config_subdir_${package}}")
  if(NOT "${GENDIL_FIXTURE_${package}_DIR}" STREQUAL "${expected}")
    message(FATAL_ERROR "${package} selected '${GENDIL_FIXTURE_${package}_DIR}', expected '${expected}'")
  endif()
endforeach()
add_library(consumer INTERFACE)
target_link_libraries(consumer INTERFACE GENDIL::GENDIL)
]=])

set(toolchain_arguments)
foreach(language IN ITEMS C CXX)
  if(DEFINED ${language}_COMPILER AND NOT "${${language}_COMPILER}" STREQUAL "")
    list(APPEND toolchain_arguments "-DCMAKE_${language}_COMPILER=${${language}_COMPILER}")
  endif()
endforeach()
if(DEFINED GENERATOR AND NOT GENERATOR STREQUAL "")
  list(APPEND toolchain_arguments -G "${GENERATOR}")
endif()
if(DEFINED MAKE_PROGRAM AND NOT MAKE_PROGRAM STREQUAL "")
  list(APPEND toolchain_arguments "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
set(search_arguments
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
  -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=OFF)

foreach(mode IN ITEMS config-directory install-prefix)
  set(producer_build "${BINARY_DIR}/${mode}/producer-build")
  set(install_prefix "${BINARY_DIR}/${mode}/install")
  set(dependency_arguments)
  foreach(package IN ITEMS MFEM RAJA caliper)
    set(selected_${package} "${BINARY_DIR}/packages/producer/${package}/${config_subdir_${package}}")
    if(mode STREQUAL "config-directory")
      set(hint "${selected_${package}}")
    else()
      set(hint "${BINARY_DIR}/packages/producer/${package}")
    endif()
    file(RELATIVE_PATH relative_${package} "${SOURCE_DIR}" "${hint}")
    list(APPEND dependency_arguments "-D${package}_DIR:PATH=${relative_${package}}")
  endforeach()

  # Configure the real producer. BUILD_TESTING also exercises its generated
  # installed-consumer replay settings; no library/test compilation is needed.
  run_step("${mode}-producer" "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${producer_build}"
    ${toolchain_arguments} ${search_arguments} ${dependency_arguments}
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DGENDIL_ENABLE_BENCHMARKS=OFF
    -DUSE_OPENMP=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF -DUSE_HYPRE=OFF
    -DUSE_MFEM=ON -DUSE_RAJA=ON -DUSE_CALIPER=ON
    "-DCMAKE_INSTALL_PREFIX=${install_prefix}")

  foreach(package IN ITEMS MFEM RAJA caliper)
    # These generated files see the dependency wrapper's parent scope.
    require_text("${producer_build}/GENDILConfig.cmake"
      "set(${package}_DIR \"${selected_${package}}\" CACHE PATH \"\")"
      "The exported ${package} fallback must name the selected absolute configuration directory")
    require_text("${producer_build}/tests/cmake/installed-package-settings.cmake"
      "set(${package}_DIR [==[${selected_${package}}]==]"
      "The ${package} replay setting must name the selected absolute configuration directory")
    if(mode STREQUAL "config-directory")
      # find_package leaves a valid relative config-directory cache entry intact.
      require_text("${producer_build}/CMakeCache.txt"
        "${package}_DIR:PATH=${relative_${package}}\n"
        "Resolving ${package} for export must not rewrite the caller's cache")
    else()
      # A prefix is not a config directory: find_package itself replaces this
      # cache hint with the selected directory before the wrapper returns.
      require_text("${producer_build}/CMakeCache.txt"
        "${package}_DIR:PATH=${selected_${package}}\n"
        "${package} prefix discovery must retain CMake's resolved cache value")
    endif()
  endforeach()

  run_step("${mode}-install" "${CMAKE_COMMAND}" --install "${producer_build}" --config Release)
  # Do not replay producer dependency settings: the installed package's own
  # absolute fallbacks must work from an unrelated consumer source/build tree.
  run_step("${mode}-consumer" "${CMAKE_COMMAND}"
    -S "${consumer_source}" -B "${BINARY_DIR}/${mode}/consumer-build"
    ${toolchain_arguments} ${search_arguments}
    "-DGENDIL_DIR=${install_prefix}/share/gendil/cmake"
    "-DEXPECTED_DEPENDENCIES=${BINARY_DIR}/packages/producer")

  set(overrides)
  foreach(package IN ITEMS MFEM RAJA caliper)
    list(APPEND overrides "-D${package}_DIR:PATH=${BINARY_DIR}/packages/override/${package}/${config_subdir_${package}}")
  endforeach()
  run_step("${mode}-consumer-override" "${CMAKE_COMMAND}"
    -S "${consumer_source}" -B "${BINARY_DIR}/${mode}/consumer-override-build"
    ${toolchain_arguments} ${search_arguments} ${overrides}
    "-DGENDIL_DIR=${install_prefix}/share/gendil/cmake"
    "-DEXPECTED_DEPENDENCIES=${BINARY_DIR}/packages/override")
endforeach()

message(STATUS "Relative dependency hints export resolved paths, preserve cache behavior, and respect consumer overrides.")
