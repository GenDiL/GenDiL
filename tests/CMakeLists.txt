cmake_minimum_required(VERSION 3.12)

# Add a serial test
macro(add_my_test target_name source_file)
  add_executable(${target_name})

  target_link_libraries(${target_name} PRIVATE GENDIL)

  target_sources(
    ${target_name}
    PRIVATE
    ${source_file}
  )

  if (USE_CUDA)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
  endif()
  if(USE_HIP)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
  endif()

  target_include_directories(
    ${target_name}
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )

  add_test(NAME ${target_name} COMMAND ${target_name})
endmacro()

# Add a parallel test
macro(add_my_mpi_test target_name source_file num_mpi_ranks)
  add_executable(${target_name})

  target_link_libraries(${target_name} PRIVATE GENDIL)

  target_sources(
    ${target_name}
    PRIVATE
    ${source_file}
  )

  if (USE_CUDA)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
  endif()

  target_include_directories(
    ${target_name}
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )

  add_test(
    NAME ${target_name}
    COMMAND ${MPIEXEC} ${MPIEXEC_TEST_ARGS} ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_ranks} ${target_name}
  )

  set_tests_properties(${target_name} PROPERTIES PROCESSORS ${num_mpi_ranks})
endmacro()

add_subdirectory(unit)
add_subdirectory(integration)
