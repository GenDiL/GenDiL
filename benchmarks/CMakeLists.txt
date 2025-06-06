cmake_minimum_required(VERSION 3.12)

macro(add_my_benchmark target_name source_file)
  add_executable(${target_name})

  target_link_libraries(${target_name} PRIVATE GENDIL)

  add_dependencies(
    benchmarks
    ${target_name}
  )

  target_sources(
    ${target_name}
    PRIVATE
    ${source_file}
  )

  if(USE_CUDA)
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

  if(caliper_FOUND)
    set_tests_properties(${target_name} PROPERTIES ENVIRONMENT "CALI_CONFIG=runtime-report")
  endif()

endmacro()

add_custom_target(
  benchmarks
  ALL # OPTIONAL: Use ALL to make the target always built
  COMMENT "Building all benchmarks"
)

add_my_benchmark(range-benchmark-sol-3d range-benchmark-sol-3d.cpp)
add_my_benchmark(range-benchmark-sol-6d range-benchmark-sol-6d.cpp)
add_my_benchmark(benchmark-mass-3d benchmark-mass-3d.cpp)
add_my_benchmark(range-benchmark-mass-3d range-benchmark-mass-3d.cpp)
add_my_benchmark(range-benchmark-mass-4d range-benchmark-mass-4d.cpp)
add_my_benchmark(range-benchmark-mass-5d range-benchmark-mass-5d.cpp)
add_my_benchmark(benchmark-mass-6d benchmark-mass-6d.cpp)
add_my_benchmark(range-benchmark-mass-6d range-benchmark-mass-6d.cpp)
add_my_benchmark(benchmark-advection-3d benchmark-advection-3d.cpp)
add_my_benchmark(range-benchmark-advection-3d range-benchmark-advection-3d.cpp)
add_my_benchmark(range-benchmark-advection-4d range-benchmark-advection-4d.cpp)
add_my_benchmark(range-benchmark-advection-5d range-benchmark-advection-5d.cpp)
add_my_benchmark(benchmark-advection-6d benchmark-advection-6d.cpp)
add_my_benchmark(range-benchmark-advection-6d range-benchmark-advection-6d.cpp)
