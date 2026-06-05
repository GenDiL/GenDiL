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

endmacro()

add_custom_target(
  benchmarks
  ALL # OPTIONAL: Use ALL to make the target always built
  COMMENT "Building all benchmarks"
)

macro(add_non_cuda_benchmark target_name source_file)
  if(NOT USE_CUDA)
    add_my_benchmark(${target_name} ${source_file})
  endif()
endmacro()

# Speed-of-light benchmarks
add_my_benchmark(range-benchmark-sol-3d range-benchmark-sol-3d.cpp)
add_my_benchmark(range-benchmark-sol-6d range-benchmark-sol-6d.cpp)
# Mass benchmarks
add_my_benchmark(benchmark-mass-3d benchmark-mass-3d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-1d benchmark-batched-mass-operator-1d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-2d benchmark-batched-mass-operator-2d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-3d benchmark-batched-mass-operator-3d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-4d benchmark-batched-mass-operator-4d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-5d benchmark-batched-mass-operator-5d.cpp)
add_my_benchmark(benchmark-batched-mass-operator-6d benchmark-batched-mass-operator-6d.cpp)
add_custom_target(benchmark-batched-mass-operator)
add_dependencies(
  benchmark-batched-mass-operator
  benchmark-batched-mass-operator-1d
  benchmark-batched-mass-operator-2d
  benchmark-batched-mass-operator-3d
  benchmark-batched-mass-operator-4d
  benchmark-batched-mass-operator-5d
  benchmark-batched-mass-operator-6d
)
add_my_benchmark(range-benchmark-mass-3d range-benchmark-mass-3d.cpp)
add_my_benchmark(range-benchmark-mass-4d range-benchmark-mass-4d.cpp)
add_my_benchmark(range-benchmark-mass-5d range-benchmark-mass-5d.cpp)
add_my_benchmark(benchmark-mass-6d benchmark-mass-6d.cpp)
add_my_benchmark(range-benchmark-mass-6d range-benchmark-mass-6d.cpp)
# Vector Mass benchmarks
add_my_benchmark(benchmark-vector-mass-3d benchmark-vector-mass-3d.cpp)
add_my_benchmark(benchmark-vector-grad-grad-3d benchmark-vector-grad-grad-3d.cpp)
# Advection benchmarks
add_my_benchmark(benchmark-advection-3d benchmark-advection-3d.cpp)

function(add_batched_global_face_advection_order_benchmark dim order)
  set(dim_target benchmark-batched-global-face-advection-operator-${dim}d)
  set(order_target ${dim_target}-p${order})
  add_my_benchmark(${order_target} ${order_target}.cpp)
  add_dependencies(${dim_target} ${order_target})
endfunction()

add_custom_target(benchmark-batched-global-face-advection-operator)
foreach(dim RANGE 1 6)
  set(dim_target benchmark-batched-global-face-advection-operator-${dim}d)
  add_custom_target(${dim_target})
  add_dependencies(
    benchmark-batched-global-face-advection-operator
    ${dim_target}
  )
  foreach(order 0 1 2 3)
    add_batched_global_face_advection_order_benchmark(${dim} ${order})
  endforeach()
endforeach()
# add_my_benchmark(benchmark-face-read-dofs-policy-1d benchmark-face-read-dofs-policy-1d.cpp)
# add_my_benchmark(benchmark-face-read-dofs-policy-2d benchmark-face-read-dofs-policy-2d.cpp)
# add_my_benchmark(benchmark-face-read-dofs-policy-3d benchmark-face-read-dofs-policy-3d.cpp)
# add_my_benchmark(benchmark-face-read-dofs-policy-4d benchmark-face-read-dofs-policy-4d.cpp)
# add_my_benchmark(benchmark-face-read-dofs-policy-5d benchmark-face-read-dofs-policy-5d.cpp)
# add_my_benchmark(benchmark-face-read-dofs-policy-6d benchmark-face-read-dofs-policy-6d.cpp)
# add_custom_target(benchmark-face-read-dofs-policy)
# add_dependencies(
#   benchmark-face-read-dofs-policy
#   benchmark-face-read-dofs-policy-1d
#   benchmark-face-read-dofs-policy-2d
#   benchmark-face-read-dofs-policy-3d
#   benchmark-face-read-dofs-policy-4d
#   benchmark-face-read-dofs-policy-5d
#   benchmark-face-read-dofs-policy-6d
# )
add_my_benchmark(range-benchmark-advection-3d range-benchmark-advection-3d.cpp)
add_my_benchmark(range-benchmark-advection-p-adaptivity-3d range-benchmark-advection-p-adaptivity-3d.cpp)
add_my_benchmark(range-benchmark-advection-h-adaptivity-3d range-benchmark-advection-h-adaptivity-3d.cpp)
add_my_benchmark(range-benchmark-advection-4d range-benchmark-advection-4d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-4d range-benchmark-advection-p-adaptivity-4d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-4d range-benchmark-advection-h-adaptivity-4d.cpp)
add_my_benchmark(range-benchmark-advection-5d range-benchmark-advection-5d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-5d range-benchmark-advection-p-adaptivity-5d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-5d range-benchmark-advection-h-adaptivity-5d.cpp)
add_my_benchmark(benchmark-advection-6d benchmark-advection-6d.cpp)
add_my_benchmark(range-benchmark-advection-6d range-benchmark-advection-6d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-6d range-benchmark-advection-p-adaptivity-6d.cpp)
add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-6d range-benchmark-advection-h-adaptivity-6d.cpp)
