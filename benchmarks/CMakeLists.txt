include("${PROJECT_SOURCE_DIR}/cmake/GENDILBenchmarkHelpers.cmake")

add_custom_target(
  benchmarks
  COMMENT "Building all benchmarks"
)

# Speed-of-light benchmarks
gendil_add_benchmark(range-benchmark-sol-3d range-benchmark-sol-3d.cpp)
gendil_add_benchmark(range-benchmark-sol-6d range-benchmark-sol-6d.cpp)


# Mass benchmarks
gendil_add_benchmark(benchmark-mass-3d benchmark-mass-3d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-1d benchmark-batched-mass-operator-1d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-2d benchmark-batched-mass-operator-2d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-3d benchmark-batched-mass-operator-3d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-4d benchmark-batched-mass-operator-4d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-5d benchmark-batched-mass-operator-5d.cpp)
gendil_add_benchmark(benchmark-batched-mass-operator-6d benchmark-batched-mass-operator-6d.cpp)
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
gendil_add_benchmark(range-benchmark-mass-3d range-benchmark-mass-3d.cpp)
gendil_add_benchmark(range-benchmark-mass-4d range-benchmark-mass-4d.cpp)
gendil_add_benchmark(range-benchmark-mass-5d range-benchmark-mass-5d.cpp)
gendil_add_benchmark(benchmark-mass-6d benchmark-mass-6d.cpp)
gendil_add_benchmark(range-benchmark-mass-6d range-benchmark-mass-6d.cpp)
# Vector Mass benchmarks
gendil_add_benchmark(benchmark-vector-mass-3d benchmark-vector-mass-3d.cpp)
gendil_add_benchmark(benchmark-vector-grad-grad-3d benchmark-vector-grad-grad-3d.cpp)
# Advection benchmarks
gendil_add_benchmark(benchmark-advection-3d benchmark-advection-3d.cpp)

function(gendil_add_batched_global_face_advection_order_benchmark dim order)
  set(dim_target benchmark-batched-global-face-advection-operator-${dim}d)
  set(order_target ${dim_target}-p${order})
  gendil_add_benchmark(${order_target} ${order_target}.cpp)
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
    gendil_add_batched_global_face_advection_order_benchmark(${dim} ${order})
  endforeach()
endforeach()
# gendil_add_benchmark(benchmark-face-read-dofs-policy-1d benchmark-face-read-dofs-policy-1d.cpp)
# gendil_add_benchmark(benchmark-face-read-dofs-policy-2d benchmark-face-read-dofs-policy-2d.cpp)
# gendil_add_benchmark(benchmark-face-read-dofs-policy-3d benchmark-face-read-dofs-policy-3d.cpp)
# gendil_add_benchmark(benchmark-face-read-dofs-policy-4d benchmark-face-read-dofs-policy-4d.cpp)
# gendil_add_benchmark(benchmark-face-read-dofs-policy-5d benchmark-face-read-dofs-policy-5d.cpp)
# gendil_add_benchmark(benchmark-face-read-dofs-policy-6d benchmark-face-read-dofs-policy-6d.cpp)
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
gendil_add_benchmark(range-benchmark-advection-3d range-benchmark-advection-3d.cpp)
gendil_add_benchmark(range-benchmark-advection-p-adaptivity-3d range-benchmark-advection-p-adaptivity-3d.cpp)
gendil_add_benchmark(range-benchmark-advection-h-adaptivity-3d range-benchmark-advection-h-adaptivity-3d.cpp)
gendil_add_benchmark(range-benchmark-advection-4d range-benchmark-advection-4d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-4d range-benchmark-advection-p-adaptivity-4d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-4d range-benchmark-advection-h-adaptivity-4d.cpp)
gendil_add_benchmark(range-benchmark-advection-5d range-benchmark-advection-5d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-5d range-benchmark-advection-p-adaptivity-5d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-5d range-benchmark-advection-h-adaptivity-5d.cpp)
gendil_add_benchmark(benchmark-advection-6d benchmark-advection-6d.cpp)
gendil_add_benchmark(range-benchmark-advection-6d range-benchmark-advection-6d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-p-adaptivity-6d range-benchmark-advection-p-adaptivity-6d.cpp)
gendil_add_non_cuda_benchmark(range-benchmark-advection-h-adaptivity-6d range-benchmark-advection-h-adaptivity-6d.cpp)
