
gendil_add_test(test-dot test-dot.cpp)
gendil_add_test(test-kernel-configuration test-kernel-configuration.cpp)
gendil_add_test(test-getpoint test-getpoint.cpp)
gendil_add_test(test-garbage-collector test-garbage-collector.cpp)
gendil_add_test(test-read-write test-read-write.cpp)
gendil_add_test(test-vector-access test-vector-access.cpp)
gendil_add_test(test-kernel-loop test-kernel-loop.cpp)
gendil_add_test(test-atomic-add test-atomic-add.cpp)
gendil_add_test(test-dof-to-quad-policies test-dof-to-quad-policies.cpp)
gendil_add_test(test-facet-quad-data test-facet-quad-data.cpp)
gendil_add_test(test-cell-reference-geometry test-cell-reference-geometry.cpp)
gendil_add_test(test-face-read-dofs-direct-global test-face-read-dofs-direct-global.cpp)
gendil_add_test(test-face-read-dofs-direct-global-vector test-face-read-dofs-direct-global-vector.cpp)
gendil_add_non_nvcc_test(test-bsr-block-layout test-bsr-block-layout.cpp)
gendil_add_test(test-bsr-dof-layout test-bsr-dof-layout.cpp)
gendil_add_non_nvcc_test(test-sgbsr-matrix test-sgbsr-matrix.cpp)
gendil_add_non_nvcc_test(test-matrix-assembly-defaults test-matrix-assembly-defaults.cpp)
gendil_add_non_nvcc_test(test-element-block-diagonal-assembly test-element-block-diagonal-assembly.cpp)
gendil_add_non_nvcc_test(test-coo-matrix test-coo-matrix.cpp)
gendil_add_non_nvcc_test(test-csr-matrix test-csr-matrix.cpp)
gendil_add_non_nvcc_test(test-csc-matrix test-csc-matrix.cpp)
gendil_add_non_nvcc_test(test-sparse-matrix-apply-arithmetic test-sparse-matrix-apply-arithmetic.cpp)
gendil_add_non_nvcc_test(test-sparse-matrix-apply-add test-sparse-matrix-apply-add.cpp)
gendil_add_non_nvcc_test(test-raw-coo-assembly test-raw-coo-assembly.cpp)
gendil_add_test(
  test-device-sparse-finalization
  test-device-sparse-finalization.cpp
)
gendil_add_test(
  test-sparse-storage-ownership
  test-sparse-storage-ownership.cpp
)
if(USE_HYPRE)
  gendil_add_test(test-hypre-csr-matrix test-hypre-csr-matrix.cpp)
endif()
if(NOT (USE_CUDA OR USE_HIP))
  gendil_add_test(test-cell-iterator test-cell-iterator.cpp)
endif()
if(USE_CUDA OR USE_HIP)
  gendil_add_test(test-bsr-device-apply test-bsr-device-apply.cpp)
  gendil_add_test(test-coo-device-apply test-coo-device-apply.cpp)
  gendil_add_test(test-csr-device-apply test-csr-device-apply.cpp)
  gendil_add_test(test-csc-device-apply test-csc-device-apply.cpp)
  gendil_add_test(
    test-sgbsr-device-gather-scatter
    test-sgbsr-device-gather-scatter.cpp
  )
  gendil_add_test(
    test-vendor-sparse-device-apply
    test-vendor-sparse-device-apply.cpp
  )
  gendil_add_test(test-device-kernel-configuration test-device-kernel-configuration.cpp)
  gendil_add_test(test-cell-iterator-device test-cell-iterator-device.cpp)
  gendil_add_test(test-threaded-dot-shared-arena test-threaded-dot-shared-arena.cpp)

  if(GENDIL_ENABLE_FILTERED_SYNC_EXPERIMENT_CTEST)
    gendil_add_test(
    test-cell-iterator-filtered-sync-experiment
      test-cell-iterator-filtered-sync-experiment.cpp
      EXCLUDE_FROM_ALL
      TIMEOUT 15
    )
  else()
    gendil_add_test(
      test-cell-iterator-filtered-sync-experiment
      test-cell-iterator-filtered-sync-experiment.cpp
      EXCLUDE_FROM_ALL
      NO_CTEST
    )
  endif()

  gendil_add_test(
    test-device-kernel-configuration-one-index-batched
    test-device-kernel-configuration-one-index-batched.cpp
  )
endif()
# gendil_add_test(test-gmres test-gmres.cpp)
# gendil_add_test(test-solvers test-solvers.cpp)
gendil_add_non_nvcc_test(test-weak-form test-weak-form.cpp)
gendil_add_non_nvcc_test(test-vector-h1-generic-operator test-vector-h1-generic-operator.cpp)
gendil_add_non_nvcc_test(test-named-field-requirements test-named-field-requirements.cpp)
gendil_add_non_nvcc_test(test-vector-weakform-traits test-vector-weakform-traits.cpp)
gendil_add_non_nvcc_test(test-vector-expression-construction test-vector-expression-construction.cpp)
gendil_add_non_nvcc_test(test-transpose-expression test-transpose-expression.cpp)
gendil_add_non_nvcc_test(test-pullback-skeleton test-pullback-skeleton.cpp)
gendil_add_non_nvcc_test(test-pullback-generic-cell test-pullback-generic-cell.cpp)

if(USE_OPENMP)
  gendil_add_test(test-openmp test-openmp.cpp)
endif()
