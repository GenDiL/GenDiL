
gendil_add_test(test-generalized-interior-residual test-generalized-interior-residual.cpp)
gendil_add_test(test-upwind-transported-fields test-upwind-transported-fields.cpp)
gendil_add_test(test-generalized-partition-facets test-generalized-partition-facets.cpp)
gendil_add_test(test-boundary-upwind test-boundary-upwind.cpp)
gendil_add_test(test-execution-context test-execution-context.cpp)
gendil_add_test(test-cell-invocation-validation test-cell-invocation-validation.cpp NO_CTEST)
foreach(mode input-replace input-add direct-input output-replace output-add
             load-replace load-add direct-output direct-load domain-topology)
  if(mode STREQUAL "domain-topology")
    set(diagnostic "Active finite-element field is not compatible with the selected integration-domain cell topology.")
  elseif(mode MATCHES "input")
    set(diagnostic "Weak-form input vector is smaller than its required algebraic extent.")
  else()
    set(diagnostic "Weak-form output vector is smaller than its required algebraic extent.")
  endif()
  add_test(NAME test-cell-invocation-${mode}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-cell-invocation-validation>
      -DMODE=${mode} "-DEXPECTED_TEXT=${diagnostic}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
endforeach()

gendil_add_test(test-boundary-invocation-validation test-cell-invocation-validation.cpp NO_CTEST)
target_compile_definitions(test-boundary-invocation-validation PRIVATE GENDIL_TEST_BOUNDARY_INVOCATION)
foreach(mode input-replace input-add direct-input output-replace output-add
             load-replace load-add direct-output direct-load domain-topology)
  if(mode STREQUAL "domain-topology")
    set(diagnostic "Active finite-element field is not compatible with the selected integration-domain cell topology.")
  elseif(mode MATCHES "input")
    set(diagnostic "Weak-form input vector is smaller than its required algebraic extent.")
  else()
    set(diagnostic "Weak-form output vector is smaller than its required algebraic extent.")
  endif()
  add_test(NAME test-boundary-invocation-${mode}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-boundary-invocation-validation>
      -DMODE=${mode} "-DEXPECTED_TEXT=${diagnostic}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
endforeach()

gendil_add_test(test-dot test-dot.cpp)
gendil_add_test(test-vector-l2-error test-vector-l2-error.cpp)
gendil_add_test(test-divergence-pullback test-divergence-pullback.cpp)
gendil_add_test(test-curl-pullback test-curl-pullback.cpp)
gendil_add_test(test-threaded-view-indexing test-threaded-view-indexing.cpp)
gendil_add_test(test-kernel-configuration test-kernel-configuration.cpp)
gendil_add_test(test-getpoint test-getpoint.cpp)
gendil_add_test(test-point-array-origin test-point-array-origin.cpp)
gendil_add_test(test-garbage-collector test-garbage-collector.cpp)
gendil_add_test(test-read-write test-read-write.cpp)
gendil_add_test(test-vector-access test-vector-access.cpp)
gendil_add_test(test-container-access test-container-access.cpp)
gendil_add_test(test-vector-operations test-vector-operations.cpp TIMEOUT 60)
gendil_add_test(test-extent-validation test-extent-validation.cpp NO_CTEST)
if(TARGET test-extent-validation)
  foreach(case negative query_negative_after_zero query_overflow
               narrowing signed_narrowing array_negative product_overflow
               negative_stride negative_offset strided_offset_overflow indirected_negative
               fifo_negative_after_zero fifo_overflow_before_zero permuted_overflow_before_zero
               oriented_reverse_overflow oriented_base_offset_overflow
               oriented_stride_overflow oriented_product_overflow
               oriented_zero_dimension oriented_pair_zero_dimension)
    if(case STREQUAL "strided_offset_overflow")
      set(extent_diagnostic "Strided layout offset overflows")
    elseif(case STREQUAL "fifo_overflow_before_zero")
      set(extent_diagnostic "FIFO strided layout stride overflows")
    elseif(case STREQUAL "permuted_overflow_before_zero")
      set(extent_diagnostic "Permuted strided layout size overflows")
    elseif(case STREQUAL "oriented_reverse_overflow")
      set(extent_diagnostic "Oriented layout reverse offset overflows")
    elseif(case STREQUAL "oriented_base_offset_overflow")
      set(extent_diagnostic "Oriented layout base offset overflows")
    elseif(case STREQUAL "oriented_stride_overflow")
      set(extent_diagnostic "Oriented layout stride does not fit in int")
    elseif(case STREQUAL "oriented_product_overflow")
      set(extent_diagnostic "Oriented layout extent product overflows")
    elseif(case STREQUAL "oriented_zero_dimension" OR case STREQUAL "oriented_pair_zero_dimension")
      set(extent_diagnostic "Oriented layout dimensions must be positive")
    else()
      set(extent_diagnostic "extent")
    endif()
    add_test(NAME test-extents-reject-${case}
      COMMAND ${CMAKE_COMMAND}
        -DTEST_PROGRAM=$<TARGET_FILE:test-extent-validation>
        -DMODE=${case}
        "-DEXPECTED_TEXT=${extent_diagnostic}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  endforeach()
endif()
if(TARGET test-extent-validation)
  add_test(NAME test-vector-view-rejects-negative-extent
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-extent-validation>
      -DMODE=negative_vector_extent
      "-DEXPECTED_TEXT=View extents must be nonnegative"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  add_test(
    NAME test-vector-view-rejects-invalid-range
    COMMAND
      ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-extent-validation>
      -DMODE=invalid_vector_range
      "-DEXPECTED_TEXT=Vector subview range exceeds its parent vector"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake
  )
endif()
gendil_add_non_nvcc_test(test-gmres-variants test-gmres-variants.cpp)
gendil_add_non_nvcc_test(test-gmres-borrowed-host test-gmres-borrowed-host.cpp)
gendil_add_non_nvcc_test(test-gmres-borrowed-host-release test-gmres-borrowed-host.cpp)
if(TARGET test-gmres-borrowed-host)
  target_compile_definitions(test-gmres-borrowed-host PRIVATE GENDIL_TEST_FORCE_DEBUG)
  target_compile_definitions(test-gmres-borrowed-host-release PRIVATE NDEBUG)
  foreach(target test-gmres-borrowed-host test-gmres-borrowed-host-release)
  foreach(mode short-rotations short-sines short-rhs short-coefficients short-basis short-work short-Hrows restart short-directions)
    add_test(NAME ${target}-rejects-${mode}
      COMMAND ${CMAKE_COMMAND}
        -DTEST_PROGRAM=$<TARGET_FILE:${target}>
        -DMODE=${mode}
        "-DEXPECTED_TEXT=GMRES"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  endforeach()
  endforeach()
endif()
if(NOT (USE_CUDA OR USE_HIP))
  gendil_add_test(test-gmres-core test-gmres-core.cpp TIMEOUT 60)
endif()

# Shared arithmetic is independently includable; no solver header is needed.
gendil_add_test(test-vector-reductions test-vector-reductions.cpp)
foreach(mode missing-dot-scratch missing-norm-scratch missing-project-scratch
             missing-combine-scratch projection-size-overflow)
  add_test(NAME test-vector-reductions-rejects-${mode}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-vector-reductions>
      -DMODE=${mode}
      "-DEXPECTED_TEXT=scratch"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
endforeach()
if(USE_OPENMP AND NOT (USE_CUDA OR USE_HIP))
  gendil_add_test(test-gmres-cooperative test-gmres-cooperative.cpp TIMEOUT 60)
endif()

gendil_add_test(test-kernel-loop test-kernel-loop.cpp)
gendil_add_test(test-atomic-add test-atomic-add.cpp)
gendil_add_non_nvcc_test(test-checked-arithmetic test-checked-arithmetic.cpp)
gendil_add_non_nvcc_test(test-checked-arithmetic-release test-checked-arithmetic.cpp)
if(TARGET test-checked-arithmetic)
  target_compile_definitions(test-checked-arithmetic PRIVATE GENDIL_TEST_FORCE_DEBUG)
  target_compile_definitions(test-checked-arithmetic-release PRIVATE NDEBUG)
  add_test(NAME test-assert-evaluates-failure-once
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-checked-arithmetic>
      -DMODE=assert-evaluation-failure
      "-DEXPECTED_TEXT=Assertion predicate and message evaluated once."
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  foreach(target test-checked-arithmetic test-checked-arithmetic-release)
    add_test(NAME ${target}-verify-evaluates-failure-once
      COMMAND ${CMAKE_COMMAND}
        -DTEST_PROGRAM=$<TARGET_FILE:${target}>
        -DMODE=verify-evaluation-failure
        "-DEXPECTED_TEXT=Verification predicate and message evaluated once."
        -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  endforeach()
  foreach(operation add multiply)
    if(operation STREQUAL "add")
      set(message "CheckedAdd overflow test")
    else()
      set(message "CheckedMultiply overflow test")
    endif()
    add_test(NAME test-checked-${operation}-release-rejects-overflow
      COMMAND ${CMAKE_COMMAND}
        -DTEST_PROGRAM=$<TARGET_FILE:test-checked-arithmetic-release>
        -DMODE=${operation}-overflow
        "-DEXPECTED_TEXT=${message}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
  endforeach()
  add_test(
    NAME test-checked-add-rejects-overflow
    COMMAND
      ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-checked-arithmetic>
      -DMODE=add-overflow
      "-DEXPECTED_TEXT=CheckedAdd overflow test"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake
  )
  add_test(
    NAME test-checked-multiply-rejects-overflow
    COMMAND
      ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-checked-arithmetic>
      -DMODE=multiply-overflow
      "-DEXPECTED_TEXT=CheckedMultiply overflow test"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake
  )
endif()
gendil_add_test(test-dof-to-quad-policies test-dof-to-quad-policies.cpp)
gendil_add_test(test-facet-quad-data test-facet-quad-data.cpp)
gendil_add_test(test-restriction-layout test-restriction-layout.cpp)
gendil_add_test(
  test-weighted-restriction-operations
  test-weighted-restriction-operations.cpp
)
gendil_add_non_nvcc_test(
  test-p-nonconforming-fixed-stencil
  test-p-nonconforming-fixed-stencil.cpp
)
gendil_add_test(test-structural-broken-space-restriction test-structural-broken-space-restriction.cpp)
gendil_add_test(test-cell-reference-geometry test-cell-reference-geometry.cpp)
gendil_add_test(test-orientation test-orientation.cpp)
gendil_add_test(test-face-read-dofs-direct-global test-face-read-dofs-direct-global.cpp)
gendil_add_non_nvcc_test(test-face-read-dofs-direct-global-vector test-face-read-dofs-direct-global-vector.cpp)
gendil_add_test(test-generic-oriented-dof-view test-generic-oriented-dof-view.cpp)
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
gendil_add_test(test-conjugate-gradient test-conjugate-gradient.cpp)
gendil_add_test(test-bicgstab test-bicgstab.cpp)
foreach(mode negative-tolerance nan-tolerance infinite-tolerance
             single-negative-tolerance single-nan-tolerance single-infinite-tolerance)
  add_test(NAME test-bicgstab-rejects-${mode}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-bicgstab>
      -DMODE=${mode}
      "-DEXPECTED_TEXT=BiCGSTAB requires a finite, nonnegative tolerance."
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
endforeach()
gendil_add_test(test-iterative-solver-utilities test-iterative-solver-utilities.cpp)
foreach(mode negative-tolerance nan-tolerance infinite-tolerance size-mismatch)
  if(mode STREQUAL "size-mismatch")
    set(diagnostic "CG requires matching RHS and solution sizes.")
  else()
    set(diagnostic "CG requires a finite, nonnegative tolerance.")
  endif()
  add_test(NAME test-conjugate-gradient-rejects-${mode}
    COMMAND ${CMAKE_COMMAND}
      -DTEST_PROGRAM=$<TARGET_FILE:test-conjugate-gradient>
      -DMODE=${mode} "-DEXPECTED_TEXT=${diagnostic}"
      -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_runtime_failure.cmake)
endforeach()
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
