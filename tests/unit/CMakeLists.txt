
add_my_test(test-dot test-dot.cpp)
add_my_test(test-kernel-configuration test-kernel-configuration.cpp)
add_my_test(test-getpoint test-getpoint.cpp)
add_my_test(test-garbage-collector test-garbage-collector.cpp)
add_my_test(test-read-write test-read-write.cpp)
add_my_test(test-dof-to-quad-policies test-dof-to-quad-policies.cpp)
add_my_test(test-face-read-dofs-direct-global test-face-read-dofs-direct-global.cpp)
  add_my_test(
    test-face-read-dofs-direct-global-vector
    test-face-read-dofs-direct-global-vector.cpp
  )
add_non_cuda_test(test-bsr-block-layout test-bsr-block-layout.cpp)
add_my_test(test-bsr-dof-layout test-bsr-dof-layout.cpp)
add_my_test(test-sgbsr-matrix test-sgbsr-matrix.cpp)
add_non_cuda_test(test-raw-coo-assembly test-raw-coo-assembly.cpp)
if(NOT (USE_CUDA OR USE_HIP))
  add_my_test(test-cell-iterator test-cell-iterator.cpp)
endif()
if(USE_CUDA OR USE_HIP)
  add_my_test(test-bsr-device-apply test-bsr-device-apply.cpp)
  add_my_test(test-device-kernel-configuration test-device-kernel-configuration.cpp)
  add_my_test(test-cell-iterator-device test-cell-iterator-device.cpp)
  add_my_test(test-threaded-dot-shared-arena test-threaded-dot-shared-arena.cpp)

  add_executable(
    test-cell-iterator-filtered-sync-experiment
    EXCLUDE_FROM_ALL
    test-cell-iterator-filtered-sync-experiment.cpp
  )
  target_link_libraries(
    test-cell-iterator-filtered-sync-experiment
    PRIVATE
    GENDIL
  )
  if(USE_CUDA)
    set_source_files_properties(
      test-cell-iterator-filtered-sync-experiment.cpp
      PROPERTIES
      LANGUAGE CUDA
    )
  endif()
  if(USE_HIP)
    set_source_files_properties(
      test-cell-iterator-filtered-sync-experiment.cpp
      PROPERTIES
      LANGUAGE HIP
    )
  endif()
  target_include_directories(
    test-cell-iterator-filtered-sync-experiment
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )
  option(
    GENDIL_ENABLE_FILTERED_SYNC_EXPERIMENT_CTEST
    "Register filtered CellIterator Sync experiment in CTest."
    OFF
  )
  if(GENDIL_ENABLE_FILTERED_SYNC_EXPERIMENT_CTEST)
  add_test(
      NAME test-cell-iterator-filtered-sync-experiment
      COMMAND test-cell-iterator-filtered-sync-experiment
  )
  set_tests_properties(
      test-cell-iterator-filtered-sync-experiment
    PROPERTIES
      TIMEOUT 15
    )
  endif()

  add_my_test(
    test-device-kernel-configuration-one-index-batched
    test-device-kernel-configuration-one-index-batched.cpp
  )
endif()
# add_my_test(test-gmres test-gmres.cpp)
# add_my_test(test-solvers test-solvers.cpp)
add_non_cuda_test(test-weak-form test-weak-form.cpp)
add_non_cuda_test(test-named-field-requirements test-named-field-requirements.cpp)
add_non_cuda_test(test-vector-weakform-traits test-vector-weakform-traits.cpp)
add_non_cuda_test(test-vector-expression-construction test-vector-expression-construction.cpp)
add_non_cuda_test(test-transpose-expression test-transpose-expression.cpp)
add_non_cuda_test(test-pullback-skeleton test-pullback-skeleton.cpp)
add_non_cuda_test(test-pullback-generic-cell test-pullback-generic-cell.cpp)

if(USE_OPENMP)
  add_my_test(test-openmp test-openmp.cpp)
endif()
