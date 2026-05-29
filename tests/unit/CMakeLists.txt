
add_my_test(test-dot test-dot.cpp)
add_my_test(test-kernel-configuration test-kernel-configuration.cpp)
add_my_test(test-getpoint test-getpoint.cpp)
add_my_test(test-garbage-collector test-garbage-collector.cpp)
add_my_test(test-read-write test-read-write.cpp)
add_my_test(test-bsr-block-layout test-bsr-block-layout.cpp)
add_my_test(test-bsr-dof-layout test-bsr-dof-layout.cpp)
add_my_test(test-sgbsr-matrix test-sgbsr-matrix.cpp)
add_my_test(test-cell-iterator test-cell-iterator.cpp)
if(USE_CUDA OR USE_HIP)
  add_my_test(test-bsr-device-apply test-bsr-device-apply.cpp)
  add_my_test(test-device-kernel-configuration test-device-kernel-configuration.cpp)
  add_my_test(test-cell-iterator-device test-cell-iterator-device.cpp)

  add_executable(
    test-device-kernel-configuration-one-index-fail
    EXCLUDE_FROM_ALL
    test-device-kernel-configuration-one-index-fail.cpp
  )
  target_link_libraries(
    test-device-kernel-configuration-one-index-fail
    PRIVATE
    GENDIL
  )
  if(USE_CUDA)
    set_source_files_properties(
      test-device-kernel-configuration-one-index-fail.cpp
      PROPERTIES
      LANGUAGE CUDA
    )
  endif()
  if(USE_HIP)
    set_source_files_properties(
      test-device-kernel-configuration-one-index-fail.cpp
      PROPERTIES
      LANGUAGE HIP
    )
  endif()
  target_include_directories(
    test-device-kernel-configuration-one-index-fail
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )
  add_test(
    NAME test-device-kernel-configuration-one-index-fail
    COMMAND
      ${CMAKE_COMMAND}
      --build ${CMAKE_BINARY_DIR}
      --target test-device-kernel-configuration-one-index-fail
  )
  set_tests_properties(
    test-device-kernel-configuration-one-index-fail
    PROPERTIES
    WILL_FAIL TRUE
  )
endif()
# add_my_test(test-gmres test-gmres.cpp)
# add_my_test(test-solvers test-solvers.cpp)
add_my_test(test-weak-form test-weak-form.cpp)
add_my_test(test-named-field-requirements test-named-field-requirements.cpp)
add_my_test(test-vector-weakform-traits test-vector-weakform-traits.cpp)
add_my_test(test-vector-expression-construction test-vector-expression-construction.cpp)
add_my_test(test-transpose-expression test-transpose-expression.cpp)
add_my_test(test-pullback-skeleton test-pullback-skeleton.cpp)
add_my_test(test-pullback-generic-cell test-pullback-generic-cell.cpp)

if(USE_OPENMP)
  add_my_test(test-openmp test-openmp.cpp)
endif()
