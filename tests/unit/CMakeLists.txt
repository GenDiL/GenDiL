
add_my_test(test-dot test-dot.cpp)
add_my_test(test-kernel-configuration test-kernel-configuration.cpp)
add_my_test(test-getpoint test-getpoint.cpp)
add_my_test(test-garbage-collector test-garbage-collector.cpp)
add_my_test(test-read-write test-read-write.cpp)
add_my_test(test-face-read-dofs-direct-global test-face-read-dofs-direct-global.cpp)
add_non_cuda_test(test-bsr-block-layout test-bsr-block-layout.cpp)
add_my_test(test-bsr-dof-layout test-bsr-dof-layout.cpp)
add_my_test(test-sgbsr-matrix test-sgbsr-matrix.cpp)
if(NOT (USE_CUDA OR USE_HIP))
  add_my_test(test-cell-iterator test-cell-iterator.cpp)
endif()
if(USE_CUDA OR USE_HIP)
  add_my_test(test-bsr-device-apply test-bsr-device-apply.cpp)
  add_my_test(test-device-kernel-configuration test-device-kernel-configuration.cpp)
  add_my_test(test-cell-iterator-device test-cell-iterator-device.cpp)
  add_my_test(test-threaded-dot-shared-arena test-threaded-dot-shared-arena.cpp)

  if(USE_CUDA)
    set(
      GENDIL_CUDA_WORK_ITEM_SYNC_TILE_SIZES
      1 2 3 4 5 7 8 9 16 25 27 32 49 64 81 125
    )

    function(gendil_probe_cuda_static_tile_size tile_size)
      set(probe_source_dir
          "${CMAKE_CURRENT_BINARY_DIR}/cuda-static-tile-${tile_size}-source")
      set(probe_binary_dir
          "${CMAKE_CURRENT_BINARY_DIR}/cuda-static-tile-${tile_size}-probe")
      file(MAKE_DIRECTORY "${probe_source_dir}")
      file(
        WRITE
        "${probe_source_dir}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(cuda_static_tile_probe LANGUAGES CUDA)\n"
        "set(CMAKE_CUDA_STANDARD 20)\n"
        "set(CMAKE_CUDA_STANDARD_REQUIRED ON)\n"
        "add_library(cuda_static_tile_${tile_size} STATIC probe.cu)\n"
      )
      file(
        WRITE
        "${probe_source_dir}/probe.cu"
        "#include <cooperative_groups.h>\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void kernel()\n"
        "{\n"
        "  auto block = cg::this_thread_block();\n"
        "  auto tile = cg::tiled_partition<${tile_size}>(block);\n"
        "  volatile unsigned int size = tile.size();\n"
        "  volatile unsigned int rank = tile.thread_rank();\n"
        "  (void)size;\n"
        "  (void)rank;\n"
        "  tile.sync();\n"
        "}\n"
      )

      set(
        probe_cmake_flags
        "-DCMAKE_CUDA_STANDARD=20"
        "-DCMAKE_CUDA_STANDARD_REQUIRED=ON"
      )
      if(CMAKE_CUDA_COMPILER)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
        )
      endif()
      if(CMAKE_CUDA_HOST_COMPILER)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
        )
      endif()
      if(CMAKE_CXX_COMPILER)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        )
      endif()
      if(CMAKE_CUDA_ARCHITECTURES)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
        )
      endif()
      if(CMAKE_BUILD_TYPE)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        )
      endif()
      if(CMAKE_CUDA_FLAGS)
        list(
          APPEND
          probe_cmake_flags
          "-DCMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}"
        )
      endif()

      try_compile(
        probe_result
        "${probe_binary_dir}"
        "${probe_source_dir}"
        cuda_static_tile_probe
        cuda_static_tile_${tile_size}
        CMAKE_FLAGS
        ${probe_cmake_flags}
        OUTPUT_VARIABLE
        probe_output
      )
      set(probe_log
          "${CMAKE_CURRENT_BINARY_DIR}/cuda-static-tile-${tile_size}.log")
      file(WRITE "${probe_log}" "${probe_output}")

      if(probe_result)
        set(
          "GENDIL_CUDA_STATIC_TILE_${tile_size}_SUPPORTED"
          1
          PARENT_SCOPE
        )
        message(
          STATUS
          "CUDA cooperative_groups::tiled_partition<${tile_size}> supported"
        )
      else()
        set(
          "GENDIL_CUDA_STATIC_TILE_${tile_size}_SUPPORTED"
          0
          PARENT_SCOPE
        )
        message(
          STATUS
          "CUDA cooperative_groups::tiled_partition<${tile_size}> unsupported; see ${probe_log}"
        )
      endif()
    endfunction()

    add_my_test(test-cuda-work-item-sync test-cuda-work-item-sync.cpp)

    set(GENDIL_CUDA_WORK_ITEM_SYNC_SUPPORTED_TILE_SIZES)
    foreach(tile_size IN LISTS GENDIL_CUDA_WORK_ITEM_SYNC_TILE_SIZES)
      gendil_probe_cuda_static_tile_size(${tile_size})
      set(
        support_variable
        "GENDIL_CUDA_STATIC_TILE_${tile_size}_SUPPORTED"
      )
      if(${support_variable})
        list(
          APPEND
          GENDIL_CUDA_WORK_ITEM_SYNC_SUPPORTED_TILE_SIZES
          ${tile_size}
        )
      endif()
      target_compile_definitions(
        test-cuda-work-item-sync
        PRIVATE
        "GENDIL_CUDA_STATIC_TILE_${tile_size}_SUPPORTED=${${support_variable}}"
      )
    endforeach()
    if(NOT GENDIL_CUDA_WORK_ITEM_SYNC_SUPPORTED_TILE_SIZES)
      message(
        WARNING
        "No CUDA cooperative_groups::tiled_partition<N> sizes were detected. "
        "The probe likely failed before testing tile-size support; inspect "
        "${CMAKE_CURRENT_BINARY_DIR}/cuda-static-tile-*.log."
      )
    endif()
  endif()

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
