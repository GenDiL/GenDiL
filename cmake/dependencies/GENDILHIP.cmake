include_guard(GLOBAL)

function(gendil_configure_hip target output_sparse_enabled)
  include(CheckCXXSourceCompiles)
  include(CheckSourceCompiles)
  include(CMakePushCheckState)

  find_package(HIP REQUIRED)
  find_package(rocsparse REQUIRED CONFIG)
  target_link_libraries(${target} INTERFACE roc::rocsparse)
  target_include_directories(${target} INTERFACE ${hip_INCLUDE_DIRS})
  target_compile_options(
    ${target}
    INTERFACE
      $<$<COMPILE_LANGUAGE:HIP>:-ftemplate-backtrace-limit=0>
  )
  target_compile_definitions(
    ${target} INTERFACE GENDIL_USE_HIP GENDIL_USE_DEVICE
  )

  set(sparse_enabled OFF)
  if(NOT GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "OFF")
    find_package(rocprim CONFIG QUIET)
    if(NOT TARGET roc::rocprim AND GENDIL_FETCH_ROCPRIM)
      include(FetchContent)

      # CMP0077 lets these function-local normal variables initialize rocPRIM's
      # options without leaking generic cache entries into the parent project.
      set(BUILD_TEST OFF)
      set(BUILD_BENCHMARK OFF)
      set(BUILD_EXAMPLE OFF)
      set(ROCPRIM_BUILD_TESTS OFF)
      set(ROCPRIM_BUILD_BENCHMARKS OFF)
      set(ROCPRIM_BUILD_EXAMPLES OFF)

      FetchContent_Declare(
        rocprim
        GIT_REPOSITORY https://github.com/ROCm/rocPRIM.git
        GIT_TAG ${GENDIL_ROCPRIM_GIT_TAG}
        GIT_SHALLOW TRUE
      )
      FetchContent_MakeAvailable(rocprim)
      if(TARGET rocprim AND NOT TARGET roc::rocprim)
        add_library(roc::rocprim ALIAS rocprim)
      endif()
    endif()

    if(TARGET roc::rocprim)
      cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_LIBRARIES roc::rocprim)
      check_source_compiles(
        HIP
        "
        #include <rocprim/device/device_radix_sort.hpp>
        #include <rocprim/device/device_reduce_by_key.hpp>
        #include <rocprim/device/device_scan.hpp>
        struct Coordinate
        {
          unsigned int major;
          unsigned int minor;
        };
        struct CoordinateEqual
        {
          __host__ __device__ bool operator()(
            const Coordinate& lhs, const Coordinate& rhs) const
          {
            return lhs.major == rhs.major && lhs.minor == rhs.minor;
          }
        };
        int main()
        {
          void * temporary_storage = nullptr;
          size_t temporary_storage_bytes = 0;
          unsigned int * keys = nullptr;
          unsigned int * alternate_keys = nullptr;
          double * values = nullptr;
          double * alternate_values = nullptr;
          Coordinate * coordinates = nullptr;
          Coordinate * unique_coordinates = nullptr;
          unsigned int * unique_count = nullptr;
          rocprim::radix_sort_pairs(
            temporary_storage, temporary_storage_bytes,
            keys, alternate_keys, values, alternate_values, size_t(0));
          rocprim::reduce_by_key(
            temporary_storage, temporary_storage_bytes,
            coordinates, values, size_t(0),
            unique_coordinates, alternate_values,
            unique_count, rocprim::plus<double>{}, CoordinateEqual{});
          rocprim::inclusive_scan(
            temporary_storage, temporary_storage_bytes,
            keys, alternate_keys, size_t(0),
            rocprim::maximum<unsigned int>{});
          return 0;
        }
        "
        GENDIL_HIP_HAS_SPARSE_FINALIZATION_PRIMITIVES
      )
      cmake_pop_check_state()
    endif()

    if(TARGET roc::rocprim AND
       GENDIL_HIP_HAS_SPARSE_FINALIZATION_PRIMITIVES)
      target_link_libraries(
        ${target}
        INTERFACE
          $<BUILD_INTERFACE:roc::rocprim>
          $<INSTALL_INTERFACE:roc::rocprim>
      )
      set(sparse_enabled ON)
    elseif(NOT TARGET roc::rocprim)
      gendil_sparse_finalization_unavailable(
        "rocPRIM was not found. Install its CMake package or configure with "
        "GENDIL_FETCH_ROCPRIM=ON."
      )
    else()
      gendil_sparse_finalization_unavailable(
        "the HIP compiler could not compile the required rocPRIM radix-sort, "
        "reduce-by-key, and scan APIs."
      )
    endif()
  endif()

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES roc::rocsparse)
  check_cxx_source_compiles(
    "
    #include <rocsparse/rocsparse.h>
    int main()
    {
      rocsparse_spmat_descr descriptor = nullptr;
      auto algorithm = rocsparse_spmv_alg_bsr;
      (void) algorithm;
      return static_cast<int>(
        rocsparse_create_bsr_descr(
          &descriptor, 1, 1, 1,
          rocsparse_direction_row, 1,
          nullptr, nullptr, nullptr,
          rocsparse_indextype_i32, rocsparse_indextype_i32,
          rocsparse_index_base_zero, rocsparse_datatype_f64_r));
    }
    "
    GENDIL_ROCSPARSE_HAS_GENERIC_BSR
  )
  cmake_pop_check_state()
  if(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
    target_compile_definitions(
      ${target} INTERFACE GENDIL_ROCSPARSE_HAS_GENERIC_BSR
    )
  endif()

  set(
    GENDIL_ROCSPARSE_HAS_GENERIC_BSR
    ${GENDIL_ROCSPARSE_HAS_GENERIC_BSR}
    PARENT_SCOPE
  )
  set(${output_sparse_enabled} ${sparse_enabled} PARENT_SCOPE)
  message(STATUS "HIP support enabled!")
endfunction()
