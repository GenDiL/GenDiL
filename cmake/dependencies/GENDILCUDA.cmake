include_guard(GLOBAL)

function(gendil_configure_cuda target openmp_flags output_sparse_enabled)
  include(CheckCXXSourceCompiles)
  include(CheckSourceCompiles)
  include(CMakePushCheckState)

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${target} INTERFACE CUDA::cusparse)
  target_compile_features(${target} INTERFACE cuda_std_20)
  target_include_directories(
    ${target}
    INTERFACE ${CUDAToolkit_INCLUDE_DIRS}
  )

  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(
      ${target}
      INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-keep>
        $<$<COMPILE_LANGUAGE:CUDA>:-ftemplate-backtrace-limit=0>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        $<$<COMPILE_LANGUAGE:CUDA>:-ccbin=${CMAKE_CXX_COMPILER}>
    )
  endif()
  if(USE_OPENMP AND NOT "${openmp_flags}" STREQUAL "")
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      target_compile_options(
        ${target}
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${openmp_flags}>
      )
    else()
      target_compile_options(
        ${target}
        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${openmp_flags}>
      )
    endif()
  endif()

  target_compile_definitions(
    ${target} INTERFACE GENDIL_USE_CUDA GENDIL_USE_DEVICE
  )

  set(sparse_enabled OFF)
  if(NOT GENDIL_DEVICE_SPARSE_FINALIZATION STREQUAL "OFF")
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    check_source_compiles(
      CUDA
      "
      #include <cub/device/device_radix_sort.cuh>
      #include <cub/device/device_reduce.cuh>
      #include <cub/device/device_scan.cuh>
      struct Coordinate
      {
        unsigned int major;
        unsigned int minor;
        __host__ __device__ bool operator==(const Coordinate& other) const
        {
          return major == other.major && minor == other.minor;
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
        cub::DeviceRadixSort::SortPairs(
          temporary_storage, temporary_storage_bytes,
          keys, alternate_keys, values, alternate_values, 0);
        cub::DeviceReduce::ReduceByKey(
          temporary_storage, temporary_storage_bytes,
          coordinates, unique_coordinates,
          values, alternate_values, unique_count,
          cub::Sum{}, 0);
        cub::DeviceScan::InclusiveScan(
          temporary_storage, temporary_storage_bytes,
          keys, alternate_keys, cub::Max(), 0);
        return 0;
      }
      "
      GENDIL_CUDA_HAS_SPARSE_FINALIZATION_PRIMITIVES
    )
    cmake_pop_check_state()

    if(GENDIL_CUDA_HAS_SPARSE_FINALIZATION_PRIMITIVES)
      set(sparse_enabled ON)
    else()
      gendil_sparse_finalization_unavailable(
        "the CUDA compiler could not compile the required CUB radix-sort, "
        "reduce-by-key, and scan APIs."
      )
    endif()
  endif()

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES CUDA::cusparse)
  check_cxx_source_compiles(
    "
    #include <cusparse.h>
    int main()
    {
      auto function = &cusparseSpMV_preprocess;
      (void) function;
      return 0;
    }
    "
    GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS
  )
  check_cxx_source_compiles(
    "
    #include <cusparse.h>
    int main()
    {
      cusparseSpMatDescr_t descriptor = nullptr;
      auto algorithm = CUSPARSE_SPMV_BSR_ALG1;
      (void) algorithm;
      return static_cast<int>(
        cusparseCreateBsr(
          &descriptor, 1, 1, 1, 1, 1,
          nullptr, nullptr, nullptr,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
          CUSPARSE_ORDER_ROW));
    }
    "
    GENDIL_CUSPARSE_HAS_GENERIC_BSR
  )
  cmake_pop_check_state()

  if(GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS)
    target_compile_definitions(
      ${target} INTERFACE GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS
    )
  endif()
  if(GENDIL_CUSPARSE_HAS_GENERIC_BSR)
    target_compile_definitions(
      ${target} INTERFACE GENDIL_CUSPARSE_HAS_GENERIC_BSR
    )
  endif()

  set(has_float_double_spmv OFF)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.5.82")
    set(has_float_double_spmv ON)
    target_compile_definitions(
      ${target} INTERFACE GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV
    )
  endif()

  set(
    GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS
    ${GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS}
    PARENT_SCOPE
  )
  set(
    GENDIL_CUSPARSE_HAS_GENERIC_BSR
    ${GENDIL_CUSPARSE_HAS_GENERIC_BSR}
    PARENT_SCOPE
  )
  set(
    GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV
    ${has_float_double_spmv}
    PARENT_SCOPE
  )
  set(${output_sparse_enabled} ${sparse_enabled} PARENT_SCOPE)
endfunction()
