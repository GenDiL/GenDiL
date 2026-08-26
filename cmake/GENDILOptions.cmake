include_guard(GLOBAL)

option(USE_MFEM "Enables MFEM interface." OFF)
option(USE_OPENMP "Enables OpenMP parallelization features." ON)
option(USE_CUDA "Enables CUDA support." OFF)
option(USE_HIP "Enables HIP support." OFF)
option(USE_RAJA "Enables RAJA features." OFF)
option(USE_CALIPER "Enables Caliper interface for benchmarking." OFF)
option(USE_HYPRE "Enables Hypre interface." OFF)
option(GENDIL_ENABLE_BENCHMARKS "Build GenDiL benchmark executables" OFF)
option(
  GENDIL_ENABLE_FILTERED_SYNC_EXPERIMENT_CTEST
  "Register filtered CellIterator Sync experiment in CTest."
  OFF
)

set(
  GENDIL_DEVICE_SPARSE_FINALIZATION
  "AUTO"
  CACHE STRING
  "GPU sparse finalization support: AUTO, ON, or OFF"
)
set_property(
  CACHE GENDIL_DEVICE_SPARSE_FINALIZATION
  PROPERTY STRINGS AUTO ON OFF
)
option(
  GENDIL_FETCH_ROCPRIM
  "Fetch rocPRIM when HIP sparse finalization cannot find it"
  OFF
)
set(
  GENDIL_ROCPRIM_GIT_TAG
  "rocm-6.3.1"
  CACHE STRING
  "rocPRIM git tag or branch used by GENDIL_FETCH_ROCPRIM"
)

string(
  TOUPPER
  "${GENDIL_DEVICE_SPARSE_FINALIZATION}"
  GENDIL_DEVICE_SPARSE_FINALIZATION
)
if(NOT GENDIL_DEVICE_SPARSE_FINALIZATION MATCHES "^(AUTO|ON|OFF)$")
  message(
    FATAL_ERROR
    "GENDIL_DEVICE_SPARSE_FINALIZATION must be AUTO, ON, or OFF (received "
    "'${GENDIL_DEVICE_SPARSE_FINALIZATION}')."
  )
endif()

set(GENDIL_USE_HYPRE_DEVICE OFF)
set(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION OFF)
set(GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV OFF)
