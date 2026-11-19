# Reproduce the producer's toolchain and dependency search paths in a fresh
# project. GenDiL usage requirements must come from its installed target.
set(gendil_consumer_settings
  "${CMAKE_CURRENT_BINARY_DIR}/installed-package-settings.cmake")
file(WRITE "${gendil_consumer_settings}" "# Generated installed-consumer settings.\n")

function(gendil_consumer_setting name value cache)
  set(equals "==")
  string(FIND "${value}" "]${equals}]" delimiter)
  while(NOT delimiter EQUAL -1)
    string(APPEND equals "=")
    string(FIND "${value}" "]${equals}]" delimiter)
  endwhile()
  if(cache)
    set(suffix " CACHE STRING \"Producer toolchain or dependency setting\"")
  else()
    set(suffix "")
  endif()
  file(APPEND "${gendil_consumer_settings}"
    "set(${name} [${equals}[${value}]${equals}]${suffix})\n")
endfunction()

gendil_consumer_setting(GENDIL_PRODUCER_SOURCE "${PROJECT_SOURCE_DIR}" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_BUILD "${PROJECT_BINARY_DIR}" FALSE)
gendil_consumer_setting(GENDIL_TEST_ROOT
  "${PROJECT_BINARY_DIR}/installed-package-test" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_GENERATOR "${CMAKE_GENERATOR}" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_PLATFORM "${CMAKE_GENERATOR_PLATFORM}" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_TOOLSET "${CMAKE_GENERATOR_TOOLSET}" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}" FALSE)
gendil_consumer_setting(GENDIL_PRODUCER_DATA_DIR "${CMAKE_INSTALL_DATAROOTDIR}" FALSE)

set(gendil_consumer_backend CPU)
if(USE_CUDA)
  set(gendil_consumer_backend CUDA)
elseif(USE_HIP)
  set(gendil_consumer_backend HIP)
endif()
gendil_consumer_setting(GENDIL_TEST_BACKEND "${gendil_consumer_backend}" TRUE)

foreach(variable IN ITEMS
    CMAKE_BUILD_TYPE CMAKE_TOOLCHAIN_FILE CMAKE_MAKE_PROGRAM
    CMAKE_PREFIX_PATH CMAKE_FIND_ROOT_PATH CMAKE_SYSROOT
    CMAKE_OSX_ARCHITECTURES CMAKE_OSX_SYSROOT CMAKE_OSX_DEPLOYMENT_TARGET
    CMAKE_CUDA_HOST_COMPILER CMAKE_CUDA_ARCHITECTURES CMAKE_HIP_ARCHITECTURES
    AMDGPU_TARGETS GPU_TARGETS
    CMAKE_HIP_PLATFORM CMAKE_HIP_COMPILER_ROCM_ROOT
    CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_RELEASE
    CMAKE_EXE_LINKER_FLAGS_DEBUG CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
    CUDAToolkit_ROOT HIP_ROOT_DIR ROCM_ROOT_DIR HIP_DIR hip_DIR
    HIPBLAS_DIR HIPSPARSE_DIR rocsparse_DIR rocprim_DIR
    MFEM_DIR HYPRE_DIR caliper_DIR RAJA_DIR
    OpenMP_ROOT OpenMP_C_FLAGS OpenMP_CXX_FLAGS
    OpenMP_C_LIB_NAMES OpenMP_CXX_LIB_NAMES)
  if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
    gendil_consumer_setting(${variable} "${${variable}}" TRUE)
  endif()
endforeach()
foreach(language IN ITEMS C CXX)
  foreach(library IN LISTS OpenMP_${language}_LIB_NAMES)
    set(variable "OpenMP_${library}_LIBRARY")
    if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
      gendil_consumer_setting(${variable} "${${variable}}" TRUE)
    endif()
  endforeach()
endforeach()
foreach(language IN ITEMS C CXX CUDA HIP)
  foreach(suffix IN ITEMS COMPILER COMPILER_ARG1 COMPILER_TARGET COMPILER_LAUNCHER
                          FLAGS FLAGS_DEBUG FLAGS_RELEASE FLAGS_RELWITHDEBINFO)
    set(variable "CMAKE_${language}_${suffix}")
    if(DEFINED ${variable} AND NOT "${${variable}}" STREQUAL "")
      gendil_consumer_setting(${variable} "${${variable}}" TRUE)
    endif()
  endforeach()
endforeach()

add_test(
  NAME test-installed-package
  COMMAND "${CMAKE_COMMAND}"
    "-DPRODUCER_SETTINGS=${gendil_consumer_settings}"
    "-DBUILD_CONFIG=$<CONFIG>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/test-installed-package.cmake"
)
set_tests_properties(test-installed-package PROPERTIES
  LABELS package
  RUN_SERIAL TRUE
)
