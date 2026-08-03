cmake_minimum_required(VERSION 3.25.2)

project(
  "GENDIL"
  VERSION 0.0.5
  DESCRIPTION "A C++ Generic Discretization Library."
  HOMEPAGE_URL "https://github.com/GenDiL/GenDiL"
)

include(GNUInstallDirs)
include(CTest)
include("${PROJECT_SOURCE_DIR}/cmake/GENDILOptions.cmake")
include(
  "${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILDeviceSparseFinalization.cmake"
)
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILOpenMP.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILRAJA.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILMFEM.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILCaliper.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILHypre.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILCUDA.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/dependencies/GENDILHIP.cmake")

# Set optimization flags for different build types.
set(
  CMAKE_CXX_FLAGS_DEBUG
  "-g -fsanitize=address -fno-omit-frame-pointer"
  CACHE STRING ""
)
set(
  CMAKE_CXX_FLAGS_RELEASE
  "-O3 -march=native -funroll-loops -DNDEBUG"
  CACHE STRING ""
)
set(
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "-O2 -DNDEBUG -g"
  CACHE STRING ""
)

# Print warnings.
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -pedantic -Wall -ftemplate-backtrace-limit=0"
)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-unused-local-typedefs"
)

add_library(GENDIL INTERFACE)
add_library(GENDIL::GENDIL ALIAS GENDIL)
target_include_directories(
  GENDIL
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(GENDIL INTERFACE cxx_std_20)

gendil_validate_device_platform()

gendil_configure_openmp(GENDIL GENDIL_OPENMP_CXX_FLAGS)
gendil_configure_raja(GENDIL)
gendil_configure_mfem(GENDIL GENDIL_MFEM_ENABLE_CUDA)
if(GENDIL_MFEM_ENABLE_CUDA)
  set(USE_CUDA ON)
endif()

# MFEM can select CUDA after the initial user-option validation.
gendil_validate_device_platform()

gendil_configure_caliper(GENDIL)
gendil_configure_hypre(GENDIL GENDIL_USE_HYPRE_DEVICE)

set(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION OFF)
if(USE_CUDA)
  if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
  endif()
  enable_language(CUDA)
  gendil_configure_cuda(
    GENDIL
    "${GENDIL_OPENMP_CXX_FLAGS}"
    GENDIL_HAS_DEVICE_SPARSE_FINALIZATION
  )
elseif(USE_HIP)
  enable_language(HIP)
  gendil_configure_hip(
    GENDIL
    GENDIL_HAS_DEVICE_SPARSE_FINALIZATION
  )
endif()

gendil_finalize_device_sparse_finalization(
  GENDIL
  "${GENDIL_HAS_DEVICE_SPARSE_FINALIZATION}"
)

get_target_property(GENDIL_INCLUDES GENDIL INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "GENDIL include directories: ${GENDIL_INCLUDES}")
get_target_property(GENDIL_DEFINITIONS GENDIL INTERFACE_COMPILE_DEFINITIONS)
message(STATUS "GENDIL definitions: ${GENDIL_DEFINITIONS}")
get_target_property(GENDIL_LIBRARIES GENDIL INTERFACE_LINK_LIBRARIES)
message(STATUS "GENDIL libraries: ${GENDIL_LIBRARIES}")

install(DIRECTORY include/ DESTINATION include)
install(
  TARGETS GENDIL
  EXPORT GENDIL_Targets
  INCLUDES DESTINATION include
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "GENDILConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/GENDILConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/GENDILConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake
)
install(
  EXPORT GENDIL_Targets
  FILE GENDILTargets.cmake
  NAMESPACE GENDIL::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake
)
install(
  FILES
    "${PROJECT_BINARY_DIR}/GENDILConfig.cmake"
    "${PROJECT_BINARY_DIR}/GENDILConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(GENDIL_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
add_subdirectory(examples)
