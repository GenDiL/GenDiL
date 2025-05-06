cmake_minimum_required(VERSION 3.12)

project(
    "GENDIL"
    VERSION 0.0.1
    DESCRIPTION "A header-only c++ library."
    HOMEPAGE_URL "https://github.com/????")

include(GNUInstallDirs)

# Set optimization flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -DNDEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING "")

# Print warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -ftemplate-backtrace-limit=0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-unused-local-typedefs") # supress some warnings

# Options for third-party dependencies
option(USE_MFEM "Enables MFEM interface." ON)
option(USE_OPENMP "Enables OpenMP parallelization features." ON)
option(USE_CUDA "Enables CUDA support." OFF)
option(USE_HIP "Enables HIP support." OFF)
option(USE_RAJA "Enables RAJA features." OFF)
option(USE_CALIPER "Enables Caliper interface for benchmarking." OFF)

add_library(GENDIL INTERFACE)

# Add -pg to compile and link flags to enable gprof profiling
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

# OpenMP
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "OpenMP found!")
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_OPENMP)
  target_link_libraries(GENDIL INTERFACE OpenMP::OpenMP_CXX)
endif()

# RAJA
if(USE_RAJA)
  find_package(RAJA REQUIRED CONFIG HINTS "${RAJA_DIR}"
              "${RAJA_DIR}/share/raja/cmake"
              "${RAJA_DIR}/lib/cmake/raja" NO_DEFAULT_PATH)
  message(STATUS "RAJA found!")
  target_link_libraries(GENDIL INTERFACE RAJA)
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_RAJA)
  # Print RAJA include directories
  get_target_property(RAJA_INCLUDE_DIRS RAJA INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "RAJA include directories: ${RAJA_INCLUDE_DIRS}")
  get_target_property(RAJA_LIBRARIES RAJA INTERFACE_LINK_LIBRARIES)
  message(STATUS "RAJA libraries: ${RAJA_LIBRARIES}")
else()
  message(STATUS "RAJA support is not enabled.")
endif()

# MFEM
if(USE_MFEM)
  if (USE_HIP)
    find_package(HIP REQUIRED)
    find_package(HIPBLAS REQUIRED)
    find_package(HIPSPARSE REQUIRED)
  endif()
  find_package(MFEM REQUIRED NAMES MFEM HINTS "${MFEM_DIR}"
              "${MFEM_DIR}/lib/cmake/mfem" NO_DEFAULT_PATH)
  message(STATUS "MFEM found!")
  target_include_directories(GENDIL INTERFACE ${MFEM_INCLUDE_DIRS})
  target_link_libraries(GENDIL INTERFACE ${MFEM_LIBRARIES})
  message(STATUS "mfem include directories: ${MFEM_INCLUDE_DIRS}")
  message(STATUS "mfem libraries: ${MFEM_LIBRARIES}")
  message(STATUS "MFEM_DIR is set to: ${MFEM_LIBRARY_DIR}/..")
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_MFEM)
  add_definitions(-DMFEM_DIR="${MFEM_LIBRARY_DIR}/..")

  if (MFEM_USE_CUDA)
    set(USE_CUDA ON)
  endif()
else()
  message(STATUS "MFEM support is not enabled.")
endif()

# Caliper
if(USE_CALIPER)
  find_package(caliper REQUIRED HINTS "${caliper_DIR}"
              "${caliper_DIR}/share/cmake/caliper" NO_DEFAULT_PATH)
  message(STATUS "Caliper found!")
  target_link_libraries(GENDIL INTERFACE caliper caliper-tools-util)
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_CALIPER)
  # Print Caliper include directories and libraries
  get_target_property(CALIPER_INCLUDE_DIRS caliper INTERFACE_INCLUDE_DIRECTORIES)
  message(STATUS "Caliper include directories: ${CALIPER_INCLUDE_DIRS}")
  get_target_property(CALIPER_LIBRARIES caliper INTERFACE_LINK_LIBRARIES)
  message(STATUS "Caliper libraries: ${CALIPER_LIBRARIES}")
else()
  message(STATUS "Caliper support is not enabled.")
endif()

target_include_directories(
  GENDIL
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(GENDIL INTERFACE cxx_std_17)

# CUDA
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_features(GENDIL INTERFACE cuda_std_17)
  target_include_directories(GENDIL
    INTERFACE
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  if (NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()

  target_compile_options(GENDIL INTERFACE
    --extended-lambda
    --expt-relaxed-constexpr
    -keep
    -ftemplate-backtrace-limit=0
    -Xptxas=-v
    -ccbin=${CMAKE_CXX_COMPILER}
  )
  if(USE_OPENMP)
    target_compile_options(GENDIL INTERFACE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    )
  endif()

  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_CUDA)
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_DEVICE)
endif()

# HIP
if(USE_HIP)
  enable_language(HIP)
  find_package(HIP REQUIRED)
  # target_compile_features(GENDIL INTERFACE HIP)
  target_include_directories(GENDIL INTERFACE ${hip_INCLUDE_DIRS})
  target_compile_options(GENDIL INTERFACE $<$<COMPILE_LANGUAGE:HIP>:
    -ftemplate-backtrace-limit=0
  >)
  target_compile_definitions(GENDIL INTERFACE GENDIL_USE_HIP GENDIL_USE_DEVICE)
  message(STATUS "HIP support enabled!")
endif()

# Debug message to print include directories
get_target_property(GENDIL_INCLUDES GENDIL INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "GENDIL include directories: ${GENDIL_INCLUDES}")

# Debug message to print compile definitions
get_target_property(GENDIL_DEFINITIONS GENDIL INTERFACE_COMPILE_DEFINITIONS)
message(STATUS "GENDIL definitions: ${GENDIL_DEFINITIONS}")

# Debug message to print linked libraries
get_target_property(GENDIL_LIBRARIES GENDIL INTERFACE_LINK_LIBRARIES)
message(STATUS "GENDIL libraries: ${GENDIL_LIBRARIES}")

# Install header files
install(DIRECTORY include/ DESTINATION include)

install(TARGETS GENDIL
        EXPORT GENDIL_Targets
        INCLUDES DESTINATION include
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "GENDILConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/GENDILConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/GENDILConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake)

# Export the targets
install(EXPORT GENDIL_Targets
        FILE GENDILTargets.cmake
        NAMESPACE GENDIL::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake)

install(FILES "${PROJECT_BINARY_DIR}/GENDILConfig.cmake"
              "${PROJECT_BINARY_DIR}/GENDILConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/gendil/cmake)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/gendil DESTINATION include)

enable_testing()
add_subdirectory(tests)
add_subdirectory(benchmarks)
