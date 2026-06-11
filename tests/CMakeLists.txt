cmake_minimum_required(VERSION 3.12)

macro(add_my_test target_name source_file)
  add_executable(${target_name})

  target_link_libraries(${target_name} PRIVATE GENDIL)

  target_sources(
    ${target_name}
    PRIVATE
    ${source_file}
  )

  if (USE_CUDA)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
  endif()
  if(USE_HIP)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
  endif()

  target_include_directories(
    ${target_name}
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  )

  add_test(NAME ${target_name} COMMAND ${target_name})
endmacro()

macro(add_non_nvcc_test target_name source_file)
  # Skip only when using NVCC as the CUDA compiler
  # Add test for: CPU/OpenMP builds, HIP builds, and CUDA builds with clang
  if(USE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_COMPILER_ID)
      message(FATAL_ERROR "USE_CUDA is ON but CMAKE_CUDA_COMPILER_ID is undefined. Ensure enable_language(CUDA) has been called before using add_non_nvcc_test.")
    endif()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
      # Skip this test - it doesn't compile with NVCC
      message(STATUS "Skipping test ${target_name}: excluded for NVCC compiler")
    else()
      # CUDA with non-NVCC compiler (e.g., clang): add the test
      add_my_test(${target_name} ${source_file})
    endif()
  else()
    # CPU/OpenMP or HIP build: add the test
    add_my_test(${target_name} ${source_file})
  endif()
endmacro()

add_subdirectory(unit)
add_subdirectory(integration)
