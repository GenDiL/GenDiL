cmake_minimum_required(VERSION 3.12)

macro(add_my_example target_name source_file)
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
endmacro()

add_subdirectory(mass)
add_subdirectory(advection)
add_subdirectory(poisson)
