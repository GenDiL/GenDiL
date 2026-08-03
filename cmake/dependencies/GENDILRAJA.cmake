include_guard(GLOBAL)

function(gendil_configure_raja target)
  if(USE_RAJA)
    find_package(
      RAJA REQUIRED CONFIG
      HINTS
        "${RAJA_DIR}"
        "${RAJA_DIR}/share/raja/cmake"
        "${RAJA_DIR}/lib/cmake/raja"
      NO_DEFAULT_PATH
    )
    message(STATUS "RAJA found!")
    target_link_libraries(${target} INTERFACE RAJA)
    target_compile_definitions(${target} INTERFACE GENDIL_USE_RAJA)
    get_target_property(
      RAJA_INCLUDE_DIRS RAJA INTERFACE_INCLUDE_DIRECTORIES
    )
    get_target_property(RAJA_LIBRARIES RAJA INTERFACE_LINK_LIBRARIES)
    message(STATUS "RAJA include directories: ${RAJA_INCLUDE_DIRS}")
    message(STATUS "RAJA libraries: ${RAJA_LIBRARIES}")
  else()
    message(STATUS "RAJA support is not enabled.")
  endif()
endfunction()
