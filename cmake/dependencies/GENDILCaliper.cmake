include_guard(GLOBAL)

function(gendil_configure_caliper target)
  if(USE_CALIPER)
    find_package(
      caliper REQUIRED
      HINTS "${caliper_DIR}" "${caliper_DIR}/share/cmake/caliper"
      NO_DEFAULT_PATH
    )
    message(STATUS "Caliper found!")
    target_link_libraries(${target} INTERFACE caliper)
    target_compile_definitions(${target} INTERFACE GENDIL_USE_CALIPER)
    get_target_property(
      CALIPER_INCLUDE_DIRS caliper INTERFACE_INCLUDE_DIRECTORIES
    )
    get_target_property(
      CALIPER_LIBRARIES caliper INTERFACE_LINK_LIBRARIES
    )
    message(STATUS "Caliper include directories: ${CALIPER_INCLUDE_DIRS}")
    message(STATUS "Caliper libraries: ${CALIPER_LIBRARIES}")
  else()
    message(STATUS "Caliper support is not enabled.")
  endif()
endfunction()
