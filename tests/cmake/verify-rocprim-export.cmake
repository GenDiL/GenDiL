if(NOT DEFINED PACKAGE_CONFIG)
  message(FATAL_ERROR "PACKAGE_CONFIG is required")
endif()

file(READ "${PACKAGE_CONFIG}" package_config_contents)
string(REGEX MATCH
  "if \\(ON\\)[ \t\r\n]+find_dependency\\(rocprim CONFIG REQUIRED\\)"
  rocprim_dependency
  "${package_config_contents}")
if(NOT rocprim_dependency)
  message(FATAL_ERROR
    "The installed-package configuration does not require enabled rocPRIM")
endif()
