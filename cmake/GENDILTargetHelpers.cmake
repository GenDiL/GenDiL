include_guard(GLOBAL)

function(gendil_set_device_source_language target source)
  if(USE_CUDA)
    set_source_files_properties(
      "${source}"
      TARGET_DIRECTORY ${target}
      PROPERTIES LANGUAGE CUDA
    )
  elseif(USE_HIP)
    set_source_files_properties(
      "${source}"
      TARGET_DIRECTORY ${target}
      PROPERTIES LANGUAGE HIP
    )
  endif()
endfunction()
