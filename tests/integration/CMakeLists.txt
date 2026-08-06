
add_subdirectory(gendil)
add_subdirectory(bsr)

if(USE_HYPRE)
    add_subdirectory(hypre)
endif()

if(USE_MFEM)
    add_subdirectory(mfem)
endif()
