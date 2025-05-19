
add_my_test(test-dot test-dot.cpp)
add_my_test(test-getpoint test-getpoint.cpp)
add_my_test(test-garbage-collector test-garbage-collector.cpp)

# OpenMP tests
if(USE_OPENMP)
  add_my_test(test-openmp test-openmp.cpp)
endif()

# MPI tests
if(USE_MPI)
  add_my_mpi_test(test-mpi test-mpi.cpp 2)
endif()
