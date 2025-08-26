
add_my_test(test-dot test-dot.cpp)
add_my_test(test-getpoint test-getpoint.cpp)
add_my_test(test-garbage-collector test-garbage-collector.cpp)
add_my_test(test-gmres test-gmres.cpp)
add_my_test(test-solvers test-solvers.cpp)

if(USE_OPENMP)
  add_my_test(test-openmp test-openmp.cpp)
endif()
