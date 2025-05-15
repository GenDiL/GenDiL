mkdir -p build_parallel_ruby
cd build_parallel_ruby

module load gcc/12.1.1-magic

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_parallel_ruby \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build_parallel_ruby \
      -D CMAKE_CXX_COMPILER=g++ \
      ..

# make -j 8 && make test-periodic-mesh
make -j 8 && make test && make install

cd ..