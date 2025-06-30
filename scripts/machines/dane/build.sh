mkdir -p build_dane
cd build_dane

module load gcc/12.1.1-magic

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_dane \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build_dane \
      -D CMAKE_CXX_COMPILER=g++ \
      ..

# make -j 8 && make test-periodic-mesh
make -j 8 && make test && make install

cd ..