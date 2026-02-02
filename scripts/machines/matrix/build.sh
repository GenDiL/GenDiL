# build gendil on matrix

mkdir -p build_matrix
cd build_matrix

module purge && module load cuda/12.2.2 gcc/12.1.1 cmake/3.30.5

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_matrix \
      -D USE_CUDA=ON \
      -D CMAKE_CUDA_ARCHITECTURES=90 \
      ..

make -j 16 && make test && make install

cd ..