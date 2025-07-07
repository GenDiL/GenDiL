# build gendil on matrix

mkdir -p build_matrix
cd build_matrix

module purge && module load cmake/3.30.5 gcc/10.3.1-magic cuda/11.8.0

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_matrix \
      -D USE_CUDA=ON \
      -D CMAKE_CUDA_ARCHITECTURES=90 \
      ..

make -j 16 && make test && make install

cd ..
