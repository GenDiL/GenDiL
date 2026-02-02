# build gendil on lassen

mkdir -p build_lassen
cd build_lassen

module purge && module load cuda/12.2.2 gcc/12.2.1 cmake/3.29.2

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_lassen \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build_lassen \
      -D CMAKE_CUDA_ARCHITECTURES=70 \
      ..

make -j 16 && make test && make install

cd ..