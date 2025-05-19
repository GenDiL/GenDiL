mkdir -p build_parallel_dane
cd build_parallel_dane

module load gcc/12.1.1-magic

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_parallel_dane \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build_parallel_dane \
      -D USE_MPI=ON \
      ..

make -j 8 && make test ARGS="--rerun-failed --output-on-failure" && make install

cd ..