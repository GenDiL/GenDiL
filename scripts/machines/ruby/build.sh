mkdir -p build_ruby
cd build_ruby

module load gcc/12.1.1-magic

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_ruby \
      -D MFEM_DIR=../mfem/build_ruby \
      -D CMAKE_CXX_COMPILER=g++ \
      ..

make -j 8 && make test && make install

cd ..