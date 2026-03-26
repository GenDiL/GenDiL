mkdir -p build
cd build

export OpenMP_ROOT=$(brew --prefix)/opt/libomp

export LDFLAGS="$LDFLAGS -L${OpenMP_ROOT}/lib"

cmake -D CMAKE_BUILD_TYPE=Release \
      -D USE_OPENMP=ON \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build \
      -D CMAKE_INSTALL_PREFIX=../install \
      ..

make -j 8 && make test && make install

cd ..
