# build gendil on tioga

mkdir -p build_tioga
cd build_tioga

export ROCM_VER="6.3.1"
module purge && module load rocmcc/${ROCM_VER}-magic cmake/3.23.1

export ROCM_PATH=/opt/rocm-${ROCM_VER}
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$ROCM_PATH/lib/cmake/hip:$ROCM_PATH/lib/cmake/hipblas:$ROCM_PATH/lib/cmake/hipblas-common:$ROCM_PATH/lib/cmake/hipsparse:$ROCM_PATH/lib/cmake/rocsparse:$ROCM_PATH/lib/cmake/rocrand
export CXX=hipcc
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=../install_tioga \
      -D USE_MFEM=ON \
      -D MFEM_DIR=../mfem/build_tioga \
      -D USE_HIP=ON \
      -D ROCM_ROOT_DIR="/opt/rocm-${ROCM_VER}" \
      -D HIP_ROOT_DIR="/opt/rocm-${ROCM_VER}/hip" \
      -D HIP_PATH=/opt/rocm-${ROCM_VER}/bin \
      -D CMAKE_CXX_COMPILER=hipcc \
      -D CMAKE_HIP_ARCHITECTURES="gfx90a" \
      ..

make -j 16 && make test && make install

cd ..