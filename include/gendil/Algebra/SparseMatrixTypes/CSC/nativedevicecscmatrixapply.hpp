// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixapplycommon.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <limits>

namespace gendil
{

namespace details
{

#if defined(GENDIL_USE_DEVICE)
template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType >
__global__
void CSCDeviceApplyKernel(
   const Backend backend,
   const MatrixView matrix,
   const InputValue * x,
   OutputValue * y,
   const AlphaType alpha )
{
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using InputValueType = std::remove_cv_t< InputValue >;
   using OutputValueType = std::remove_cv_t< OutputValue >;
   using ComputeType =
      ResolveSparseComputeType_t< Backend, ValueType, InputValueType >;

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();
   const ComputeType compute_alpha =
      static_cast< ComputeType >( alpha );

   const IndexType stride =
      static_cast< IndexType >( blockDim.x ) *
      static_cast< IndexType >( gridDim.x );

   for ( IndexType col =
            static_cast< IndexType >( blockIdx.x ) *
            static_cast< IndexType >( blockDim.x ) +
            static_cast< IndexType >( threadIdx.x );
         col < matrix.num_cols;
         col += stride )
   {
      const ComputeType x_col =
         static_cast< ComputeType >( x[col] );

      for ( IndexType entry = matrix.col_ptr[col];
            entry < matrix.col_ptr[col + 1];
            ++entry )
      {
         const ComputeType contribution =
            compute_alpha *
            static_cast< ComputeType >( matrix.values[entry] ) * x_col;
         AtomicAddInPlace(
            y[matrix.row_ind[entry]],
            static_cast< OutputValueType >( contribution ) );
      }
   }
}

inline dim3 MakeCSCApplyGrid(
   const GlobalIndex work_items,
   const char * error_message )
{
   constexpr unsigned int threads_per_block = 256;
   const GlobalIndex grid_x_size =
      ( work_items + threads_per_block - 1 ) / threads_per_block;

   GENDIL_VERIFY(
      grid_x_size <=
         static_cast< GlobalIndex >( std::numeric_limits< unsigned int >::max() ),
      error_message );

   return dim3( static_cast< unsigned int >( grid_x_size ) );
}

template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType,
   typename BetaType >
void ApplyDeviceCSCToRawPointers(
   const Backend & backend,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const AlphaType alpha,
   const BetaType beta )
{
   constexpr unsigned int threads_per_block = 256;
   const dim3 block_dim( threads_per_block );

   ScaleSparseDeviceOutput(
      y_data,
      matrix.num_rows,
      beta );

   if ( matrix.num_cols > 0 )
   {
      const dim3 grid_dim =
         MakeCSCApplyGrid(
            static_cast< GlobalIndex >( matrix.num_cols ),
            "Apply(NativeDeviceCSCBackend, ...) apply launch grid is too large." );
      CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "Apply(NativeDeviceCSCBackend, ...): before apply launch" );
      CSCDeviceApplyKernel<<< grid_dim, block_dim >>>(
         backend,
         matrix,
         x_data,
         y_data,
         alpha );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "Apply(NativeDeviceCSCBackend, ...) apply" );
   }
}
#endif

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const NativeDeviceCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCSCApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );

   details::ApplyDeviceCSCToRawPointers(
      backend,
      matrix_data,
      x_data,
      y_data,
      Real( 1 ),
      Real( 0 ) );
#else
   static_assert(
      dependent_false_v< CSCMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSCBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use CSCMatrix::operator() for CPU execution." );
#endif
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const NativeDeviceCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCSCApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      return;
   }

   details::ApplyDeviceCSCToRawPointers(
      backend,
      matrix_data,
      ReadDeviceVector( x ),
      ReadWriteDeviceVector( y ),
      Real( 1 ),
      Real( 1 ) );
#else
   static_assert(
      dependent_false_v< CSCMatrix< ValueType, IndexType, MatrixBackend > >,
      "ApplyAdd(NativeDeviceCSCBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP)." );
#endif
}

} // namespace gendil
