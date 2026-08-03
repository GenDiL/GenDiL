// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapplycommon.hpp"

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
   typename AlphaType,
   typename BetaType >
__global__
void CSRDeviceApplyKernel(
   const Backend backend,
   const MatrixView matrix,
   const InputValue * x,
   OutputValue * y,
   const AlphaType alpha,
   const BetaType beta )
{
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using InputValueType = std::remove_cv_t< InputValue >;
   using OutputValueType = std::remove_cv_t< OutputValue >;
   using ComputeType =
      ResolveSparseComputeType_t< Backend, ValueType, InputValueType >;
   using AccumulatorType =
      ResolveSparseAccumulatorType_t< Backend, ComputeType >;

   CheckRowOwnedSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType,
      AccumulatorType >();
   CheckSparseOutputScalingArithmetic<
      OutputValueType,
      ComputeType >();
   const ComputeType compute_alpha =
      static_cast< ComputeType >( alpha );
   const ComputeType compute_beta =
      static_cast< ComputeType >( beta );

   const IndexType stride =
      static_cast< IndexType >( blockDim.x ) *
      static_cast< IndexType >( gridDim.x );

   for ( IndexType row =
            static_cast< IndexType >( blockIdx.x ) *
            static_cast< IndexType >( blockDim.x ) +
            static_cast< IndexType >( threadIdx.x );
         row < matrix.num_rows;
         row += stride )
   {
      AccumulatorType sum = AccumulatorType( 0 );

      for ( IndexType entry = matrix.row_ptr[row];
            entry < matrix.row_ptr[row + 1];
            ++entry )
      {
         const ComputeType contribution =
         static_cast< ComputeType >( matrix.values[entry] ) *
            static_cast< ComputeType >( x[matrix.col_ind[entry]] );
         sum += static_cast< AccumulatorType >( contribution );
      }

      ComputeType result =
         compute_alpha * static_cast< ComputeType >( sum );
      if ( compute_beta != ComputeType( 0 ) )
      {
         result +=
            compute_beta * static_cast< ComputeType >( y[row] );
      }
      y[row] = static_cast< OutputValueType >( result );
   }
}

inline dim3 MakeCSRApplyGrid(
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
void ApplyDeviceCSRToRawPointers(
   const Backend & backend,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const AlphaType alpha,
   const BetaType beta )
{
   if ( matrix.num_rows == 0 )
   {
      return;
   }

   constexpr unsigned int threads_per_block = 256;
   const dim3 block_dim( threads_per_block );
   const dim3 grid_dim =
      MakeCSRApplyGrid(
         static_cast< GlobalIndex >( matrix.num_rows ),
         "Apply(NativeDeviceCSRBackend, ...) apply launch grid is too large." );

   CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
   GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
      "Apply(NativeDeviceCSRBackend, ...): before apply launch" );
   CSRDeviceApplyKernel<<< grid_dim, block_dim >>>(
      backend,
      matrix,
      x_data,
      y_data,
      alpha,
      beta );
   GENDIL_CHECK_LAST_DEVICE_LAUNCH(
      "Apply(NativeDeviceCSRBackend, ...) apply" );
}
#endif

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const NativeDeviceCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCSRApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );

   details::ApplyDeviceCSRToRawPointers(
      backend,
      matrix_data,
      x_data,
      y_data,
      Real( 1 ),
      Real( 0 ) );
#else
   static_assert(
      dependent_false_v< CSRMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use CSRMatrix::operator() for CPU execution." );
#endif
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const NativeDeviceCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCSRApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      return;
   }

   details::ApplyDeviceCSRToRawPointers(
      backend,
      matrix_data,
      ReadDeviceVector( x ),
      ReadWriteDeviceVector( y ),
      Real( 1 ),
      Real( 1 ) );
#else
   static_assert(
      dependent_false_v< CSRMatrix< ValueType, IndexType, MatrixBackend > >,
      "ApplyAdd(NativeDeviceCSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP)." );
#endif
}

} // namespace gendil
