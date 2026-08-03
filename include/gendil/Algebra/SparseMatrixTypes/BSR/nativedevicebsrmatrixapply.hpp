// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapplycommon.hpp"

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
void BSRDeviceApplyKernel(
   const Backend backend,
   const MatrixView matrix,
   const InputValue * x,
   OutputValue * y,
   const AlphaType alpha,
   const BetaType beta )
{
   using ValueType = typename MatrixView::value_type;
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

   const GlobalIndex stride =
      static_cast< GlobalIndex >( blockDim.x ) *
      static_cast< GlobalIndex >( gridDim.x );
   const GlobalIndex block_rows =
      static_cast< GlobalIndex >( matrix.block_rows );
   const GlobalIndex block_cols =
      static_cast< GlobalIndex >( matrix.block_cols );
   const GlobalIndex total_rows =
      static_cast< GlobalIndex >( matrix.num_row_blocks ) * block_rows;

   for ( GlobalIndex global_row =
            static_cast< GlobalIndex >( blockIdx.x ) *
            static_cast< GlobalIndex >( blockDim.x ) +
            static_cast< GlobalIndex >( threadIdx.x );
         global_row < total_rows;
         global_row += stride )
   {
      const GlobalIndex block_row = global_row / block_rows;
      const GlobalIndex local_row = global_row % block_rows;

      AccumulatorType sum = AccumulatorType( 0 );

      for ( auto block_it = matrix.row_offsets[ block_row ];
            block_it < matrix.row_offsets[ block_row + 1 ];
            ++block_it )
      {
         const auto block_col = matrix.col_indices[ block_it ];

         for ( GlobalIndex local_col = 0;
               local_col < block_cols;
               ++local_col )
         {
            const GlobalIndex global_col =
               static_cast< GlobalIndex >( block_col ) * block_cols +
               local_col;

            const ComputeType contribution =
               static_cast< ComputeType >(
                  matrix.GetBlockEntry(
                     block_it,
                     local_row,
                     local_col ) ) *
               static_cast< ComputeType >( x[ global_col ] );
            sum += static_cast< AccumulatorType >( contribution );
         }
      }

      ComputeType result =
         compute_alpha * static_cast< ComputeType >( sum );
      if ( compute_beta != ComputeType( 0 ) )
      {
         result +=
            compute_beta *
            static_cast< ComputeType >( y[global_row] );
      }
      y[ global_row ] = static_cast< OutputValueType >( result );
   }
}

template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType,
   typename BetaType >
void ApplyDeviceBSRToRawPointers(
   const Backend & backend,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const GlobalIndex total_rows,
   const AlphaType alpha,
   const BetaType beta )
{
   if ( total_rows == 0 )
   {
      return;
   }

   constexpr unsigned int threads_per_block = 256;
   const GlobalIndex grid_x_size =
      ( total_rows + threads_per_block - 1 ) / threads_per_block;

   GENDIL_VERIFY(
      grid_x_size <=
         static_cast< GlobalIndex >( std::numeric_limits< unsigned int >::max() ),
      "Apply(NativeDeviceBSRBackend, ...) launch grid is too large." );

   const dim3 block_dim( threads_per_block );
   const dim3 grid_dim( static_cast< unsigned int >( grid_x_size ) );

   CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
   GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
      "Apply(NativeDeviceBSRBackend, ...): before launch" );
   BSRDeviceApplyKernel<<< grid_dim, block_dim >>>(
      backend,
      matrix,
      x_data,
      y_data,
      alpha,
      beta );
   GENDIL_CHECK_LAST_DEVICE_LAUNCH(
      "Apply(NativeDeviceBSRBackend, ...)" );
}
#endif

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const NativeDeviceBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix_data,
         GetVectorSize( x ),
         GetVectorSize( y ) );

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );

   details::ApplyDeviceBSRToRawPointers(
      backend,
      matrix_data,
      x_data,
      y_data,
      expected_y_size,
      Real( 1 ),
      Real( 0 ) );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "Apply(NativeDeviceBSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use BSRMatrix::operator() for CPU execution." );
#endif
}

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const NativeDeviceBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix_data,
         GetVectorSize( x ),
         GetVectorSize( y ) );
   if ( matrix_data.num_blocks == 0 )
   {
      return;
   }

   details::ApplyDeviceBSRToRawPointers(
      backend,
      matrix_data,
      ReadDeviceVector( x ),
      ReadWriteDeviceVector( y ),
      expected_y_size,
      Real( 1 ),
      Real( 1 ) );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "ApplyAdd(NativeDeviceBSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP)." );
#endif
}

} // namespace gendil
