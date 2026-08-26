// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapplycommon.hpp"

namespace gendil
{

namespace details
{

template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType,
   typename BetaType >
void ApplyHostBSRToRawPointers(
   const Backend &,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const AlphaType alpha,
   const BetaType beta )
{
   using IndexType = typename MatrixView::index_type;
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

   #pragma omp parallel for
   for ( IndexType block_row = 0;
         block_row < matrix.num_row_blocks;
         ++block_row )
   {
      for ( IndexType local_row = 0;
            local_row < matrix.block_rows;
            ++local_row )
      {
         AccumulatorType sum = AccumulatorType( 0 );

         for ( IndexType block_it = matrix.row_offsets[block_row];
               block_it < matrix.row_offsets[block_row + 1];
               ++block_it )
         {
            const IndexType block_col = matrix.col_indices[block_it];

            for ( IndexType local_col = 0;
                  local_col < matrix.block_cols;
                  ++local_col )
            {
               const IndexType global_col =
                  block_col * matrix.block_cols + local_col;

               const ComputeType contribution =
                  static_cast< ComputeType >(
                     matrix.GetBlockEntry(
                        block_it,
                        local_row,
                        local_col ) ) *
                  static_cast< ComputeType >( x_data[global_col] );
               sum += static_cast< AccumulatorType >( contribution );
            }
         }

         const IndexType global_row =
            block_row * matrix.block_rows + local_row;
         ComputeType result =
            compute_alpha * static_cast< ComputeType >( sum );
         if ( compute_beta != ComputeType( 0 ) )
         {
            result +=
               compute_beta *
               static_cast< ComputeType >( y_data[global_row] );
         }
         y_data[global_row] =
            static_cast< OutputValueType >( result );
      }
   }
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void Apply(
   const HostBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix_data,
         GetVectorSize( x ),
         GetVectorSize( y ) );
   if ( matrix_data.num_blocks == 0 )
   {
      ScaleSparseHostOutput(
         WriteHostVector( y ),
         expected_y_size,
         Real( 0 ) );
      return;
   }
   const auto * x_data = ReadHostVector( x );
   auto * y_data = WriteHostVector( y );

   details::ApplyHostBSRToRawPointers(
      backend,
      matrix_data,
      x_data,
      y_data,
      Real( 1 ),
      Real( 0 ) );
}

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void ApplyAdd(
   const HostBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   details::CheckBSRApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.num_blocks == 0 )
   {
      return;
   }

   details::ApplyHostBSRToRawPointers(
      backend,
      matrix_data,
      ReadHostVector( x ),
      ReadWriteHostVector( y ),
      Real( 1 ),
      Real( 1 ) );
}

} // namespace gendil
