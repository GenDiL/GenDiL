// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapplycommon.hpp"

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
void ApplyHostCSRToRawPointers(
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
   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      AccumulatorType sum = AccumulatorType( 0 );

      for ( IndexType entry = matrix.row_ptr[row];
            entry < matrix.row_ptr[row + 1];
            ++entry )
      {
         const ComputeType contribution =
            static_cast< ComputeType >( matrix.values[entry] ) *
            static_cast< ComputeType >( x_data[matrix.col_ind[entry]] );
         sum += static_cast< AccumulatorType >( contribution );
      }

      ComputeType result =
         compute_alpha * static_cast< ComputeType >( sum );
      if ( compute_beta != ComputeType( 0 ) )
      {
         result +=
            compute_beta * static_cast< ComputeType >( y_data[row] );
      }
      y_data[row] = static_cast< OutputValueType >( result );
   }
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void Apply(
   const HostCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   details::CheckCSRApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      ScaleSparseHostOutput(
         WriteHostVector( y ),
         matrix_data.num_rows,
         Real( 0 ) );
      return;
   }

   const auto * x_data = ReadHostVector( x );
   auto * y_data = WriteHostVector( y );

   details::ApplyHostCSRToRawPointers(
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
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType,
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void ApplyAdd(
   const HostCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   details::CheckCSRApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      return;
   }

   details::ApplyHostCSRToRawPointers(
      backend,
      matrix_data,
      ReadHostVector( x ),
      ReadWriteHostVector( y ),
      Real( 1 ),
      Real( 1 ) );
}

} // namespace gendil
