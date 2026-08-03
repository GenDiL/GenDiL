// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapplycommon.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

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
void ApplyHostCOOToRawPointers(
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

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();
   const ComputeType compute_alpha =
      static_cast< ComputeType >( alpha );
   const ComputeType compute_beta =
      static_cast< ComputeType >( beta );

   ScaleSparseHostOutput(
      y_data,
      matrix.num_rows,
      compute_beta );

   #pragma omp parallel for
   for ( IndexType entry = 0; entry < matrix.nnz; ++entry )
   {
      const IndexType row = matrix.rows[entry];
      const IndexType col = matrix.cols[entry];
      const ComputeType contribution =
         compute_alpha *
         static_cast< ComputeType >( matrix.values[entry] ) *
         static_cast< ComputeType >( x_data[col] );
      AtomicAddInPlace(
         y_data[row],
         static_cast< OutputValueType >( contribution ) );
   }
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void Apply(
   const HostCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   details::CheckCOOApplyDimensions(
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

   details::ApplyHostCOOToRawPointers(
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
   HostAccessibleVector InputVector,
   HostAccessibleVector OutputVector >
void ApplyAdd(
   const HostCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto matrix_data = GetHostReadView( matrix );
   details::CheckCOOApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      return;
   }

   details::ApplyHostCOOToRawPointers(
      backend,
      matrix_data,
      ReadHostVector( x ),
      ReadWriteHostVector( y ),
      Real( 1 ),
      Real( 1 ) );
}

} // namespace gendil
