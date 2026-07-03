// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixstorage.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

namespace details
{

template < typename Matrix >
void CheckCSRApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(CSR backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(CSR backend, ...) output vector has the wrong size." );
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCSRToRawPointers(
   const Backend &,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;
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

      y_data[row] = static_cast< OutputValueType >( sum );
   }
}

#if defined(GENDIL_USE_DEVICE)
template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
__global__
void CSRDeviceApplyKernel(
   const Backend backend,
   const Matrix matrix,
   const InputValue * x,
   OutputValue * y )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;
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

      y[row] = static_cast< OutputValueType >( sum );
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
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyDeviceCSRToRawPointers(
   const Backend & backend,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
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
      y_data );
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
   typename BackendAccumulatorType >
void Apply(
   const HostCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCSRApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadHostData();
   auto * y_data = y.WriteHostData();

   details::ApplyHostCSRToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType >
void Apply(
   const NativeDeviceCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSRApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadDeviceData();
   auto * y_data = y.WriteDeviceData();

   details::ApplyDeviceCSRToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSRMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use CSRMatrix::operator() for CPU execution." );
#endif
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType >
void Apply(
   const HostCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCSRApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.HostRead();
   auto * y_data = y.HostWrite();

   details::ApplyHostCSRToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType >
void Apply(
   const NativeDeviceCSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSRApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.Read();
   auto * y_data = y.Write();

   details::ApplyDeviceCSRToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSRMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSRBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void CSRMatrix< ValueType, IndexType, MatrixBackend >::operator()(
   const InputVector & x,
   OutputVector & y ) const
{
   Apply( backend, *this, x, y );
}

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         CSRMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No CSRMatrix Apply overload is available for this backend/vector "
      "combination. Include csrmatrixapply.hpp and use a supported CSR "
      "backend/vector type." );
}

} // namespace gendil
