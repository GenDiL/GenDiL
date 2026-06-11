// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/cscmatrixstorage.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

namespace details
{

template < typename Matrix >
void CheckCSCApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(CSC backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(CSC backend, ...) output vector has the wrong size." );
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCSCToRawPointers(
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

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();

   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      y_data[row] = OutputValue( 0 );
   }

   for ( IndexType col = 0; col < matrix.num_cols; ++col )
   {
      const ComputeType x_col =
         static_cast< ComputeType >( x_data[col] );

      for ( IndexType entry = matrix.col_ptr[col];
            entry < matrix.col_ptr[col + 1];
            ++entry )
      {
         const IndexType row = matrix.row_ind[entry];
         const ComputeType contribution =
            static_cast< ComputeType >( matrix.values[entry] ) * x_col;
         y_data[row] += static_cast< OutputValueType >( contribution );
      }
   }
}

#if defined(GENDIL_USE_DEVICE)
template < typename ValueType, typename IndexType >
__global__
void CSCDeviceZeroKernel(
   ValueType * y,
   const IndexType num_rows )
{
   const IndexType stride =
      static_cast< IndexType >( blockDim.x ) *
      static_cast< IndexType >( gridDim.x );

   for ( IndexType row =
            static_cast< IndexType >( blockIdx.x ) *
            static_cast< IndexType >( blockDim.x ) +
            static_cast< IndexType >( threadIdx.x );
         row < num_rows;
         row += stride )
   {
      y[row] = ValueType( 0 );
   }
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
__global__
void CSCDeviceApplyKernel(
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

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();

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
            static_cast< ComputeType >( matrix.values[entry] ) * x_col;
         AtomicAdd(
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
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyDeviceCSCToRawPointers(
   const Backend & backend,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   constexpr unsigned int threads_per_block = 256;
   const dim3 block_dim( threads_per_block );

   if ( matrix.num_rows > 0 )
   {
      const dim3 grid_dim =
         MakeCSCApplyGrid(
            static_cast< GlobalIndex >( matrix.num_rows ),
            "Apply(NativeDeviceCSCBackend, ...) zero launch grid is too large." );
      CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "Apply(NativeDeviceCSCBackend, ...): before zero launch" );
      CSCDeviceZeroKernel<<< grid_dim, block_dim >>>(
         y_data,
         matrix.num_rows );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "Apply(NativeDeviceCSCBackend, ...) zero" );
   }

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
         y_data );
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
   typename BackendComputeType >
void Apply(
   const HostCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCSCApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadHostData();
   auto * y_data = y.WriteHostData();

   details::ApplyHostCSCToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSCApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadDeviceData();
   auto * y_data = y.WriteDeviceData();

   details::ApplyDeviceCSCToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSCMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSCBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use CSCMatrix::operator() for CPU execution." );
#endif
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const HostCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCSCApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.HostRead();
   auto * y_data = y.HostWrite();

   details::ApplyHostCSCToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSCApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.Read();
   auto * y_data = y.Write();

   details::ApplyDeviceCSCToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSCMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSCBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void CSCMatrix< ValueType, IndexType, MatrixBackend >::operator()(
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
   const CSCMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         CSCMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No CSCMatrix Apply overload is available for this backend/vector "
      "combination. Include cscmatrixapply.hpp and use a supported CSC "
      "backend/vector type." );
}

} // namespace gendil
