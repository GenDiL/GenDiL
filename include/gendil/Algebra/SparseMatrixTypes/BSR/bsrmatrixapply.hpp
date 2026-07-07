// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixstorage.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

namespace details
{

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostBSRToRawPointers(
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
   for (IndexType block_row = 0;
        block_row < matrix.num_row_blocks;
        ++block_row)
   {
      for (IndexType local_row = 0;
           local_row < matrix.block_rows;
           ++local_row)
      {
         AccumulatorType sum = AccumulatorType(0);

         for (IndexType block_it = matrix.row_offsets[block_row];
              block_it < matrix.row_offsets[block_row + 1];
              ++block_it)
         {
            const IndexType block_col = matrix.col_indices[block_it];

            for (IndexType local_col = 0;
                 local_col < matrix.block_cols;
                 ++local_col)
            {
               const IndexType global_col =
                  block_col * matrix.block_cols + local_col;

               const ComputeType contribution =
                  static_cast< ComputeType >(
                     matrix.GetBlockEntry(block_it, local_row, local_col) ) *
                  static_cast< ComputeType >( x_data[global_col] );
               sum += static_cast< AccumulatorType >( contribution );
            }
         }

         const IndexType global_row =
            block_row * matrix.block_rows + local_row;
         y_data[global_row] = static_cast< OutputValueType >( sum );
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
   typename BackendAccumulatorType >
void Apply(
   const HostBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y )
{
   const auto * x_data = x.ReadHostData();
   auto * y_data = y.WriteHostData();

   details::ApplyHostBSRToRawPointers( backend, matrix, x_data, y_data );
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType >
void Apply(
   const HostBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   const auto * x_data = x.HostRead();
   auto * y_data = y.HostWrite();

   details::ApplyHostBSRToRawPointers( backend, matrix, x_data, y_data );
}
#endif

namespace details
{

template < typename Matrix >
GlobalIndex CheckBSRApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      matrix.block_rows > 0,
      "Apply(BSR backend, ...) requires a positive row block size." );
   GENDIL_VERIFY(
      matrix.block_cols > 0,
      "Apply(BSR backend, ...) requires a positive column block size." );

   const GlobalIndex expected_x_size =
      static_cast< GlobalIndex >( matrix.num_col_blocks ) *
      static_cast< GlobalIndex >( matrix.block_cols );
   const GlobalIndex expected_y_size =
      static_cast< GlobalIndex >( matrix.num_row_blocks ) *
      static_cast< GlobalIndex >( matrix.block_rows );

   GENDIL_VERIFY(
      x_size == static_cast< size_t >( expected_x_size ),
      "Apply(BSR backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( expected_y_size ),
      "Apply(BSR backend, ...) output vector has the wrong size." );

   return expected_y_size;
}

#if defined(GENDIL_USE_DEVICE)
template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
__global__
void BSRDeviceApplyKernel(
   const Backend backend,
   const Matrix matrix,
   const InputValue * x,
   OutputValue * y,
   const GlobalIndex total_rows )
{
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

   const GlobalIndex stride =
      static_cast< GlobalIndex >( blockDim.x ) *
      static_cast< GlobalIndex >( gridDim.x );
   const GlobalIndex block_rows =
      static_cast< GlobalIndex >( matrix.block_rows );
   const GlobalIndex block_cols =
      static_cast< GlobalIndex >( matrix.block_cols );

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
                     static_cast< decltype( matrix.block_rows ) >( local_row ),
                     static_cast< decltype( matrix.block_cols ) >( local_col ) ) ) *
               static_cast< ComputeType >( x[ global_col ] );
            sum += static_cast< AccumulatorType >( contribution );
         }
      }

      y[ global_row ] = static_cast< OutputValueType >( sum );
   }
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyDeviceBSRToRawPointers(
   const Backend & backend,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const GlobalIndex total_rows )
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
      total_rows );
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
   typename BackendAccumulatorType >
void Apply(
   const NativeDeviceBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix,
         x.Size(),
         y.Size() );

   const auto * x_data = x.ReadDeviceData();
   auto * y_data = y.WriteDeviceData();

   details::ApplyDeviceBSRToRawPointers(
      backend,
      matrix,
      x_data,
      y_data,
      expected_y_size );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "Apply(NativeDeviceBSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use BSRMatrix::operator() for CPU execution." );
#endif
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   typename BackendAccumulatorType >
void Apply(
   const NativeDeviceBSRBackend< BackendComputeType, BackendAccumulatorType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix,
         static_cast< size_t >( x.Size() ),
         static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.Read();
   auto * y_data = y.Write();

   details::ApplyDeviceBSRToRawPointers(
      backend,
      matrix,
      x_data,
      y_data,
      expected_y_size );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "Apply(NativeDeviceBSRBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void BSRMatrix< ValueType, IndexType, Layout, MatrixBackend >::operator()(
   const InputVector & x,
   OutputVector & y ) const
{
   Apply( backend, *this, x, y );
}

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const BSRMatrix< ValueType, IndexType, Layout, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         BSRMatrix< ValueType, IndexType, Layout, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No BSRMatrix Apply overload is available for this backend/vector "
      "combination. Include bsrmatrixapply.hpp and use a supported BSR "
      "backend/vector type." );
}

} // namespace gendil
