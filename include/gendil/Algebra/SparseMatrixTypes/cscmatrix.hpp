// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

struct HostCSCBackend
{ };
struct NativeDeviceCSCBackend
{ };

using DefaultCSCBackend = HostCSCBackend;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSCBackend >
struct CSCMatrix;

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );
#endif

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

/**
 * Canonical compressed sparse column matrix.
 *
 * Row indices are sorted within each column, each coordinate appears at most
 * once, and exact zeros produced by duplicate reduction are retained.
 */
template <
   typename ValueType,
   typename IndexType,
   typename Backend >
struct CSCMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   HostDevicePointer< IndexType > col_ptr; // size = num_cols + 1
   HostDevicePointer< IndexType > row_ind; // size = nnz
   HostDevicePointer< ValueType > values;  // size = nnz

   Backend backend{};

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const
   {
      Apply( backend, *this, x, y );
   }
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSCBackend >
auto MakeCSCMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz,
   Backend backend = Backend{} )
{
   CSCMatrix< ValueType, IndexType, Backend > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;
   matrix.backend = backend;

   const IndexType col_ptr_size = num_cols + IndexType( 1 );
   AllocateHostPointer( col_ptr_size, matrix.col_ptr );
   AllocateDevicePointer( col_ptr_size, matrix.col_ptr );
   AllocateHostPointer( nnz, matrix.row_ind );
   AllocateDevicePointer( nnz, matrix.row_ind );
   AllocateHostPointer( nnz, matrix.values );
   AllocateDevicePointer( nnz, matrix.values );

   return matrix;
}

template < typename Matrix >
void FreeCSCMatrix( Matrix & matrix )
{
   FreeHostPointer( matrix.col_ptr );
   FreeDevicePointer( matrix.col_ptr );
   FreeHostPointer( matrix.row_ind );
   FreeDevicePointer( matrix.row_ind );
   FreeHostPointer( matrix.values );
   FreeDevicePointer( matrix.values );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< InputVector >,
      "Apply(HostCSCBackend, CSCMatrix, x, y) supports only explicit "
      "host vector overloads. Add an overload that acquires the correct "
      "read/write memory space before entering the CSC loop." );
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
      dependent_false_v< Backend >,
      "Apply(backend, CSCMatrix, x, y) currently supports only "
      "HostCSCBackend and NativeDeviceCSCBackend. OpenMP, transpose, and "
      "other CSC backends are not implemented yet." );
}

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
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCSCToRawPointers(
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;

   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      y_data[row] = OutputValue( 0 );
   }

   for ( IndexType col = 0; col < matrix.num_cols; ++col )
   {
      const ValueType x_col =
         static_cast< ValueType >( x_data[col] );

      for ( IndexType entry = matrix.col_ptr[col];
            entry < matrix.col_ptr[col + 1];
            ++entry )
      {
         const IndexType row = matrix.row_ind[entry];
         y_data[row] +=
            static_cast< OutputValue >(
               matrix.values[entry] * x_col );
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

template < typename Matrix >
__global__
void CSCDeviceApplyKernel(
   const Matrix matrix,
   const Real * x,
   Real * y )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;

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
      const ValueType x_col =
         static_cast< ValueType >( x[col] );

      for ( IndexType entry = matrix.col_ptr[col];
            entry < matrix.col_ptr[col + 1];
            ++entry )
      {
         AtomicAdd(
            y[matrix.row_ind[entry]],
            static_cast< Real >( matrix.values[entry] * x_col ) );
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

template < typename Matrix >
void ApplyDeviceCSCToRawPointers(
   const Matrix & matrix,
   const Real * x_data,
   Real * y_data )
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
   typename MatrixBackend >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCSCApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   details::ApplyHostCSCToRawPointers( matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSCApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const Real * x_data = x.ReadDeviceData();
   Real * y_data = y.WriteDeviceData();

   details::ApplyDeviceCSCToRawPointers( matrix, x_data, y_data );
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
   typename MatrixBackend >
void Apply(
   const HostCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCSCApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.HostRead();
   Real * y_data = y.HostWrite();

   details::ApplyHostCSCToRawPointers( matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSCBackend &,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSCApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.Read();
   Real * y_data = y.Write();

   details::ApplyDeviceCSCToRawPointers( matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSCMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSCBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

} // namespace gendil
