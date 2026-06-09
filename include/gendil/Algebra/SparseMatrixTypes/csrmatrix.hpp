// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

struct HostCSRBackend
{ };
struct NativeDeviceCSRBackend
{ };

using DefaultCSRBackend = HostCSRBackend;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSRBackend >
struct CSRMatrix;

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

/**
 * Canonical compressed sparse row matrix.
 *
 * Column indices are sorted within each row, each coordinate appears at most
 * once, and exact zeros produced by duplicate reduction are retained.
 */
template <
   typename ValueType,
   typename IndexType,
   typename Backend >
struct CSRMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   HostDevicePointer< IndexType > row_ptr; // size = num_rows + 1
   HostDevicePointer< IndexType > col_ind; // size = nnz
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
   typename Backend = DefaultCSRBackend >
auto MakeCSRMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz,
   Backend backend = Backend{} )
{
   CSRMatrix< ValueType, IndexType, Backend > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;
   matrix.backend = backend;

   const IndexType row_ptr_size = num_rows + IndexType( 1 );
   AllocateHostPointer( row_ptr_size, matrix.row_ptr );
   AllocateDevicePointer( row_ptr_size, matrix.row_ptr );
   AllocateHostPointer( nnz, matrix.col_ind );
   AllocateDevicePointer( nnz, matrix.col_ind );
   AllocateHostPointer( nnz, matrix.values );
   AllocateDevicePointer( nnz, matrix.values );

   return matrix;
}

template < typename Matrix >
void FreeCSRMatrix( Matrix & matrix )
{
   FreeHostPointer( matrix.row_ptr );
   FreeDevicePointer( matrix.row_ptr );
   FreeHostPointer( matrix.col_ind );
   FreeDevicePointer( matrix.col_ind );
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
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< InputVector >,
      "Apply(HostCSRBackend, CSRMatrix, x, y) supports only explicit "
      "host vector overloads. Add an overload that acquires the correct "
      "read/write memory space before entering the CSR loop." );
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
      dependent_false_v< Backend >,
      "Apply(backend, CSRMatrix, x, y) currently supports only "
      "HostCSRBackend and NativeDeviceCSRBackend. OpenMP is enabled through "
      "HostCSRBackend when GENDIL_USE_OPENMP is defined; CSC, transpose, and "
      "other CSR backends are not implemented yet." );
}

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
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCSRToRawPointers(
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;

   #pragma omp parallel for
   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      ValueType sum = ValueType( 0 );

      for ( IndexType entry = matrix.row_ptr[row];
            entry < matrix.row_ptr[row + 1];
            ++entry )
      {
         sum +=
            matrix.values[entry] *
            static_cast< ValueType >( x_data[matrix.col_ind[entry]] );
      }

      y_data[row] = static_cast< OutputValue >( sum );
   }
}

#if defined(GENDIL_USE_DEVICE)
template < typename Matrix >
__global__
void CSRDeviceApplyKernel(
   const Matrix matrix,
   const Real * x,
   Real * y )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;

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
      ValueType sum = ValueType( 0 );

      for ( IndexType entry = matrix.row_ptr[row];
            entry < matrix.row_ptr[row + 1];
            ++entry )
      {
         sum +=
            matrix.values[entry] *
            static_cast< ValueType >( x[matrix.col_ind[entry]] );
      }

      y[row] = static_cast< Real >( sum );
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

template < typename Matrix >
void ApplyDeviceCSRToRawPointers(
   const Matrix & matrix,
   const Real * x_data,
   Real * y_data )
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
   typename MatrixBackend >
void Apply(
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCSRApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   details::ApplyHostCSRToRawPointers( matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSRApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const Real * x_data = x.ReadDeviceData();
   Real * y_data = y.WriteDeviceData();

   details::ApplyDeviceCSRToRawPointers( matrix, x_data, y_data );
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
   typename MatrixBackend >
void Apply(
   const HostCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCSRApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.HostRead();
   Real * y_data = y.HostWrite();

   details::ApplyHostCSRToRawPointers( matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const NativeDeviceCSRBackend &,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCSRApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.Read();
   Real * y_data = y.Write();

   details::ApplyDeviceCSRToRawPointers( matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< CSRMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCSRBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

} // namespace gendil
