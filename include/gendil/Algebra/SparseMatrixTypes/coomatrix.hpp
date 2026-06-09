// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil {

struct HostCOOBackend
{ };

using DefaultCOOBackend = HostCOOBackend;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCOOBackend >
struct COOMatrix;

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

/**
 * Canonical coordinate-list sparse matrix.
 *
 * Entries are sorted lexicographically by `(row, col)`, and each coordinate
 * appears at most once. Duplicate raw assembly triplets are finalized by
 * additive reduction before they are stored here. Exact zero values produced by
 * reduction are retained.
 */
template <
   typename ValueType,
   typename IndexType,
   typename Backend >
struct COOMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   HostDevicePointer< IndexType > rows;
   HostDevicePointer< IndexType > cols;
   HostDevicePointer< ValueType > values;

   // HostCOOBackend reads host-current matrix/vector data and marks the
   // output vector host-current through Vector::WriteHostData().
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
   typename Backend = DefaultCOOBackend >
auto MakeCOOMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz,
   Backend backend = Backend{} )
{
   COOMatrix< ValueType, IndexType, Backend > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;
   matrix.backend = backend;

   AllocateHostPointer( nnz, matrix.rows );
   AllocateDevicePointer( nnz, matrix.rows );
   AllocateHostPointer( nnz, matrix.cols );
   AllocateDevicePointer( nnz, matrix.cols );
   AllocateHostPointer( nnz, matrix.values );
   AllocateDevicePointer( nnz, matrix.values );

   return matrix;
}

template < typename Matrix >
void FreeCOOMatrix( Matrix & matrix )
{
   FreeHostPointer( matrix.rows );
   FreeDevicePointer( matrix.rows );
   FreeHostPointer( matrix.cols );
   FreeDevicePointer( matrix.cols );
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
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< InputVector >,
      "Apply(HostCOOBackend, COOMatrix, x, y) supports only explicit "
      "host vector overloads. Add an overload that acquires the correct "
      "read/write memory space before entering the COO loop." );
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
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< Backend >,
      "Apply(backend, COOMatrix, x, y) currently supports only "
      "HostCOOBackend. Device, OpenMP, row-segmented, and other COO "
      "backends are not implemented yet." );
}

namespace details
{

template < typename Matrix >
void CheckCOOApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(COO backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(COO backend, ...) output vector has the wrong size." );
}

template <
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCOOToRawPointers(
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using IndexType = typename Matrix::index_type;

   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      y_data[row] = OutputValue( 0 );
   }

   for ( IndexType entry = 0; entry < matrix.nnz; ++entry )
   {
      const IndexType row = matrix.rows[entry];
      const IndexType col = matrix.cols[entry];
      y_data[row] +=
         static_cast< OutputValue >(
            matrix.values[entry] * x_data[col] );
   }
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCOOApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   details::ApplyHostCOOToRawPointers( matrix, x_data, y_data );
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
void Apply(
   const HostCOOBackend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCOOApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.HostRead();
   Real * y_data = y.HostWrite();

   details::ApplyHostCOOToRawPointers( matrix, x_data, y_data );
}
#endif

} // namespace gendil
