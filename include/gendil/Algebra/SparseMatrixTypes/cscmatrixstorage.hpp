// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

namespace gendil {

template < typename ComputeType = void >
struct HostCSCBackend
{
   using compute_type = ComputeType;
};

template < typename ComputeType = void >
struct NativeDeviceCSCBackend
{
   using compute_type = ComputeType;
};

// Standalone CSC storage default. GenericAssembly dispatch uses
// DefaultBackendFor_t<MatrixAssemblyType::CSC> instead.
using DefaultCSCBackend = HostCSCBackend<>;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSCBackend >
struct CSCMatrix;

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
   void operator()( const InputVector & x, OutputVector & y ) const;
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

} // namespace gendil
