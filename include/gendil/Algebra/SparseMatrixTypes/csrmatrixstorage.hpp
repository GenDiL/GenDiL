// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

namespace gendil {

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct HostCSRBackend : HostMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct NativeDeviceCSRBackend : DeviceMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

// Standalone CSR storage default. GenericAssembly dispatch uses
// DefaultBackendFor_t<MatrixAssemblyType::CSR> instead.
using DefaultCSRBackend = HostCSRBackend<>;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSRBackend >
struct CSRMatrix;

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
   void operator()( const InputVector & x, OutputVector & y ) const;
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

} // namespace gendil
