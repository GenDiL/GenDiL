// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

namespace gendil {

template < typename ComputeType = void >
struct HostCOOBackend : HostMatVecBackend
{
   using compute_type = ComputeType;
};

template < typename ComputeType = void >
struct NativeDeviceCOOBackend : DeviceMatVecBackend
{
   using compute_type = ComputeType;
};

// Standalone COO storage default. GenericAssembly dispatch uses
// DefaultBackendFor_t<MatrixAssemblyType::COO> instead.
using DefaultCOOBackend = HostCOOBackend<>;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCOOBackend >
struct COOMatrix;

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

   // Backend controls operator() dispatch. HostCOOBackend reads/writes host
   // data; NativeDeviceCOOBackend reads/writes device data.
   Backend backend{};

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const;
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

} // namespace gendil
