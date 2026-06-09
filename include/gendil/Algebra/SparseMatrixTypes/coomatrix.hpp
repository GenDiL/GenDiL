// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

namespace gendil {

/**
 * Canonical coordinate-list sparse matrix.
 *
 * Entries are sorted lexicographically by `(row, col)`, and each coordinate
 * appears at most once. Duplicate raw assembly triplets are finalized by
 * additive reduction before they are stored here. Exact zero values produced by
 * reduction are retained.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct COOMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   HostDevicePointer< IndexType > rows;
   HostDevicePointer< IndexType > cols;
   HostDevicePointer< ValueType > values;
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
auto MakeCOOMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz )
{
   COOMatrix< ValueType, IndexType > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;

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
