// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"

namespace gendil {

/**
 * Host-only RawCOO-to-CSC finalization policy.
 *
 * The policy reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(col, row)`, and additively reduces exact duplicate coordinates before
 * building canonical CSC column storage. Exact reduced zeros are retained.
 */
struct HostSortReduceRawCOOToCSCPolicy
{ };

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSC(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSCPolicy &,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::ColumnThenRow >( raw );

   auto matrix =
      MakeCSCMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   const StoredIndexType col_ptr_size =
      matrix_data.num_cols + StoredIndexType( 1 );
   for ( StoredIndexType col = 0; col < col_ptr_size; ++col )
   {
      matrix_data.col_ptr[col] = StoredIndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix_data.num_rows,
         "FinalizeRawCOOToCSC received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix_data.num_cols,
         "FinalizeRawCOOToCSC received a column outside the matrix dimensions." );
      ++matrix_data.col_ptr[triplet.col + StoredIndexType( 1 )];
   }

   for ( StoredIndexType col = 0; col < matrix_data.num_cols; ++col )
   {
      matrix_data.col_ptr[col + StoredIndexType( 1 )] +=
         matrix_data.col_ptr[col];
   }

   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.row_ind[i] = triplet.row;
      matrix_data.values[i] = triplet.value;
   }

   Sync( matrix );

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSC(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSCPolicy & policy )
{
   return FinalizeRawCOOToCSC(
      GetHostReadView( raw ),
      policy,
      DefaultCSCBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSC(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSCPolicy & policy,
   Backend backend )
{
   return FinalizeRawCOOToCSC(
      GetHostReadView( raw ),
      policy,
      std::move( backend ) );
}

} // namespace gendil
