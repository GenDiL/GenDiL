// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"

namespace gendil {

/**
 * Host-only RawCOO-to-CSR finalization policy.
 *
 * The policy reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(row, col)`, and additively reduces exact duplicate coordinates before
 * building canonical CSR row storage. Exact reduced zeros are retained.
 */
struct HostSortReduceRawCOOToCSRPolicy
{ };

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSR(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSRPolicy &,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCSRMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   const StoredIndexType row_ptr_size =
      matrix_data.num_rows + StoredIndexType( 1 );
   for ( StoredIndexType row = 0; row < row_ptr_size; ++row )
   {
      matrix_data.row_ptr[row] = StoredIndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix_data.num_rows,
         "FinalizeRawCOOToCSR received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix_data.num_cols,
         "FinalizeRawCOOToCSR received a column outside the matrix dimensions." );
      ++matrix_data.row_ptr[triplet.row + StoredIndexType( 1 )];
   }

   for ( StoredIndexType row = 0; row < matrix_data.num_rows; ++row )
   {
      matrix_data.row_ptr[row + StoredIndexType( 1 )] +=
         matrix_data.row_ptr[row];
   }

   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.col_ind[i] = triplet.col;
      matrix_data.values[i] = triplet.value;
   }

   Sync( matrix );

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSRPolicy & policy )
{
   return FinalizeRawCOOToCSR(
      GetHostReadView( raw ),
      policy,
      DefaultCSRBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSRPolicy & policy,
   Backend backend )
{
   return FinalizeRawCOOToCSR(
      GetHostReadView( raw ),
      policy,
      std::move( backend ) );
}

} // namespace gendil
