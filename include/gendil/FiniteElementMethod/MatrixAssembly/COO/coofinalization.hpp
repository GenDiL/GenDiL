// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"

namespace gendil {

/**
 * Host-only RawCOO finalization policy.
 *
 * The policy reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(row, col)`, and additively reduces exact duplicate coordinates. Callers are
 * responsible for ensuring the raw host buffers are current before calling this
 * policy. The buffer overload obtains a host read view and therefore performs
 * any required device-to-host synchronization lazily.
 */
struct HostSortReduceRawCOOPolicy
{ };

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy &,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCOOMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.rows[i] = triplet.row;
      matrix_data.cols[i] = triplet.col;
      matrix_data.values[i] = triplet.value;
   }

   Sync( matrix );

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy & policy )
{
   return FinalizeRawCOOToCOO(
      GetHostReadView( raw ),
      policy,
      DefaultCOOBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy & policy,
   Backend backend )
{
   return FinalizeRawCOOToCOO(
      GetHostReadView( raw ),
      policy,
      std::move( backend ) );
}

} // namespace gendil
