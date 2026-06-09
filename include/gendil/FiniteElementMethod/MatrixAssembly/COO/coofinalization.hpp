// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/rawcoosortreduce.hpp"

namespace gendil {

/**
 * Host-only RawCOO finalization policy.
 *
 * The policy reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(row, col)`, and additively reduces exact duplicate coordinates. Callers are
 * responsible for ensuring the raw host buffers are current before calling this
 * policy; `GenericRawCOOAssembly` does that before returning its RawCOO buffer.
 */
struct HostSortReduceRawCOOPolicy
{ };

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy &,
   Backend backend )
{
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCOOMatrix< ValueType, IndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< IndexType >( reduced.size() ),
         backend );

   for ( IndexType i = 0; i < matrix.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix.rows[i] = triplet.row;
      matrix.cols[i] = triplet.col;
      matrix.values[i] = triplet.value;
   }

   if ( matrix.nnz > 0 )
   {
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.rows );
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.cols );
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.values );
   }

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy & policy )
{
   return FinalizeRawCOOToCOO(
      raw,
      policy,
      DefaultCOOBackend{} );
}

} // namespace gendil
