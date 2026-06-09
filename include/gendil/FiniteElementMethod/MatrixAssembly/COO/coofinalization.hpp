// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/coomatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"

#include <algorithm>
#include <vector>

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

namespace details
{

template < typename ValueType, typename IndexType >
struct COOTriplet
{
   IndexType row;
   IndexType col;
   ValueType value;
};

} // namespace details

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOPolicy & )
{
   using Triplet = details::COOTriplet< ValueType, IndexType >;

   std::vector< Triplet > triplets;
   triplets.reserve( static_cast< size_t >( raw.nnz_raw ) );

   for ( IndexType i = 0; i < raw.nnz_raw; ++i )
   {
      triplets.push_back(
         Triplet{ raw.rows[i], raw.cols[i], raw.values[i] } );
   }

   std::sort(
      triplets.begin(),
      triplets.end(),
      [] ( const Triplet & lhs, const Triplet & rhs )
      {
         if ( lhs.row != rhs.row )
         {
            return lhs.row < rhs.row;
         }
         return lhs.col < rhs.col;
      } );

   std::vector< Triplet > reduced;
   reduced.reserve( triplets.size() );

   for ( const auto & triplet : triplets )
   {
      if ( !reduced.empty() &&
           reduced.back().row == triplet.row &&
           reduced.back().col == triplet.col )
      {
         reduced.back().value += triplet.value;
      }
      else
      {
         reduced.push_back( triplet );
      }
   }

   auto matrix =
      MakeCOOMatrix< ValueType, IndexType >(
         raw.num_rows,
         raw.num_cols,
         static_cast< IndexType >( reduced.size() ) );

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

} // namespace gendil
