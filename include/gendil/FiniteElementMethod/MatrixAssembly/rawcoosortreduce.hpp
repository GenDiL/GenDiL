// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"

#include <algorithm>
#include <vector>

namespace gendil {

enum class SparseCoordinateOrder
{
   RowThenColumn,
   ColumnThenRow
};

namespace details
{

template < typename ValueType, typename IndexType >
struct RawCOOTriplet
{
   IndexType row;
   IndexType col;
   ValueType value;
};

template < SparseCoordinateOrder Order, typename Triplet >
bool RawCOOTripletLess( const Triplet & lhs, const Triplet & rhs )
{
   if constexpr ( Order == SparseCoordinateOrder::RowThenColumn )
   {
      if ( lhs.row != rhs.row )
      {
         return lhs.row < rhs.row;
      }
      return lhs.col < rhs.col;
   }
   else
   {
      if ( lhs.col != rhs.col )
      {
         return lhs.col < rhs.col;
      }
      return lhs.row < rhs.row;
   }
}

template <
   SparseCoordinateOrder Order,
   typename ValueType,
   typename IndexType >
auto MakeSortedReducedRawCOOTriplets(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   using Triplet = RawCOOTriplet< ValueType, IndexType >;

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
         return RawCOOTripletLess< Order >( lhs, rhs );
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

   return reduced;
}

} // namespace details

} // namespace gendil
