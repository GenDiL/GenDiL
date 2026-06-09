// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/cscmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/rawcoosortreduce.hpp"

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
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSCPolicy &,
   Backend backend )
{
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::ColumnThenRow >( raw );

   auto matrix =
      MakeCSCMatrix< ValueType, IndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< IndexType >( reduced.size() ),
         backend );

   const IndexType col_ptr_size = matrix.num_cols + IndexType( 1 );
   for ( IndexType col = 0; col < col_ptr_size; ++col )
   {
      matrix.col_ptr[col] = IndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix.num_rows,
         "FinalizeRawCOOToCSC received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix.num_cols,
         "FinalizeRawCOOToCSC received a column outside the matrix dimensions." );
      ++matrix.col_ptr[triplet.col + IndexType( 1 )];
   }

   for ( IndexType col = 0; col < matrix.num_cols; ++col )
   {
      matrix.col_ptr[col + IndexType( 1 )] += matrix.col_ptr[col];
   }

   for ( IndexType i = 0; i < matrix.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix.row_ind[i] = triplet.row;
      matrix.values[i] = triplet.value;
   }

   ToDevice( static_cast< GlobalIndex >( col_ptr_size ), matrix.col_ptr );
   if ( matrix.nnz > 0 )
   {
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.row_ind );
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.values );
   }

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSC(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSCPolicy & policy )
{
   return FinalizeRawCOOToCSC(
      raw,
      policy,
      DefaultCSCBackend{} );
}

} // namespace gendil
