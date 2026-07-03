// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcootripletbuffer.hpp"
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
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSRPolicy &,
   Backend backend )
{
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCSRMatrix< ValueType, IndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< IndexType >( reduced.size() ),
         backend );

   const IndexType row_ptr_size = matrix.num_rows + IndexType( 1 );
   for ( IndexType row = 0; row < row_ptr_size; ++row )
   {
      matrix.row_ptr[row] = IndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix.num_rows,
         "FinalizeRawCOOToCSR received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix.num_cols,
         "FinalizeRawCOOToCSR received a column outside the matrix dimensions." );
      ++matrix.row_ptr[triplet.row + IndexType( 1 )];
   }

   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      matrix.row_ptr[row + IndexType( 1 )] += matrix.row_ptr[row];
   }

   for ( IndexType i = 0; i < matrix.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix.col_ind[i] = triplet.col;
      matrix.values[i] = triplet.value;
   }

   ToDevice( static_cast< GlobalIndex >( row_ptr_size ), matrix.row_ptr );
   if ( matrix.nnz > 0 )
   {
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.col_ind );
      ToDevice( static_cast< GlobalIndex >( matrix.nnz ), matrix.values );
   }

   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToCSRPolicy & policy )
{
   return FinalizeRawCOOToCSR(
      raw,
      policy,
      DefaultCSRBackend{} );
}

} // namespace gendil
