// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * @brief Host-only RawCOO-to-HypreCSR finalization policy.
 *
 * This is the Hypre-specific counterpart to the native CSR finalizer. It reads
 * the current host copy of `RawCOOTripletBuffer`, sorts entries by row, then
 * places explicit diagonal entries before the remaining increasing-column
 * entries in each row, and additively reduces exact duplicate coordinates
 * before building CSR storage. Exact reduced zeros are retained.
 *
 * The resulting matrix stores values as `HYPRE_Complex` and indices as
 * `HYPRE_Int`. This Hypre-oriented row ordering makes the explicit diagonal,
 * when present, the first entry in its row. The owned CSR arrays are suitable
 * for zero-copy aliasing by the `HYPRE_ParCSRMatrix` shell owned by
 * `HypreCSRMatrix`.
 */
struct HostSortReduceRawCOOToHypreCSRPolicy
{ };

namespace details
{

/**
 * @brief Cast a nonnegative size/index value to a Hypre integer type.
 *
 * Hypre's public ParCSR interfaces use `HYPRE_Int` for local dimensions and
 * entry counts. GenDiL assembly buffers may use wider index types, so this
 * helper checks both sign and upper range before narrowing.
 */
template < typename Target, typename Source >
Target CheckedHypreNarrow(
   const Source value,
   const char * message )
{
   if constexpr ( std::numeric_limits< Source >::is_signed )
   {
      GENDIL_VERIFY( value >= Source( 0 ), message );
   }

   using UnsignedSource = std::make_unsigned_t< Source >;
   GENDIL_VERIFY(
      static_cast< UnsignedSource >( value ) <=
         static_cast< UnsignedSource >(
            std::numeric_limits< Target >::max() ),
      message );

   return static_cast< Target >( value );
}

/**
 * @brief Build serial/rank-1 Hypre metadata for a host `HypreCSRMatrix`.
 *
 * Version 1 of the Hypre interface is intentionally serial: the full local
 * matrix is also the global matrix, row/column ownership starts at zero, and
 * Hypre uses `hypre_MPI_COMM_SELF`.
 */
inline HypreCSRMetadata MakeHypreCSRMetadata(
   const HYPRE_Int num_rows,
   const HYPRE_Int num_cols )
{
   HypreCSRMetadata metadata{};
   metadata.global_num_rows = static_cast< HYPRE_BigInt >( num_rows );
   metadata.global_num_cols = static_cast< HYPRE_BigInt >( num_cols );
   metadata.row_starts[0] = HYPRE_BigInt( 0 );
   metadata.row_starts[1] = metadata.global_num_rows;
   metadata.col_starts[0] = HYPRE_BigInt( 0 );
   metadata.col_starts[1] = metadata.global_num_cols;
   metadata.comm = hypre_MPI_COMM_SELF;
   metadata.is_square = ( num_rows == num_cols );
   metadata.diagonal_rows = std::min( num_rows, num_cols );
   return metadata;
}

} // namespace details

/**
 * @brief Finalize raw COO assembly into a Hypre-compatible owning CSR matrix.
 *
 * Algorithm:
 * 1. Sort raw COO triplets by `(row, diagonal first, col)` and reduce
 *    duplicates.
 * 2. Narrow local dimensions and reduced entry count to `HYPRE_Int`.
 * 3. Build CSR row counts and convert them to prefix sums.
 * 4. Fill `col_ind` and `values` using `HYPRE_Int` and `HYPRE_Complex`.
 * 5. Record diagonal metadata from the first entry of each eligible row.
 * 6. Mark the CSR arrays host/device current through `ToDevice`.
 *
 * This function does not insert missing diagonals. Missing diagonal entries are
 * recorded in the metadata so matvec-only and solver-oriented callers can make
 * different policy decisions later.
 */
template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToHypreCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToHypreCSRPolicy &,
   Backend backend )
{
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenDiagonalThenColumn >( raw );

   const HYPRE_Int num_rows =
      details::CheckedHypreNarrow< HYPRE_Int >(
         raw.num_rows,
         "FinalizeRawCOOToHypreCSR received too many rows for HYPRE_Int." );
   const HYPRE_Int num_cols =
      details::CheckedHypreNarrow< HYPRE_Int >(
         raw.num_cols,
         "FinalizeRawCOOToHypreCSR received too many columns for HYPRE_Int." );
   const HYPRE_Int nnz =
      details::CheckedHypreNarrow< HYPRE_Int >(
         reduced.size(),
         "FinalizeRawCOOToHypreCSR received too many entries for HYPRE_Int." );

   auto csr =
      MakeCSRMatrix< HYPRE_Complex, HYPRE_Int, HostCSRBackend<> >(
         num_rows,
         num_cols,
         nnz,
         HostCSRBackend<>{} );

   const HYPRE_Int row_ptr_size = num_rows + HYPRE_Int( 1 );
   for ( HYPRE_Int row = 0; row < row_ptr_size; ++row )
   {
      csr.row_ptr[row] = HYPRE_Int( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < raw.num_rows,
         "FinalizeRawCOOToHypreCSR received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < raw.num_cols,
         "FinalizeRawCOOToHypreCSR received a column outside the matrix dimensions." );

      const HYPRE_Int row =
         details::CheckedHypreNarrow< HYPRE_Int >(
            triplet.row,
            "FinalizeRawCOOToHypreCSR received a row outside HYPRE_Int range." );
      ++csr.row_ptr[row + HYPRE_Int( 1 )];
   }

   for ( HYPRE_Int row = 0; row < num_rows; ++row )
   {
      csr.row_ptr[row + HYPRE_Int( 1 )] += csr.row_ptr[row];
   }

   for ( HYPRE_Int i = 0; i < nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      csr.col_ind[i] =
         details::CheckedHypreNarrow< HYPRE_Int >(
            triplet.col,
            "FinalizeRawCOOToHypreCSR received a column outside HYPRE_Int range." );
      csr.values[i] = static_cast< HYPRE_Complex >( triplet.value );
   }

   auto metadata = details::MakeHypreCSRMetadata( num_rows, num_cols );

   // Hypre matvec does not require diagonal-first rows, but many Hypre solver
   // and preconditioner paths assume or prefer fast access to the diagonal as
   // the first row entry. The Hypre-specific sort order above produces that
   // layout directly, so metadata only needs to inspect the row start.
   for ( HYPRE_Int row = 0; row < num_rows; ++row )
   {
      const HYPRE_Int row_start = csr.row_ptr[row];
      const HYPRE_Int row_end = csr.row_ptr[row + HYPRE_Int( 1 )];
      bool has_diagonal = false;

      // Rectangular matrices can still be used as Hypre matvec operators. Only
      // rows with a valid local diagonal column can have a diagonal entry.
      if ( row < num_cols )
      {
         has_diagonal = row_start < row_end && csr.col_ind[row_start] == row;
      }

      if ( has_diagonal )
      {
         ++metadata.explicit_diagonal_count;
      }
      else if ( row < num_cols )
      {
         // Do not insert explicit zero diagonals here. Recording the first
         // missing diagonal keeps the finalizer usable for matvec while giving
         // solver wrappers enough information to reject unsuitable matrices.
         ++metadata.missing_diagonal_count;
         if ( metadata.first_missing_diagonal < 0 )
         {
            metadata.first_missing_diagonal = row;
         }
      }
   }

   metadata.has_explicit_diagonal =
      metadata.missing_diagonal_count == 0;

   // Mark both views current so native GenDiL apply and host Hypre aliasing can
   // share the same storage state without an immediate synchronization surprise.
   ToDevice( static_cast< GlobalIndex >( row_ptr_size ), csr.row_ptr );
   if ( csr.nnz > 0 )
   {
      ToDevice( static_cast< GlobalIndex >( csr.nnz ), csr.col_ind );
      ToDevice( static_cast< GlobalIndex >( csr.nnz ), csr.values );
   }

   return HypreCSRMatrix< Backend >{ std::move( csr ), metadata, backend };
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToHypreCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   const HostSortReduceRawCOOToHypreCSRPolicy & policy )
{
   return FinalizeRawCOOToHypreCSR(
      raw,
      policy,
      typename HypreCSRMatrix<>::backend_type{} );
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToHypreCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToHypreCSR(
      raw,
      HostSortReduceRawCOOToHypreCSRPolicy{} );
}

} // namespace gendil

#endif // GENDIL_USE_HYPRE
