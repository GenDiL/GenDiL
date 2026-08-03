// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/vendorsparsebackend.hpp"

namespace gendil
{

/**
 * Configure a BSR apply backend from assembled storage metadata.
 *
 * Most backends require no configuration. Vendor backends provide overloads
 * that record format capabilities used for diagnostics and backend selection.
 */
template < typename Backend, typename IndexType >
void ConfigureBSRBackend(
   Backend &,
   const IndexType,
   const IndexType,
   const IndexType,
   const IndexType,
   const IndexType )
{ }

template < typename ComputeType, typename IndexType >
void ConfigureBSRBackend(
   CuSparseBSRBackend< ComputeType > & backend,
   const IndexType block_rows,
   const IndexType block_cols,
   const IndexType num_row_blocks,
   const IndexType num_col_blocks,
   const IndexType num_blocks )
{
   const bool storage_eligible =
      details::IsVendorBSRStorageEligible(
         block_rows,
         block_cols,
         num_row_blocks,
         num_col_blocks,
         num_blocks );

   backend.ConfigureStorage(
      static_cast< std::uint64_t >( block_rows ),
      static_cast< std::uint64_t >( block_cols ),
      storage_eligible );
}

template < typename ComputeType, typename IndexType >
void ConfigureBSRBackend(
   RocSparseBSRBackend< ComputeType > & backend,
   const IndexType block_rows,
   const IndexType block_cols,
   const IndexType num_row_blocks,
   const IndexType num_col_blocks,
   const IndexType num_blocks )
{
   const bool storage_eligible =
      details::IsVendorBSRStorageEligible(
         block_rows,
         block_cols,
         num_row_blocks,
         num_col_blocks,
         num_blocks );

   backend.ConfigureStorage(
      static_cast< std::uint64_t >( block_rows ),
      static_cast< std::uint64_t >( block_cols ),
      storage_eligible );
}

} // namespace gendil
