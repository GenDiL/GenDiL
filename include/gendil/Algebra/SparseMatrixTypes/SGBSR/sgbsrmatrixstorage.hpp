// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/prelude.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * Move-only scatter/gather composition around an owning BSR matrix.
 *
 * The composed BSR matrix, gather/scatter operators, and reusable workspaces
 * are public low-level state. Before changing the BSR sparsity structure,
 * callers must reset every vendor backend cached against it, including
 * `bsr_matrix.backend`.
 *
 * The backend selected by `Apply` also selects the memory space used by every
 * nonidentity gather and scatter stage.
 *
 * One instance is not safe for concurrent application because `x_bsr` and
 * `y_bsr` are reused by logically const operations.
 */
template < typename BSRType, typename TrialGather, typename TestScatter >
struct SGBSRMatrix
{
   using bsr_type = BSRType;
   using backend_type = typename BSRType::backend_type;

   BSRType bsr_matrix;
   TrialGather trial_gather;
   TestScatter test_scatter;

   mutable Vector x_bsr;
   mutable Vector y_bsr;

   SGBSRMatrix(
      BSRType && bsr_matrix_,
      TrialGather trial_gather_,
      TestScatter test_scatter_ )
   : bsr_matrix( std::move( bsr_matrix_ ) ),
     trial_gather( std::move( trial_gather_ ) ),
     test_scatter( std::move( test_scatter_ ) ),
     x_bsr( static_cast< size_t >( TrialBsrSize_impl( bsr_matrix ) ) ),
     y_bsr( static_cast< size_t >( TestBsrSize_impl( bsr_matrix ) ) )
   { }

   SGBSRMatrix( const SGBSRMatrix & ) = delete;
   SGBSRMatrix & operator=( const SGBSRMatrix & ) = delete;
   SGBSRMatrix( SGBSRMatrix && )
      noexcept(
         std::is_nothrow_move_constructible_v< BSRType > &&
         std::is_nothrow_move_constructible_v< TrialGather > &&
         std::is_nothrow_move_constructible_v< TestScatter > &&
         std::is_nothrow_move_constructible_v< Vector > ) = default;
   SGBSRMatrix & operator=( SGBSRMatrix && )
      noexcept(
         std::is_nothrow_move_assignable_v< BSRType > &&
         std::is_nothrow_move_assignable_v< TrialGather > &&
         std::is_nothrow_move_assignable_v< TestScatter > &&
         std::is_nothrow_move_assignable_v< Vector > ) = default;

   GlobalIndex TrialBsrSize() const
   {
      return TrialBsrSize_impl( bsr_matrix );
   }

   GlobalIndex TestBsrSize() const
   {
      return TestBsrSize_impl( bsr_matrix );
   }

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x_fe, OutputVector & y_fe ) const;

private:
   static GlobalIndex TrialBsrSize_impl( const BSRType & matrix )
   {
      return static_cast< GlobalIndex >( matrix.num_col_blocks ) *
         static_cast< GlobalIndex >( matrix.block_cols );
   }

   static GlobalIndex TestBsrSize_impl( const BSRType & matrix )
   {
      return static_cast< GlobalIndex >( matrix.num_row_blocks ) *
         static_cast< GlobalIndex >( matrix.block_rows );
   }
};

} // namespace gendil
