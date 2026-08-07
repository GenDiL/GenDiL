// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/prelude.hpp"

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace gendil
{

template <
   typename Gather,
   typename Backend,
   typename ExternalInputVector,
   typename InternalOutputVector >
concept GatherOperatorType = requires(
   const Gather & gather,
   const Backend & backend,
   const ExternalInputVector & external,
   InternalOutputVector & internal )
{
   { gather.ExternalSize() } -> std::convertible_to< GlobalIndex >;
   { gather.InternalSize() } -> std::convertible_to< GlobalIndex >;
   gather( backend, external, internal );
};

template <
   typename Scatter,
   typename Backend,
   typename InternalInputVector,
   typename ExternalOutputVector >
concept ScatterOperatorType = requires(
   const Scatter & scatter,
   const Backend & backend,
   const InternalInputVector & internal,
   ExternalOutputVector & external )
{
   { scatter.ExternalSize() } -> std::convertible_to< GlobalIndex >;
   { scatter.InternalSize() } -> std::convertible_to< GlobalIndex >;
   scatter( backend, internal, external );
   scatter.ApplyAdd( backend, internal, external );
};

/**
 * Move-only scatter/gather composition around an owning BSR matrix.
 *
 * The composed BSR matrix, gather/scatter operators, and reusable workspaces
 * are public low-level state. Before changing the BSR sparsity structure,
 * callers must reset every vendor backend cached against it, including
 * `bsr_matrix.backend`.
 *
 * The backend selected by `Apply` also selects the memory space used by every
 * gather and scatter stage.
 *
 * One instance is not safe for concurrent application because `x_bsr` and
 * `y_bsr` are reused by logically const operations.
 */
template < typename BSRType, typename TrialGather, typename TestScatter >
struct SGBSRMatrix
{
   using bsr_type = BSRType;
   using backend_type = typename BSRType::backend_type;

   static_assert(
      GatherOperatorType< TrialGather, backend_type, Vector, Vector >,
      "SGBSRMatrix trial gather must satisfy GatherOperatorType for the "
      "stored backend and GenDiL Vector." );
   static_assert(
      ScatterOperatorType< TestScatter, backend_type, Vector, Vector >,
      "SGBSRMatrix test scatter must satisfy ScatterOperatorType for the "
      "stored backend and GenDiL Vector." );

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
     x_bsr( static_cast< size_t >(
        ValidatedTrialBsrSize( bsr_matrix, trial_gather ) ) ),
     y_bsr( static_cast< size_t >(
        ValidatedTestBsrSize( bsr_matrix, test_scatter ) ) )
   { }

   SGBSRMatrix( const SGBSRMatrix & ) = delete;
   SGBSRMatrix & operator=( const SGBSRMatrix & ) = delete;
   SGBSRMatrix( SGBSRMatrix && )
      noexcept(
         std::is_nothrow_move_constructible_v< BSRType > &&
         std::is_nothrow_move_constructible_v< TrialGather > &&
         std::is_nothrow_move_constructible_v< TestScatter > &&
         std::is_nothrow_move_constructible_v< Vector > ) = default;
   SGBSRMatrix & operator=( SGBSRMatrix && other )
      noexcept(
         std::is_nothrow_move_constructible_v< BSRType > &&
         std::is_nothrow_move_constructible_v< TrialGather > &&
         std::is_nothrow_move_constructible_v< TestScatter > &&
         std::is_nothrow_move_constructible_v< Vector > )
   {
      if ( this != &other )
      {
         std::destroy_at( this );
         std::construct_at( this, std::move( other ) );
      }
      return *this;
   }

   GlobalIndex TrialBsrSize() const
   {
      return TrialBsrSize_impl( bsr_matrix );
   }

   GlobalIndex TestBsrSize() const
   {
      return TestBsrSize_impl( bsr_matrix );
   }

   GlobalIndex NumCols() const
   {
      return static_cast< GlobalIndex >( trial_gather.ExternalSize() );
   }

   GlobalIndex NumRows() const
   {
      return static_cast< GlobalIndex >( test_scatter.ExternalSize() );
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

   static GlobalIndex ValidatedTrialBsrSize(
      const BSRType & matrix,
      const TrialGather & gather )
   {
      const GlobalIndex bsr_size = TrialBsrSize_impl( matrix );
      GENDIL_VERIFY(
         static_cast< GlobalIndex >( gather.InternalSize() ) == bsr_size,
         "SGBSRMatrix trial gather internal size does not match the BSR "
         "column-block span." );
      return bsr_size;
   }

   static GlobalIndex ValidatedTestBsrSize(
      const BSRType & matrix,
      const TestScatter & scatter )
   {
      const GlobalIndex bsr_size = TestBsrSize_impl( matrix );
      GENDIL_VERIFY(
         static_cast< GlobalIndex >( scatter.InternalSize() ) == bsr_size,
         "SGBSRMatrix test scatter internal size does not match the BSR "
         "row-block span." );
      return bsr_size;
   }
};

} // namespace gendil
