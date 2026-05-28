// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"

#include <utility>

namespace gendil
{

template < typename BSRType, typename TrialGather, typename TestScatter >
class SGBSRMatrix
{
public:
   SGBSRMatrix(
      BSRType bsr_matrix_,
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
   SGBSRMatrix( SGBSRMatrix && ) noexcept = default;
   SGBSRMatrix & operator=( SGBSRMatrix && ) noexcept = default;

   GlobalIndex TrialBsrSize() const
   {
      return TrialBsrSize_impl( bsr_matrix );
   }

   GlobalIndex TestBsrSize() const
   {
      return TestBsrSize_impl( bsr_matrix );
   }

   void operator()( const Vector & x_fe, Vector & y_fe ) const
   {
      // Workspaces are owned and reused; one SGBSRMatrix instance is not
      // thread-safe for concurrent applies.
      if constexpr ( TrialGather::is_identity && TestScatter::is_identity )
      {
         GENDIL_VERIFY(
            x_fe.Size() == static_cast< size_t >( TrialBsrSize() ),
            "SGBSRMatrix identity gather input has the wrong BSR size." );
         GENDIL_VERIFY(
            y_fe.Size() == static_cast< size_t >( TestBsrSize() ),
            "SGBSRMatrix identity scatter output has the wrong BSR size." );
         bsr_matrix( x_fe, y_fe );
      }
      else if constexpr ( TrialGather::is_identity )
      {
         GENDIL_VERIFY(
            x_fe.Size() == static_cast< size_t >( TrialBsrSize() ),
            "SGBSRMatrix identity gather input has the wrong BSR size." );
         bsr_matrix( x_fe, y_bsr );
         test_scatter( y_bsr, y_fe );
      }
      else if constexpr ( TestScatter::is_identity )
      {
         GENDIL_VERIFY(
            y_fe.Size() == static_cast< size_t >( TestBsrSize() ),
            "SGBSRMatrix identity scatter output has the wrong BSR size." );
         trial_gather( x_fe, x_bsr );
         bsr_matrix( x_bsr, y_fe );
      }
      else
      {
         trial_gather( x_fe, x_bsr );
         bsr_matrix( x_bsr, y_bsr );
         test_scatter( y_bsr, y_fe );
      }
   }

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

   BSRType bsr_matrix;
   TrialGather trial_gather;
   TestScatter test_scatter;

   mutable Vector x_bsr;
   mutable Vector y_bsr;
};

} // namespace gendil
