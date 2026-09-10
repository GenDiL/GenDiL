// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixview.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace gendil
{

namespace details
{

template < typename ValueType, typename IndexType, BlockLayout Layout >
GENDIL_HOST_DEVICE
constexpr bool IsEmpty(
   const BSRMatrixView< ValueType, IndexType, Layout > & matrix )
{
   return matrix.num_blocks == 0;
}

template < typename Matrix >
GlobalIndex CheckBSRApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   using IndexType = typename Matrix::index_type;

   GENDIL_VERIFY(
      matrix.block_rows > 0,
      "Apply(BSR backend, ...) requires a positive row block size." );
   GENDIL_VERIFY(
      matrix.block_cols > 0,
      "Apply(BSR backend, ...) requires a positive column block size." );

   if constexpr ( std::is_signed_v< IndexType > )
   {
      GENDIL_VERIFY(
         matrix.num_row_blocks >= 0 &&
         matrix.num_col_blocks >= 0,
         "Apply(BSR backend, ...) requires nonnegative block dimensions." );
   }

   const GlobalIndex num_col_blocks =
      static_cast< GlobalIndex >( matrix.num_col_blocks );
   const GlobalIndex num_row_blocks =
      static_cast< GlobalIndex >( matrix.num_row_blocks );
   const GlobalIndex block_cols =
      static_cast< GlobalIndex >( matrix.block_cols );
   const GlobalIndex block_rows =
      static_cast< GlobalIndex >( matrix.block_rows );
   const GlobalIndex expected_x_size = CheckedMultiply(
      num_col_blocks,
      block_cols,
      "Apply(BSR backend, ...) input dimension overflows GlobalIndex." );
   const GlobalIndex expected_y_size = CheckedMultiply(
      num_row_blocks,
      block_rows,
      "Apply(BSR backend, ...) output dimension overflows GlobalIndex." );

   GENDIL_VERIFY(
      x_size == static_cast< size_t >( expected_x_size ),
      "Apply(BSR backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( expected_y_size ),
      "Apply(BSR backend, ...) output vector has the wrong size." );

   return expected_y_size;
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const BSRMatrix< ValueType, IndexType, Layout, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
requires requires
{
   { ApplyAdd( matrix.backend, matrix, x, y ) } -> std::same_as< void >;
}
{
   ApplyAdd( matrix.backend, matrix, x, y );
}

} // namespace gendil
