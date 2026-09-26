// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixview.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/Vector/access.hpp"
#include "gendil/Algebra/vectormapconcept.hpp"

#include <concepts>
#include <cstddef>

namespace gendil
{

namespace details
{

template < typename ValueType, typename IndexType >
GENDIL_HOST_DEVICE
constexpr bool IsEmpty(
   const CSRMatrixView< ValueType, IndexType > & matrix )
{
   return matrix.nnz == 0;
}

template < typename Matrix >
void CheckCSRApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(CSR backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(CSR backend, ...) output vector has the wrong size." );
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
