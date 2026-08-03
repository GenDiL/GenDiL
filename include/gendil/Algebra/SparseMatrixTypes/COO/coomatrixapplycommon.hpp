// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixview.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <cstddef>

namespace gendil
{

namespace details
{

template < typename ValueType, typename IndexType >
GENDIL_HOST_DEVICE
constexpr bool IsEmpty(
   const COOMatrixView< ValueType, IndexType > & matrix )
{
   return matrix.nnz == 0;
}

template < typename Matrix >
void CheckCOOApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(COO backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(COO backend, ...) output vector has the wrong size." );
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void COOMatrix< ValueType, IndexType, MatrixBackend >::operator()(
   const InputVector & x,
   OutputVector & y ) const
{
   Apply( backend, *this, x, y );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   ApplyAdd( matrix.backend, matrix, x, y );
}

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         COOMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No COOMatrix Apply overload is available for this backend/vector "
      "combination. Include coomatrixapply.hpp and use a supported COO "
      "backend/vector type." );
}

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const Backend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         COOMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No COOMatrix ApplyAdd overload is available for this backend/vector "
      "combination. Include coomatrixapply.hpp and use a supported COO "
      "backend/vector type." );
}

} // namespace gendil
