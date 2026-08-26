// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixview.hpp"
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
   const CSCMatrixView< ValueType, IndexType > & matrix )
{
   return matrix.nnz == 0;
}

template < typename Matrix >
void CheckCSCApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(CSC backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(CSC backend, ...) output vector has the wrong size." );
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void CSCMatrix< ValueType, IndexType, MatrixBackend >::operator()(
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
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
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
   const CSCMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         CSCMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No CSCMatrix Apply overload is available for this backend/vector "
      "combination. Include cscmatrixapply.hpp and use a supported CSC "
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
   const CSCMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v<
         Backend,
         CSCMatrix< ValueType, IndexType, MatrixBackend >,
         InputVector,
         OutputVector >,
      "No CSCMatrix ApplyAdd overload is available for this backend/vector "
      "combination. Include cscmatrixapply.hpp and use a supported CSC "
      "backend/vector type." );
}

} // namespace gendil
