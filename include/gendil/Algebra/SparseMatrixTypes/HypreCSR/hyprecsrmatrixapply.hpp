// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_HYPRE

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapply.hpp"
#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixstorage.hpp"
#include "gendil/Interfaces/Hypre/hyprematvec.hpp"

namespace gendil
{

template <
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HypreCSRHostBackend &,
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   HypreHostMatvec( matrix, x, y );
}

template <
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HypreCSRDeviceBackend &,
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   HypreDeviceMatvec( matrix, x, y );
}

template <
   typename Backend,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend & backend,
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   Apply( backend, matrix.csr, x, y );
}

template <
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   Apply( matrix.backend, matrix, x, y );
}

template < typename MatrixBackend >
template < typename InputVector, typename OutputVector >
void HypreCSRMatrix< MatrixBackend >::operator()(
   const InputVector & x,
   OutputVector & y ) const
{
   Apply( *this, x, y );
}

} // namespace gendil

#endif // GENDIL_USE_HYPRE
