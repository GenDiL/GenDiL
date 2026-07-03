// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/HypreCSR/hyprecsrmatrixstorage.hpp"
#include "gendil/Interfaces/Hypre/hyprevector.hpp"

namespace gendil
{

template < typename MatrixBackend >
inline void HypreHostMatvec(
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const Vector & x,
   Vector & y,
   HYPRE_Complex alpha = HYPRE_Complex( 1 ),
   HYPRE_Complex beta = HYPRE_Complex( 0 ) )
{
   GENDIL_VERIFY(
      x.Size() == static_cast< size_t >( matrix.csr.num_cols ),
      "HypreMatvec input vector has the wrong size." );
   GENDIL_VERIFY(
      y.Size() == static_cast< size_t >( matrix.csr.num_rows ),
      "HypreMatvec output vector has the wrong size." );

   auto x_view =
      MakeHostHypreParVectorView(
         x,
         matrix.csr.num_cols,
         matrix.metadata.comm );
   auto y_view =
      MakeHostHypreParVectorView(
         y,
         matrix.csr.num_rows,
         matrix.metadata.comm );

   CheckHypreError(
      HYPRE_ParCSRMatrixMatvec(
         alpha,
         matrix.GetHostHypreParCSR(),
         x_view,
         beta,
         y_view ),
      "HYPRE_ParCSRMatrixMatvec failed" );
}

template < typename MatrixBackend >
inline void HypreDeviceMatvec(
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const Vector & x,
   Vector & y,
   HYPRE_Complex alpha = HYPRE_Complex( 1 ),
   HYPRE_Complex beta = HYPRE_Complex( 0 ) )
{
   GENDIL_VERIFY(
      x.Size() == static_cast< size_t >( matrix.csr.num_cols ),
      "HypreMatvec input vector has the wrong size." );
   GENDIL_VERIFY(
      y.Size() == static_cast< size_t >( matrix.csr.num_rows ),
      "HypreMatvec output vector has the wrong size." );

   auto x_view =
      MakeDeviceHypreParVectorView(
         x,
         matrix.csr.num_cols,
         matrix.metadata.comm );
   auto y_view =
      MakeDeviceHypreParVectorView(
         y,
         matrix.csr.num_rows,
         matrix.metadata.comm );

   CheckHypreError(
      HYPRE_ParCSRMatrixMatvec(
         alpha,
         matrix.GetDeviceHypreParCSR(),
         x_view,
         beta,
         y_view ),
      "HYPRE_ParCSRMatrixMatvec failed" );
}

template < typename MatrixBackend >
inline void HypreMatvec(
   const HypreCSRMatrix< MatrixBackend > & matrix,
   const Vector & x,
   Vector & y,
   HYPRE_Complex alpha = HYPRE_Complex( 1 ),
   HYPRE_Complex beta = HYPRE_Complex( 0 ) )
{
   if constexpr ( is_host_matvec_backend_v< MatrixBackend > )
   {
      HypreHostMatvec( matrix, x, y, alpha, beta );
   }
   else
   {
      HypreDeviceMatvec( matrix, x, y, alpha, beta );
   }
}

} // namespace gendil
