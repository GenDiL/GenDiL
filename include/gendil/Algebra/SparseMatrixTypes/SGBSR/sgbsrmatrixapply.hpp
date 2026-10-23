// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapply.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/SparseMatrixTypes/SGBSR/sgbsrmatrixstorage.hpp"

namespace gendil
{

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend & backend,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
requires
   details::MatVecBackendReadableVector< Backend, InputVector > &&
   details::MatVecBackendWritableVector< Backend, OutputVector > &&
   GatherOperatorType< TrialGather, Backend, InputVector, Vector > &&
   ScatterOperatorType< TestScatter, Backend, Vector, OutputVector > &&
   requires
   {
      { Apply( backend, matrix.bsr_matrix, matrix.x_bsr, matrix.y_bsr ) }
         -> std::same_as< void >;
   }
{
   // Workspaces are owned and reused; one SGBSRMatrix instance is not
   // thread-safe for concurrent applies.
   matrix.trial_gather( backend, x_fe, matrix.x_bsr );
   gendil::Apply(
      backend,
      matrix.bsr_matrix,
      matrix.x_bsr,
      matrix.y_bsr );
   matrix.test_scatter( backend, matrix.y_bsr, y_fe );
}

template <
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void Apply(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
requires requires
{
   { Apply( matrix.bsr_matrix.backend, matrix, x_fe, y_fe ) }
      -> std::same_as< void >;
}
{
   Apply( matrix.bsr_matrix.backend, matrix, x_fe, y_fe );
}

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const Backend & backend,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
requires
   details::MatVecBackendReadableVector< Backend, InputVector > &&
   details::MatVecBackendWritableVector< Backend, OutputVector > &&
   GatherOperatorType< TrialGather, Backend, InputVector, Vector > &&
   ScatterOperatorType< TestScatter, Backend, Vector, OutputVector > &&
   requires
   {
      { Apply( backend, matrix.bsr_matrix, matrix.x_bsr, matrix.y_bsr ) }
         -> std::same_as< void >;
   }
{
   matrix.trial_gather( backend, x_fe, matrix.x_bsr );
   gendil::Apply(
      backend,
      matrix.bsr_matrix,
      matrix.x_bsr,
      matrix.y_bsr );
   matrix.test_scatter.ApplyAdd(
      backend,
      matrix.y_bsr,
      y_fe );
}

template <
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x,
   OutputVector & y )
requires requires
{
   { ApplyAdd( matrix.bsr_matrix.backend, matrix, x, y ) }
      -> std::same_as< void >;
}
{
   ApplyAdd( matrix.bsr_matrix.backend, matrix, x, y );
}

} // namespace gendil
