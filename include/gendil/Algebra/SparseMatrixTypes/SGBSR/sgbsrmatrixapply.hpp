// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapply.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/SparseMatrixTypes/SGBSR/sgbsrmatrixstorage.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

namespace gendil
{

namespace details
{

template < typename Backend, typename VectorType >
concept SGBSRVectorAccessibleForBackend =
   ( is_host_matvec_backend_v< Backend > &&
     HostAccessibleVector< VectorType > ) ||
   ( is_device_matvec_backend_v< Backend > &&
     DeviceAccessibleVector< VectorType > );

} // namespace details

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
requires
   details::SGBSRVectorAccessibleForBackend< Backend, InputVector > &&
   details::SGBSRVectorAccessibleForBackend< Backend, OutputVector > &&
   GatherOperatorType< TrialGather, Backend, InputVector, Vector > &&
   ScatterOperatorType< TestScatter, Backend, Vector, OutputVector >
void Apply(
   const Backend & backend,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
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
requires
   details::SGBSRVectorAccessibleForBackend<
      typename BSRType::backend_type,
      InputVector > &&
   details::SGBSRVectorAccessibleForBackend<
      typename BSRType::backend_type,
      OutputVector > &&
   GatherOperatorType<
      TrialGather,
      typename BSRType::backend_type,
      InputVector,
      Vector > &&
   ScatterOperatorType<
      TestScatter,
      typename BSRType::backend_type,
      Vector,
      OutputVector >
void Apply(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
{
   Apply( matrix.bsr_matrix.backend, matrix, x_fe, y_fe );
}

template < typename BSRType, typename TrialGather, typename TestScatter >
template < typename InputVector, typename OutputVector >
void SGBSRMatrix< BSRType, TrialGather, TestScatter >::operator()(
   const InputVector & x_fe,
   OutputVector & y_fe ) const
{
   Apply( *this, x_fe, y_fe );
}

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
requires
   details::SGBSRVectorAccessibleForBackend< Backend, InputVector > &&
   details::SGBSRVectorAccessibleForBackend< Backend, OutputVector > &&
   GatherOperatorType< TrialGather, Backend, InputVector, Vector > &&
   ScatterOperatorType< TestScatter, Backend, Vector, OutputVector >
void ApplyAdd(
   const Backend & backend,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x_fe,
   OutputVector & y_fe )
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
requires
   details::SGBSRVectorAccessibleForBackend<
      typename BSRType::backend_type,
      InputVector > &&
   details::SGBSRVectorAccessibleForBackend<
      typename BSRType::backend_type,
      OutputVector > &&
   GatherOperatorType<
      TrialGather,
      typename BSRType::backend_type,
      InputVector,
      Vector > &&
   ScatterOperatorType<
      TestScatter,
      typename BSRType::backend_type,
      Vector,
      OutputVector >
void ApplyAdd(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   ApplyAdd( matrix.bsr_matrix.backend, matrix, x, y );
}

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > &,
   const InputVector &,
   OutputVector & )
{
   if constexpr (
      !is_host_matvec_backend_v< Backend > &&
      !is_device_matvec_backend_v< Backend > )
   {
      static_assert(
         dependent_false_v< Backend >,
         "SGBSRMatrix Apply backend must derive from HostMatVecBackend or "
         "DeviceMatVecBackend." );
   }
   else
   {
      static_assert(
         dependent_false_v<
            Backend,
            SGBSRMatrix< BSRType, TrialGather, TestScatter >,
            InputVector,
            OutputVector >,
         "No SGBSRMatrix Apply overload is available for this "
         "backend/vector combination." );
   }
}

template <
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void Apply(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > &,
   const InputVector &,
   OutputVector & )
{
   using Backend = typename BSRType::backend_type;
   if constexpr (
      !is_host_matvec_backend_v< Backend > &&
      !is_device_matvec_backend_v< Backend > )
   {
      static_assert(
         dependent_false_v< Backend >,
         "SGBSRMatrix Apply backend must derive from HostMatVecBackend or "
         "DeviceMatVecBackend." );
   }
   else
   {
      static_assert(
         dependent_false_v<
            SGBSRMatrix< BSRType, TrialGather, TestScatter >,
            InputVector,
            OutputVector >,
         "No SGBSRMatrix Apply overload is available for this vector "
         "combination." );
   }
}

template <
   typename Backend,
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const Backend &,
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > &,
   const InputVector &,
   OutputVector & )
{
   if constexpr (
      !is_host_matvec_backend_v< Backend > &&
      !is_device_matvec_backend_v< Backend > )
   {
      static_assert(
         dependent_false_v< Backend >,
         "SGBSRMatrix ApplyAdd backend must derive from HostMatVecBackend or "
         "DeviceMatVecBackend." );
   }
   else
   {
      static_assert(
         dependent_false_v<
            Backend,
            SGBSRMatrix< BSRType, TrialGather, TestScatter >,
            InputVector,
            OutputVector >,
         "No SGBSRMatrix ApplyAdd overload is available for this "
         "backend/vector combination." );
   }
}

template <
   typename BSRType,
   typename TrialGather,
   typename TestScatter,
   typename InputVector,
   typename OutputVector >
void ApplyAdd(
   const SGBSRMatrix< BSRType, TrialGather, TestScatter > &,
   const InputVector &,
   OutputVector & )
{
   using Backend = typename BSRType::backend_type;
   if constexpr (
      !is_host_matvec_backend_v< Backend > &&
      !is_device_matvec_backend_v< Backend > )
   {
      static_assert(
         dependent_false_v< Backend >,
         "SGBSRMatrix ApplyAdd backend must derive from HostMatVecBackend or "
         "DeviceMatVecBackend." );
   }
   else
   {
      static_assert(
         dependent_false_v<
            SGBSRMatrix< BSRType, TrialGather, TestScatter >,
            InputVector,
            OutputVector >,
         "No SGBSRMatrix ApplyAdd overload is available for this vector "
         "combination." );
   }
}

} // namespace gendil
