// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/cooassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSC/cscassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/CSR/csrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/defaultbackend.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/weakformtraversal.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <type_traits>

namespace gendil {

template < typename TrialSpace, typename TestSpace >
inline constexpr MatrixAssemblyType default_matrix_assembly_type_v =
   ( IsScalarDGL2Space< TrialSpace >::value &&
     IsScalarDGL2Space< TestSpace >::value )
      ? MatrixAssemblyType::BSR
      : MatrixAssemblyType::SGBSR;

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   if constexpr ( Type == MatrixAssemblyType::BSR )
   {
      return GenericBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else if constexpr ( Type == MatrixAssemblyType::SGBSR )
   {
      return GenericSGBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else if constexpr ( Type == MatrixAssemblyType::RawCOO )
   {
      (void) backend;
      return GenericRawCOOAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule );
   }
   else if constexpr ( Type == MatrixAssemblyType::COO )
   {
      return GenericCOOAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else if constexpr ( Type == MatrixAssemblyType::CSR )
   {
      return GenericCSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else if constexpr ( Type == MatrixAssemblyType::CSC )
   {
      return GenericCSCAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else
   {
      static_assert(
         dependent_false_value_v< Type >,
         "GenericAssembly: requested matrix assembly type is not implemented yet." );
   }
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   if constexpr (
      Type == MatrixAssemblyType::RawCOO ||
      Type == MatrixAssemblyType::BSR ||
      Type == MatrixAssemblyType::SGBSR ||
      Type == MatrixAssemblyType::COO ||
      Type == MatrixAssemblyType::CSR ||
      Type == MatrixAssemblyType::CSC )
   {
      return GenericAssembly<Type, KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         DefaultBackendFor_t< Type >{} );
   }
   else
   {
      static_assert(
         dependent_false_value_v< Type >,
         "GenericAssembly: requested matrix assembly type is not implemented yet." );
   }
}

} // namespace gendil
