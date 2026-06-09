// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/cooassembly.hpp"
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
   (void) backend;

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
         integration_rule );
   }
   else
   {
      static_assert(
         dependent_false_value_v< Type >,
         "GenericAssembly: CSR and CSC assembly are reserved canonical formats and are not implemented yet." );
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
   return GenericAssembly<Type, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{} );
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
      weak_form,
      wf_ctx,
      integration_rule );
}

} // namespace gendil
