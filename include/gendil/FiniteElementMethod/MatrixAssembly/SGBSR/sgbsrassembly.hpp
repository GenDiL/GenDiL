// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sgbsrmatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrgatherscatter.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template <
   class WeakForm,
   class FESpace,
   typename Backend = DefaultBSRBackend >
auto MakeSGBSRInternalPattern(
   const FESpace & trial_space,
   Backend backend = Backend{} )
{
   using I = std::remove_cvref_t< WeakForm >;

   // SGBSR applies an element-block BSR operator through gather/scatter maps.
   // Cell-only forms can use the block-diagonal element pattern; facet forms
   // still need the DG element-neighbor block structure internally.
   if constexpr (
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I > )
   {
      return MakeDGBSRPattern( trial_space, backend );
   }
   else
   {
      return MakeBlockDiagonalDGBSRPattern( trial_space, backend );
   }
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericSGBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericAssembly<SGBSR>: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericAssembly<SGBSR>: missing TestSpace in integrand.");

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;

   using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
   using TestSpace = std::remove_cvref_t<decltype(test_space)>;

   static_assert(
      std::is_same_v< TrialSpace, TestSpace >,
      "SGBSR GenericAssembly currently requires matching trial/test FE spaces; mixed/rectangular spaces are unsupported." );

   constexpr bool trial_is_h1 =
      std::is_same_v< typename TrialSpace::restriction_type, H1Restriction >;
   constexpr bool test_is_h1 =
      std::is_same_v< typename TestSpace::restriction_type, H1Restriction >;
   constexpr bool has_facet_terms =
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I >;

   static_assert(
      !( ( trial_is_h1 || test_is_h1 ) && has_facet_terms ),
      "SGBSR GenericAssembly currently supports H1Restriction cell terms only; H1 boundary/interior facet terms are unsupported." );

   auto bsr_matrix = MakeSGBSRInternalPattern< I >( trial_space, backend );

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   SyncAssembledBSRValues< KernelPolicy >( bsr_matrix );

   using BSRType = std::remove_cvref_t<decltype(bsr_matrix)>;
   using TrialGather = default_bsr_gather_t< TrialSpace >;
   using TestScatter = default_bsr_scatter_t< TestSpace >;

   return SGBSRMatrix< BSRType, TrialGather, TestScatter >(
      std::move( bsr_matrix ),
      DefaultBsrGatherFor< TrialSpace >::Make( trial_space ),
      DefaultBsrScatterFor< TestSpace >::Make( test_space ) );
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericSGBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericSGBSRAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{} );
}

} // namespace gendil
