// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrpattern.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

#include <type_traits>

namespace gendil {

template < typename KernelPolicy, typename BSRMatrixType >
void SyncAssembledBSRValues(
   BSRMatrixType & bsr_matrix )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex value_count =
      static_cast< GlobalIndex >( bsr_matrix.num_blocks ) *
      static_cast< GlobalIndex >( bsr_matrix.block_rows ) *
      static_cast< GlobalIndex >( bsr_matrix.block_cols );

   if constexpr ( is_host_configuration_v< KernelPolicy > )
   {
      ToDevice( value_count, bsr_matrix.values );
   }
   else
   {
      GENDIL_DEVICE_SYNC;
      ToHost( value_count, bsr_matrix.values );
   }
#else
   (void) bsr_matrix;
#endif
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericBSRElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   ValidateSparseAssemblyDomainSupport<WeakForm, WeakFormContext>();

   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   constexpr auto TestName = requirements<WeakForm>::test_name;
   static_assert(
      TrialName != StaticString{"Error"},
      "GenericBSRElementBlockDiagonalAssembly: missing TrialSpace in "
      "integrand.");
   static_assert(
      TestName != StaticString{"Error"},
      "GenericBSRElementBlockDiagonalAssembly: missing TestSpace in "
      "integrand.");

   const auto& trial_space =
      wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space =
      wf_ctx.template fe_field<TestName>().space;
   using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
   using TestSpace = std::remove_cvref_t<decltype(test_space)>;
   using TrialShapeFunctions =
      typename TrialSpace::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename TestSpace::finite_element_type::shape_functions;

   constexpr GlobalIndex ntrial =
      LocalDofCount<TrialShapeFunctions>();
   constexpr GlobalIndex ntest =
      LocalDofCount<TestShapeFunctions>();
   const auto& domain_mesh =
      GetCellIntegrationDomainMesh(weak_form, wf_ctx);
   const GlobalIndex num_elements =
      static_cast<GlobalIndex>(
         domain_mesh.GetNumberOfCells());

   auto bsr_matrix =
      MakeBlockDiagonalDGBSRPattern(
         num_elements,
         ntest,
         ntrial,
         backend);

   AssembleElementBlockDiagonalSparseTarget<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix);

   SyncAssembledBSRValues<KernelPolicy>(bsr_matrix);

   return bsr_matrix;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericBSRElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericBSRElementBlockDiagonalAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{});
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   ValidateSparseAssemblyDomainSupport<WeakForm, WeakFormContext>();
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   auto bsr_matrix = MakeDGBSRPattern( trial_space, backend );

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   SyncAssembledBSRValues< KernelPolicy >( bsr_matrix );

   return bsr_matrix;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericBSRAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{} );
}

} // namespace gendil
