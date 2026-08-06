// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrpattern.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

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

   ValidateWeakFormContext(weak_form, wf_ctx);

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
   details::ValidateBSRAssemblyBackendCompatibility<
      Backend,
      ntest,
      ntrial >();
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
         std::move( backend ));

   static_assert(
      is_host_configuration_v< KernelPolicy > !=
         is_device_configuration_v< KernelPolicy >,
      "BSR assembly requires a host or device kernel policy." );
   constexpr bool on_device =
      is_device_configuration_v< KernelPolicy >;
   auto matrix_view =
      GetKernelValuesReadWriteView< on_device >( bsr_matrix );

   AssembleElementBlockDiagonalSparseTarget<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      matrix_view);

   return bsr_matrix;
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
   constexpr auto TestName = requirements<WeakForm>::test_name;
   static_assert(
      TrialName != StaticString{"Error"},
      "GenericBSRAssembly: missing TrialSpace in integrand." );
   static_assert(
      TestName != StaticString{"Error"},
      "GenericBSRAssembly: missing TestSpace in integrand." );
   ValidateWeakFormContext(weak_form, wf_ctx);
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space = wf_ctx.template fe_field<TestName>().space;
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
   details::ValidateBSRAssemblyBackendCompatibility<
      Backend,
      ntest,
      ntrial >();
   const auto& domain_mesh =
      GetCellIntegrationDomainMesh(weak_form, wf_ctx);
   auto bsr_matrix =
      MakeDGBSRPattern(
         domain_mesh,
         trial_space,
         test_space,
         std::move( backend ) );

   static_assert(
      is_host_configuration_v< KernelPolicy > !=
         is_device_configuration_v< KernelPolicy >,
      "BSR assembly requires a host or device kernel policy." );
   constexpr bool on_device =
      is_device_configuration_v< KernelPolicy >;
   auto matrix_view =
      GetKernelValuesReadWriteView< on_device >( bsr_matrix );

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      matrix_view
   );

   return bsr_matrix;
}

template < class WeakForm, class WeakFormContext >
auto MakeDefaultBSRBackend(
   const WeakForm & weak_form,
   const WeakFormContext & wf_ctx )
{
   using I = std::remove_cvref_t< decltype( weak_form ) >;
   constexpr auto TrialName = requirements< I >::trial_name;
   constexpr auto TestName = requirements< I >::test_name;
   static_assert(
      TrialName != StaticString{"Error"},
      "Default BSR backend selection requires a TrialSpace in the integrand." );
   static_assert(
      TestName != StaticString{"Error"},
      "Default BSR backend selection requires a TestSpace in the integrand." );

   const auto & trial_space =
      wf_ctx.template fe_field< TrialName >().space;
   const auto & test_space =
      wf_ctx.template fe_field< TestName >().space;
   using TrialSpace =
      std::remove_cvref_t< decltype( trial_space ) >;
   using TestSpace =
      std::remove_cvref_t< decltype( test_space ) >;
   using TrialShapeFunctions =
      typename TrialSpace::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename TestSpace::finite_element_type::shape_functions;

#if defined(GENDIL_USE_DEVICE)
   constexpr GlobalIndex ntrial =
      LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex ntest =
      LocalDofCount< TestShapeFunctions >();
   constexpr bool vendor_bsr_available =
#if defined(GENDIL_USE_CUDA)
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR)
      true;
#else
      false;
#endif
#elif defined(GENDIL_USE_HIP)
#if defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
      true;
#else
      false;
#endif
#else
      false;
#endif

   if constexpr ( ntrial == ntest && vendor_bsr_available )
   {
      return VendorDeviceBSRBackend<>{};
   }
   else
   {
      return NativeDeviceBSRBackend<>{};
   }
#else
   return HostBSRBackend<>{};
#endif
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
      MakeDefaultBSRBackend( weak_form, wf_ctx ));
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
      MakeDefaultBSRBackend( weak_form, wf_ctx ) );
}

} // namespace gendil
