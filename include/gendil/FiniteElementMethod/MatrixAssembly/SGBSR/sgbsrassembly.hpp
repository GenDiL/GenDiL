// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/SGBSR/sgbsrmatrix.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/assemblydispatch.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrgatherscatter.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template <
   class WeakForm,
   class DomainMesh,
   class TrialFESpace,
   class TestFESpace,
   typename Backend = DefaultBSRBackend >
auto MakeSGBSRInternalPattern(
   const DomainMesh & domain_mesh,
   const TrialFESpace & trial_space,
   const TestFESpace & test_space,
   Backend backend = Backend{} )
{
   using I = std::remove_cvref_t< WeakForm >;
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;
   constexpr GlobalIndex block_cols =
      LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex block_rows =
      LocalDofCount< TestShapeFunctions >();

   // SGBSR applies an element-block BSR operator through gather/scatter maps.
   // Cell-only forms can use the block-diagonal element pattern; facet forms
   // still need the DG element-neighbor block structure internally.
   if constexpr (
      has_boundary_facet_contributions_v< I > ||
      has_interior_facet_contributions_v< I > )
   {
      return MakeDGBSRPattern(
         domain_mesh,
         trial_space,
         test_space,
         std::move( backend ) );
   }
   else
   {
      return MakeBlockDiagonalDGBSRPattern(
         domain_mesh.GetNumberOfCells(),
         block_rows,
         block_cols,
         std::move( backend ) );
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
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::SGBSR,
         details::SparseAssemblyMode::Full,
         KernelPolicy,
         Backend>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName  = requirements<WeakForm>::test_name;

      const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space  = wf_ctx.template fe_field<TestName>().space;
      const auto& domain_mesh =
         GetCellIntegrationDomainMesh(weak_form, wf_ctx);

      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;

      auto bsr_matrix =
         MakeSGBSRInternalPattern< WeakForm >(
            domain_mesh,
            trial_space,
            test_space,
            std::move( backend ) );

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

      using BSRType = std::remove_cvref_t<decltype(bsr_matrix)>;
      using TrialGather = default_bsr_gather_t< TrialSpace >;
      using TestScatter = default_bsr_scatter_t< TestSpace >;

      return SGBSRMatrix< BSRType, TrialGather, TestScatter >(
         std::move( bsr_matrix ),
         DefaultBsrGatherFor< TrialSpace >::Make( trial_space ),
         DefaultBsrScatterFor< TestSpace >::Make( test_space ) );
   }
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
      MakeDefaultBSRBackend( weak_form, wf_ctx ) );
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericSGBSRElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::SGBSR,
         details::SparseAssemblyMode::ElementBlockDiagonal,
         KernelPolicy,
         Backend>(weak_form, wf_ctx);

   if constexpr (can_instantiate)
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName = requirements<WeakForm>::test_name;

      const auto& trial_space =
         wf_ctx.template fe_field<TrialName>().space;
      const auto& test_space =
         wf_ctx.template fe_field<TestName>().space;
      using TrialSpace = std::remove_cvref_t<decltype(trial_space)>;
      using TestSpace = std::remove_cvref_t<decltype(test_space)>;

      auto bsr_matrix =
         GenericBSRElementBlockDiagonalAssembly<KernelPolicy>(
            weak_form,
            wf_ctx,
            integration_rule,
            std::move( backend ));

      using BSRType = std::remove_cvref_t<decltype(bsr_matrix)>;
      using TrialGather = default_bsr_gather_t<TrialSpace>;
      using TestScatter = default_bsr_scatter_t<TestSpace>;

      return SGBSRMatrix<BSRType, TrialGather, TestScatter>(
         std::move(bsr_matrix),
         DefaultBsrGatherFor<TrialSpace>::Make(trial_space),
         DefaultBsrScatterFor<TestSpace>::Make(test_space));
   }
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericSGBSRElementBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericSGBSRElementBlockDiagonalAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      MakeDefaultBSRBackend( weak_form, wf_ctx ));
}

} // namespace gendil
