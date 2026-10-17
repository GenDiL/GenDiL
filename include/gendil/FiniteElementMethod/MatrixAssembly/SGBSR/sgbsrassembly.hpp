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
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   class DomainMesh,
   class TrialFESpace,
   class TestFESpace,
   typename Backend = DefaultBSRBackend >
auto MakeSGBSRInternalPattern(
   const DomainMesh & domain_mesh,
   const TrialFESpace & trial_space,
   const TestFESpace & test_space,
   Backend && backend = Backend{} )
{
   using Form = std::remove_cvref_t< WeakForm >;
   using TrialShapeFunctions =
      finite_element_space_shape_functions_t< TrialFESpace >;
   using TestShapeFunctions =
      finite_element_space_shape_functions_t< TestFESpace >;
   constexpr GlobalIndex block_cols =
      LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex block_rows =
      LocalDofCount< TestShapeFunctions >();

   // SGBSR applies an element-block BSR operator through gather/scatter maps.
   // Full facet assembly retains the DG element-neighbor pattern, including
   // boundary-only forms. Cell-only and element-block assembly use the diagonal.
   if constexpr (
      Mode == SparseAssemblyMode::Full &&
      ( has_boundary_facet_contributions_v< Form > ||
        has_interior_facet_contributions_v< Form > ) )
   {
      return MakeDGBSRPattern(
         domain_mesh,
         trial_space,
         test_space,
         std::forward< Backend >( backend ) );
   }
   else
   {
      return MakeBlockDiagonalDGBSRPattern(
         domain_mesh.GetNumberOfCells(),
         block_rows,
         block_cols,
         std::forward< Backend >( backend ) );
   }
}

template <
   typename KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename Backend >
auto GenericSGBSRAssembly(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   Backend && backend )
{
   using BackendType = std::remove_cvref_t< Backend >;
   const auto can_instantiate =
      details::ValidateSparseAssemblyInputs<
         MatrixAssemblyType::SGBSR,
         Mode,
         KernelPolicy,
         BackendType >( weak_form, weak_form_context );

   if constexpr ( can_instantiate )
   {
      constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< WeakForm >;
      constexpr auto TestName = weak_form_traits::only_test_field_name_v< WeakForm >;

      const auto & trial_space = weak_form_context.template GetFiniteElementField< TrialName >().space;
      const auto & test_space = weak_form_context.template GetFiniteElementField< TestName >().space;
      const auto & domain_mesh = GetCellIntegrationDomainMesh( weak_form, weak_form_context );
      using TrialSpace = std::remove_cvref_t< decltype( trial_space ) >;
      using TestSpace = std::remove_cvref_t< decltype( test_space ) >;

      auto bsr_matrix = MakeSGBSRInternalPattern< WeakForm, Mode >(
         domain_mesh,
         trial_space,
         test_space,
         std::forward< Backend >( backend ) );

      if constexpr ( !is_zero< WeakForm > )
      {
         constexpr bool on_device = is_device_configuration_v< KernelPolicy >;
         auto matrix_view = GetKernelValuesReadWriteView< on_device >( bsr_matrix );
         AssembleSparseTarget< KernelPolicy, Mode >(
            weak_form,
            weak_form_context,
            integration_rule,
            matrix_view );
      }

      return SGBSRMatrix(
         std::move( bsr_matrix ),
         DefaultBsrGatherFor< TrialSpace >::Make( trial_space ),
         DefaultBsrScatterFor< TestSpace >::Make( test_space ) );
   }
}

template <
   typename KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule >
auto GenericSGBSRAssembly(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule )
{
   return GenericSGBSRAssembly< KernelPolicy, Mode >(
      weak_form,
      weak_form_context,
      integration_rule,
      MakeDefaultBSRBackend( weak_form, weak_form_context ) );
}

template <
   typename KernelPolicy,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename Backend >
auto GenericSGBSRElementBlockDiagonalAssembly(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   Backend && backend )
{
   return GenericSGBSRAssembly< KernelPolicy, SparseAssemblyMode::ElementBlockDiagonal >(
      weak_form,
      weak_form_context,
      integration_rule,
      std::forward< Backend >( backend ) );
}

template <
   typename KernelPolicy,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule >
auto GenericSGBSRElementBlockDiagonalAssembly(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule )
{
   return GenericSGBSRAssembly< KernelPolicy, SparseAssemblyMode::ElementBlockDiagonal >(
      weak_form,
      weak_form_context,
      integration_rule );
}

} // namespace gendil
