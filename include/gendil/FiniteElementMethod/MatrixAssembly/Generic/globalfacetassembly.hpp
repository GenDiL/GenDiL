// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localfacetassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/LocalFacetOperator/localboundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceDomain,
   class TrialSpace,
   class TestSpace,
   class FaceInfo,
   class WeakForm,
   class Target>
GENDIL_HOST_DEVICE
void AssembleGlobalBoundaryFaceSparseMatrix(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const FaceDomain& face_domain,
   const TrialSpace& trial_space,
   const TestSpace& test_space,
   const FaceInfo& face_info,
   const WeakForm& weak_form,
   const GlobalIndex face_index,
   Target& target)
{
   const GlobalIndex element_index =
      face_info.MinusSide().GetCellIndex();
   const auto& orientation = face_info.MinusSide().GetOrientation();
   ElementContext element_context{
      element_index,
      face_domain.GetCellMesh().GetCell(element_index)};

   auto x_zero = MakeZeroElementVector(kernel, trial_space);
   auto y_rhs = MakeZeroElementVector(kernel, test_space);
   LocalBoundaryFacetIntegrandOperator(
      kernel,
      wf_ctx,
      op_ctx,
      element_context,
      face_info,
      weak_form,
      x_zero,
      y_rhs);

   ForEachLocalTrialDof(
      kernel,
      trial_space,
      [&] (const auto& reference_trial_dof)
      {
         auto x = MakeZeroElementVector(kernel, trial_space);
         SetLocalDofOnOwnerThread(
            kernel,
            x,
            reference_trial_dof,
            Real{1});
         auto y = MakeZeroElementVector(kernel, test_space);
         LocalBoundaryFacetIntegrandOperator(
            kernel,
            wf_ctx,
            op_ctx,
            element_context,
            face_info,
            weak_form,
            x,
            y);
         SubtractLocalDofVector(
            kernel,
            test_space,
            y,
            y_rhs);

         AddOrientedRawCOOEntityBlockEntries(
            kernel,
            trial_space,
            test_space,
            element_index,
            element_index,
            orientation,
            orientation,
            reference_trial_dof,
            y,
            face_index,
            target);
      });
}

template<
   class KernelContext,
   class WeakFormContext,
   class OperatorContext,
   class FaceDomain,
   class TrialMinusSpace,
   class TrialPlusSpace,
   class TestMinusSpace,
   class TestPlusSpace,
   class FaceInfo,
   class WeakForm,
   class Targets>
GENDIL_HOST_DEVICE
void AssembleGlobalInteriorFaceSparseMatrix(
   KernelContext& kernel,
   const WeakFormContext& wf_ctx,
   const OperatorContext& op_ctx,
   const FaceDomain& face_domain,
   const TrialMinusSpace& trial_minus_space,
   const TrialPlusSpace& trial_plus_space,
   const TestMinusSpace& test_minus_space,
   const TestPlusSpace& test_plus_space,
   const FaceInfo& face_info,
   const WeakForm& weak_form,
   const GlobalIndex face_index,
   Targets& targets)
{
   const GlobalIndex minus_element =
      face_info.MinusSide().GetCellIndex();
   const GlobalIndex plus_element =
      face_info.PlusSide().GetCellIndex();
   const auto& minus_orientation =
      face_info.MinusSide().GetOrientation();
   const auto& plus_orientation =
      face_info.PlusSide().GetOrientation();
   ForEachLocalTrialDof(
      kernel,
      trial_minus_space,
      [&] (const auto& reference_trial_dof)
      {
         auto x_minus = MakeZeroElementVector(kernel, trial_minus_space);
         auto x_plus = MakeZeroElementVector(kernel, trial_plus_space);
         SetLocalDofOnOwnerThread(
            kernel,
            x_minus,
            reference_trial_dof,
            Real{1});
         auto y_minus = MakeZeroElementVector(kernel, test_minus_space);
         auto y_plus = MakeZeroElementVector(kernel, test_plus_space);

         GenericCanonicalGlobalInteriorChannelAction(
            kernel,
            wf_ctx,
            op_ctx,
            face_domain,
            trial_minus_space,
            trial_plus_space,
            test_minus_space,
            test_plus_space,
            face_info,
            weak_form,
            x_minus,
            x_plus,
            y_minus,
            y_plus);

         AddOrientedRawCOOEntityBlockEntries(
            kernel,
            trial_minus_space,
            test_minus_space,
            minus_element,
            minus_element,
            minus_orientation,
            minus_orientation,
            reference_trial_dof,
            y_minus,
            face_index,
            targets.minus_minus);
         AddOrientedRawCOOEntityBlockEntries(
            kernel,
            trial_minus_space,
            test_plus_space,
            plus_element,
            minus_element,
            minus_orientation,
            plus_orientation,
            reference_trial_dof,
            y_plus,
            face_index,
            targets.plus_minus);
      });

   ForEachLocalTrialDof(
      kernel,
      trial_plus_space,
      [&] (const auto& reference_trial_dof)
      {
         auto x_minus = MakeZeroElementVector(kernel, trial_minus_space);
         auto x_plus = MakeZeroElementVector(kernel, trial_plus_space);
         SetLocalDofOnOwnerThread(
            kernel,
            x_plus,
            reference_trial_dof,
            Real{1});
         auto y_minus = MakeZeroElementVector(kernel, test_minus_space);
         auto y_plus = MakeZeroElementVector(kernel, test_plus_space);

         GenericCanonicalGlobalInteriorChannelAction(
            kernel,
            wf_ctx,
            op_ctx,
            face_domain,
            trial_minus_space,
            trial_plus_space,
            test_minus_space,
            test_plus_space,
            face_info,
            weak_form,
            x_minus,
            x_plus,
            y_minus,
            y_plus);

         AddOrientedRawCOOEntityBlockEntries(
            kernel,
            trial_plus_space,
            test_minus_space,
            minus_element,
            plus_element,
            plus_orientation,
            minus_orientation,
            reference_trial_dof,
            y_minus,
            face_index,
            targets.minus_plus);
         AddOrientedRawCOOEntityBlockEntries(
            kernel,
            trial_plus_space,
            test_plus_space,
            plus_element,
            plus_element,
            plus_orientation,
            plus_orientation,
            reference_trial_dof,
            y_plus,
            face_index,
            targets.plus_plus);
      });
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class CellTargets,
   class BoundaryTargets,
   class InteriorTargets>
void GenericPartitionRawCOOIntegrandAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   PartitionRawCOOAssemblyTargets<
      CellTargets,
      BoundaryTargets,
      InteriorTargets>& targets)
{
   constexpr auto DomainName =
      local_facet_assembly_domain_name_v<WeakForm>;
   constexpr size_t required_shared_mem =
      generic_operator_integrand_required_shared_memory_v<
         KernelPolicy,
         IntegrationRule>;
   using Context = KernelContext<KernelPolicy, required_shared_mem>;

   if constexpr (has_cell_contributions_v<WeakForm>)
   {
      ForEachCellExecutionDomain(
         wf_ctx,
         Cells<DomainName>{},
         [&] (const auto& selected_domain)
         {
            using SelectedDomain =
               std::remove_cvref_t<decltype(selected_domain)>;
            constexpr size_t Part = SelectedDomain::cell_batch_index;
            auto restricted_ctx = MakeRestrictedWeakFormContext<WeakForm>(
               wf_ctx,
               Cells<DomainName>{},
               selected_domain);
            auto op_ctx = MakeOperatorContext(
               restricted_ctx,
               integration_rule);
            const auto& domain_mesh = selected_domain.GetCellMesh();
            auto part_target = std::get<Part>(targets.cells);

            mesh::CellIterator<KernelPolicy>(
               domain_mesh,
               [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
               {
                  GENDIL_SHARED Real _shared_mem[
                     Context::shared_memory_block_size];
                  Context kernel(_shared_mem);
                  AssembleElementSparseMatrix(
                     kernel,
                     restricted_ctx,
                     op_ctx,
                     domain_mesh,
                     element_index,
                     weak_form,
                     part_target);
               });
         });
   }

   if constexpr (has_boundary_facet_contributions_v<WeakForm>)
   {
      ForEachBoundaryFaceExecutionDomain(
         wf_ctx,
         BoundaryFacets<DomainName>{},
         [&] (const auto& selected_domain)
         {
            using SelectedDomain =
               std::remove_cvref_t<decltype(selected_domain)>;
            constexpr size_t Part = SelectedDomain::face_batch_index;
            auto restricted_ctx = MakeRestrictedWeakFormContext<WeakForm>(
               wf_ctx,
               BoundaryFacets<DomainName>{},
               selected_domain);
            auto op_ctx = MakeFacetOperatorContext(
               restricted_ctx,
               integration_rule,
               selected_domain);
            constexpr auto TrialName = requirements<WeakForm>::trial_name;
            constexpr auto TestName = requirements<WeakForm>::test_name;
            const auto& trial_space = restricted_ctx
               .template fe_field<TrialName>().space
               .GetMinusFiniteElementSpace();
            const auto& test_space = restricted_ctx
               .template fe_field<TestName>().space
               .GetMinusFiniteElementSpace();
            const auto& face_mesh = selected_domain.GetFaceMesh();
            auto part_target = std::get<Part>(targets.boundaries);

            mesh::GlobalFaceIterator<KernelPolicy>(
               face_mesh,
               [=] GENDIL_HOST_DEVICE (GlobalIndex face_index) mutable
               {
                  GENDIL_SHARED Real _shared_mem[
                     Context::shared_memory_block_size];
                  Context kernel(_shared_mem);
                  const auto face_info =
                     face_mesh.GetGlobalFaceInfo(face_index);
                  AssembleGlobalBoundaryFaceSparseMatrix(
                     kernel,
                     restricted_ctx,
                     op_ctx,
                     selected_domain,
                     trial_space,
                     test_space,
                     face_info,
                     weak_form,
                     face_index,
                     part_target);
               });
         });
   }

   if constexpr (has_interior_facet_contributions_v<WeakForm>)
   {
      ForEachInteriorFaceExecutionDomain(
         wf_ctx,
         InteriorFacets<DomainName>{},
         [&] (const auto& selected_domain)
         {
            using SelectedDomain =
               std::remove_cvref_t<decltype(selected_domain)>;
            constexpr size_t Part = SelectedDomain::face_batch_index;
            auto restricted_ctx = MakeRestrictedWeakFormContext<WeakForm>(
               wf_ctx,
               InteriorFacets<DomainName>{},
               selected_domain);
            auto op_ctx = MakeFacetOperatorContext(
               restricted_ctx,
               integration_rule,
               selected_domain);
            constexpr auto TrialName = requirements<WeakForm>::trial_name;
            constexpr auto TestName = requirements<WeakForm>::test_name;
            const auto& trial_binding =
               restricted_ctx.template fe_field<TrialName>().space;
            const auto& test_binding =
               restricted_ctx.template fe_field<TestName>().space;
            const auto& trial_minus_space =
               trial_binding.GetMinusFiniteElementSpace();
            const auto& trial_plus_space =
               trial_binding.GetPlusFiniteElementSpace();
            const auto& test_minus_space =
               test_binding.GetMinusFiniteElementSpace();
            const auto& test_plus_space =
               test_binding.GetPlusFiniteElementSpace();

            ValidateNonconformingGlobalInteriorFacetTransformSupport<
               SelectedDomain,
               std::remove_cvref_t<decltype(trial_minus_space)>,
               std::remove_cvref_t<decltype(trial_plus_space)>,
               std::remove_cvref_t<decltype(test_minus_space)>,
               std::remove_cvref_t<decltype(test_plus_space)>>();

            const auto& face_mesh = selected_domain.GetFaceMesh();
            auto part_targets = std::get<Part>(targets.interiors);
            mesh::GlobalFaceIterator<KernelPolicy>(
               face_mesh,
               [=] GENDIL_HOST_DEVICE (GlobalIndex face_index) mutable
               {
                  GENDIL_SHARED Real _shared_mem[
                     Context::shared_memory_block_size];
                  Context kernel(_shared_mem);
                  const auto face_info =
                     face_mesh.GetGlobalFaceInfo(face_index);
                  AssembleGlobalInteriorFaceSparseMatrix(
                     kernel,
                     restricted_ctx,
                     op_ctx,
                     selected_domain,
                     trial_minus_space,
                     trial_plus_space,
                     test_minus_space,
                     test_plus_space,
                     face_info,
                     weak_form,
                     face_index,
                     part_targets);
               });
         });
   }
}

template<
   class KernelPolicy,
   class Integrand,
   class WeakFormContext,
   class IntegrationRule,
   class CellTargets,
   class BoundaryTargets,
   class InteriorTargets>
void GenericGlobalFacetAssembly(
   const Integrand& integrand,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   PartitionRawCOOAssemblyTargets<
      CellTargets,
      BoundaryTargets,
      InteriorTargets>& targets)
{
   GenericPartitionRawCOOIntegrandAssembly<KernelPolicy>(
      integrand,
      wf_ctx,
      integration_rule,
      targets);
}

template<
   class KernelPolicy,
   class Map,
   class WeakFormContext,
   class IntegrationRule,
   class CellTargets,
   class BoundaryTargets,
   class InteriorTargets>
void GenericGlobalFacetAssembly(
   const SumFormExpr<Map>& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   PartitionRawCOOAssemblyTargets<
      CellTargets,
      BoundaryTargets,
      InteriorTargets>& targets)
{
   std::apply(
      [&] (const auto&... entries)
      {
         (GenericPartitionRawCOOIntegrandAssembly<KernelPolicy>(
             entries.value,
             wf_ctx,
             integration_rule,
             targets), ...);
      },
      weak_form.map.entries);
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
void GenericGlobalFacetAssembly(
   const WeakForm&,
   const WeakFormContext&,
   const IntegrationRule&,
   SparseMatrixType&)
{
   static_assert(
      dependent_false_v<SparseMatrixType>,
      "Partition sparse assembly requires a PartitionRawCOOAssemblyTargets payload.");
}

} // namespace gendil
