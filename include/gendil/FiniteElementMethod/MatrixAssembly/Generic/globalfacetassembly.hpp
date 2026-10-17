// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localfacetassembly.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/domainiteration.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/GlobalFacetOperator/globalinteriorfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/BoundaryFacetOperator/boundaryfacet.hpp"
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
   class ExecutionContext,
   class OperatorContext,
   class FaceDomain,
   class FaceInfo,
   class WeakForm,
   class Target>
GENDIL_HOST_DEVICE
void AssembleGlobalBoundaryFaceSparseMatrix(
   KernelContext& kernel,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const FaceDomain& face_domain,
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

   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v<WeakForm>;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v<WeakForm>;
   const auto& trial_space = execution_context.template GetSpace<TrialName>().GetMinusFiniteElementSpace();
   const auto& test_space = execution_context.template GetSpace<TestName>().GetMinusFiniteElementSpace();
   auto apply = [&](const auto& input, auto& output) {
      auto local = BindElementDofs(weak_form, execution_context, operator_context,
         ReadLocal<TrialName>(input), AddLocal<TestName>(output));
      std::apply( [&]( const auto &... integrand )
      {
         ( BoundaryFacetIntegrandOperator( kernel, local, operator_context,
            element_context, face_info, integrand ), ... );
      }, weak_form );
   };
   auto x_zero = MakeZeroElementVector(kernel, trial_space);
   auto y_rhs = MakeZeroElementVector(kernel, test_space);
   apply(x_zero, y_rhs);

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
         apply(x, y);
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
   class ExecutionContext,
   class OperatorContext,
   class FaceDomain,
   class FaceInfo,
   class WeakForm,
   class Targets>
GENDIL_HOST_DEVICE
void AssembleGlobalInteriorFaceSparseMatrix(
   KernelContext& kernel,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const FaceDomain& face_domain,
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
   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v<WeakForm>;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v<WeakForm>;
   const auto& trial_binding = execution_context.template GetSpace<TrialName>();
   const auto& test_binding = execution_context.template GetSpace<TestName>();
   const auto& trial_minus_space = trial_binding.GetMinusFiniteElementSpace();
   const auto& trial_plus_space = trial_binding.GetPlusFiniteElementSpace();
   const auto& test_minus_space = test_binding.GetMinusFiniteElementSpace();
   const auto& test_plus_space = test_binding.GetPlusFiniteElementSpace();

   ValidateNonconformingGlobalInteriorFacetTransformSupport<
      FaceDomain,
      std::remove_cvref_t<decltype(trial_minus_space)>,
      std::remove_cvref_t<decltype(trial_plus_space)>,
      std::remove_cvref_t<decltype(test_minus_space)>,
      std::remove_cvref_t<decltype(test_plus_space)>>();

   auto apply = [&](const auto& x_minus, const auto& x_plus, auto& y_minus, auto& y_plus)
   {
      auto minus = BindElementDofs<SemanticSide::Minus>(weak_form,
         execution_context, operator_context, ReadLocal<TrialName>(x_minus), AddLocal<TestName>(y_minus));
      auto plus = BindElementDofs<SemanticSide::Plus>(weak_form,
         execution_context, operator_context, ReadLocal<TrialName>(x_plus), AddLocal<TestName>(y_plus));
      std::apply( [&]( const auto &... integrand )
      {
         ( GlobalInteriorFacetIntegrandOperator( kernel, minus, plus, operator_context,
            face_domain, face_info, integrand ), ... );
      }, weak_form );
   };
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

         apply(x_minus, x_plus, y_minus, y_plus);

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

         apply(x_minus, x_plus, y_minus, y_plus);

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

namespace details {

template<
   class KernelPolicy,
   class WeakForm,
   class ExecutionContext,
   class OperatorContext,
   class SelectedDomain,
   class CellTargets>
void AssemblePartitionCellRawCOODomain(
   const WeakForm& weak_form,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const SelectedDomain& selected_domain,
   CellTargets& cell_targets)
{
   using Domain = std::remove_cvref_t<SelectedDomain>;
   constexpr size_t Part = Domain::cell_batch_index;
   constexpr size_t required_shared_mem = SparseAssemblyRequiredSharedMemory<KernelPolicy, WeakForm,
      ExecutionContext, OperatorContext>();
   using Context = KernelContext<KernelPolicy, required_shared_mem>;
   const auto& domain_mesh = selected_domain.GetCellMesh();
   auto part_target = std::get<Part>(cell_targets);

   mesh::CellIterator<KernelPolicy>(
      domain_mesh,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         GENDIL_SHARED Real _shared_mem[
            Context::shared_memory_block_size];
         Context kernel(_shared_mem);
         AssembleElementSparseMatrix(
            kernel,
            execution_context,
            operator_context,
            domain_mesh,
            element_index,
            weak_form,
            part_target);
      });
}

template<
   class KernelPolicy,
   class WeakForm,
   class ExecutionContext,
   class OperatorContext,
   class SelectedDomain,
   class BoundaryTargets>
void AssemblePartitionBoundaryRawCOODomain(
   const WeakForm& weak_form,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const SelectedDomain& selected_domain,
   BoundaryTargets& boundary_targets)
{
   using Domain = std::remove_cvref_t<SelectedDomain>;
   constexpr size_t Part = Domain::face_batch_index;
   constexpr size_t required_shared_mem = SparseAssemblyRequiredSharedMemory<KernelPolicy, WeakForm,
      ExecutionContext, OperatorContext>();
   using Context = KernelContext<KernelPolicy, required_shared_mem>;
   const auto& face_mesh = selected_domain.GetFaceMesh();
   auto part_target = std::get<Part>(boundary_targets);

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
            execution_context,
            operator_context,
            selected_domain,
            face_info,
            weak_form,
            face_index,
            part_target);
      });
}

template<
   class KernelPolicy,
   class WeakForm,
   class ExecutionContext,
   class OperatorContext,
   class SelectedDomain,
   class InteriorTargets>
void AssemblePartitionInteriorRawCOODomain(
   const WeakForm& weak_form,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const SelectedDomain& selected_domain,
   InteriorTargets& interior_targets)
{
   using Domain = std::remove_cvref_t<SelectedDomain>;
   constexpr size_t Part = Domain::face_batch_index;
   constexpr size_t required_shared_mem = SparseAssemblyRequiredSharedMemory<KernelPolicy, WeakForm,
      ExecutionContext, OperatorContext>();
   using Context = KernelContext<KernelPolicy, required_shared_mem>;
   const auto& face_mesh = selected_domain.GetFaceMesh();
   auto part_targets = std::get<Part>(interior_targets);
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
            execution_context,
            operator_context,
            selected_domain,
            face_info,
            weak_form,
            face_index,
            part_targets);
      });
}

template <
   class KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class CellTargets,
   class BoundaryTargets,
   class InteriorTargets >
void AssemblePartitionSparseTarget(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   PartitionRawCOOAssemblyTargets<
      CellTargets,
      BoundaryTargets,
      InteriorTargets > & targets )
{
   ValidateSparseAssemblyDomainSupport<
      MatrixAssemblyType::RawCOO,
      Mode,
      WeakForm,
      WeakFormContext >();

   if constexpr ( Mode == SparseAssemblyMode::Full && !is_zero< WeakForm > )
   {
      using FormSignature = form_signature_t< WeakForm >;
      constexpr auto DomainName = local_facet_assembly_domain_name_v< WeakForm >;
      auto assemble_phase = [&]< SemanticEntity Entity >()
      {
         if constexpr ( has_nonzero_integrands< WeakForm, DomainName, Entity > )
         {
            using Signatures = typename weak_form_traits::domain_signatures<
               typename FormSignature::integrands, DomainName, Entity >::type;
            auto integrands = weak_form_execution_detail::SelectIntegrands< DomainName, Entity >( weak_form );
            using Requirements = typename weak_form_execution_detail::signature_requirements< Signatures >::type;
            auto assemble_batch = [&]( const auto & batch )
            {
               auto selected_context = MakeRestrictedWeakFormContext< Requirements >(
                  weak_form_context,
                  batch );
               auto operator_context = [&]
               {
                  if constexpr ( Entity == SemanticEntity::Cell )
                  {
                     return MakeOperatorContext< FormSignature >(
                        integrands,
                        selected_context,
                        integration_rule );
                  }
                  else
                  {
                     return MakeFacetOperatorContext< FormSignature >(
                        integrands,
                        selected_context,
                        integration_rule,
                        batch );
                  }
               }();
               auto execution_context = execution_context_detail::BindExecutionContext<
                  bound_only_requirements_t< Requirements >,
                  KernelPolicy >( selected_context );

               if constexpr ( Entity == SemanticEntity::Cell )
               {
                  AssemblePartitionCellRawCOODomain< KernelPolicy >(
                     integrands,
                     execution_context,
                     operator_context,
                     batch,
                     targets.cells );
               }
               else if constexpr ( Entity == SemanticEntity::BoundaryFacet )
               {
                  AssemblePartitionBoundaryRawCOODomain< KernelPolicy >(
                     integrands,
                     execution_context,
                     operator_context,
                     batch,
                     targets.boundaries );
               }
               else
               {
                  AssemblePartitionInteriorRawCOODomain< KernelPolicy >(
                     integrands,
                     execution_context,
                     operator_context,
                     batch,
                     targets.interiors );
               }
            };

            if constexpr ( Entity == SemanticEntity::Cell )
            {
               ForEachCellExecutionDomain(
                  weak_form_context,
                  Cells< DomainName >{},
                  assemble_batch );
            }
            else if constexpr ( Entity == SemanticEntity::BoundaryFacet )
            {
               ForEachBoundaryFaceExecutionDomain(
                  weak_form_context,
                  BoundaryFacets< DomainName >{},
                  assemble_batch );
            }
            else
            {
               ForEachInteriorFaceExecutionDomain(
                  weak_form_context,
                  InteriorFacets< DomainName >{},
                  assemble_batch );
            }
         }
      };
      // Match local assembly: entity phases, then source order within each phase.
      assemble_phase.template operator()< SemanticEntity::Cell >();
      assemble_phase.template operator()< SemanticEntity::InteriorFacet >();
      assemble_phase.template operator()< SemanticEntity::BoundaryFacet >();
   }
}

template <
   class KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType >
void AssemblePartitionSparseTarget(
   const WeakForm &,
   const WeakFormContext &,
   const IntegrationRule &,
   SparseMatrixType & )
{
   ValidateSparseAssemblyDomainSupport<
      MatrixAssemblyType::RawCOO,
      Mode,
      WeakForm,
      WeakFormContext >();

   if constexpr ( Mode == SparseAssemblyMode::Full )
   {
      static_assert(
         dependent_false_v< SparseMatrixType >,
         "Partition sparse assembly requires a PartitionRawCOOAssemblyTargets payload." );
   }
}

} // namespace details

} // namespace gendil
