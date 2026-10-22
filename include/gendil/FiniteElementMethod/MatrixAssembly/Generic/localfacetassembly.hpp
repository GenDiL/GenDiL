// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/requiredsharedmemory.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/BoundaryFacetOperator/boundaryfacet.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   typename Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleElementSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const DomainMesh & domain_mesh,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v<Integrand>;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v<Integrand>;

   const auto& trial_fe_space = weak_form_context.template GetSpace<TrialName>();
   const auto& test_fe_space = weak_form_context.template GetSpace<TestName>();
   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index) };

   auto element_operator = [&]( const auto& dofs_in, auto& dofs_out )
   {
      auto local = BindElementDofs(integrand, weak_form_context, operator_context,
         ReadLocal<TrialName>(dofs_in), AddLocal<TestName>(dofs_out));
      std::apply( [&]( const auto &... term )
      {
         ( GenericCellIntegrandOperator( kernel_context, local, operator_context,
            element_context, term ), ... );
      }, integrand );
   };

   ForEachLocalTrialDof( kernel_context, trial_fe_space, [&] ( const auto & trial_dof )
   {
      auto x = MakeZeroElementVector( kernel_context, trial_fe_space );
      SetLocalDofOnOwnerThread( kernel_context, x, trial_dof, Real(1.0) );
      auto y = MakeZeroElementVector( kernel_context, test_fe_space );

      element_operator( x, y );

      AddSparseMatrixEntry(
         kernel_context,
         trial_fe_space,
         test_fe_space,
         element_index,
         trial_dof,
         y,
         sparse_matrix );
   });
}

template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename ElementContextType,
   typename FaceInfo,
   typename Integrands,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetSelfSparseMatrix(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const ElementContextType & element_context,
   const FaceInfo & face_info,
   const Integrands & integrands,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< Integrands >;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v< Integrands >;

   const auto & trial_fe_space = execution_context.template GetSpace< TrialName >();
   const auto & test_fe_space = execution_context.template GetSpace< TestName >();

   // Block A(e,e): current trial basis -> current test residual
   ForEachLocalTrialDof(
      kernel_context,
      trial_fe_space,
      [&] ( const auto & trial_dof )
      {
         auto x_minus = MakeZeroElementVector( kernel_context, trial_fe_space );
         auto x_plus = MakeZeroElementVector( kernel_context, trial_fe_space );
         SetLocalDofOnOwnerThread( kernel_context, x_minus, trial_dof, Real(1.0) );
         auto y_minus = MakeZeroElementVector( kernel_context, test_fe_space );

         auto current = BindElementDofs< SemanticSide::Minus >(
            integrands, execution_context, operator_context,
            ReadLocal< TrialName >( x_minus ), AddLocal< TestName >( y_minus ) );
         auto neighbor = BindElementDofs< SemanticSide::Plus >(
            integrands, execution_context, operator_context,
            ReadLocal< TrialName >( x_plus ) );
         std::apply( [&]( const auto &... term )
         {
            ( LocalInteriorFacetIntegrandOperator( kernel_context, current, neighbor,
               operator_context, element_context, face_info, term ), ... );
         }, integrands );

         AddSparseMatrixEntry(
            kernel_context,
            trial_fe_space,
            test_fe_space,
            element_context.ElementIndex(),
            trial_dof,
            y_minus,
            sparse_matrix );
      });
}

template <
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename ElementContextType,
   typename FaceInfo,
   typename Integrands,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetNeighborSparseMatrix(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const ElementContextType & element_context,
   const FaceInfo & face_info,
   const Integrands & integrands,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v< Integrands >;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v< Integrands >;

   const auto & trial_fe_space = execution_context.template GetSpace< TrialName >();
   const auto & test_fe_space = execution_context.template GetSpace< TestName >();
   const auto & plus_orientation = face_info.PlusSide().GetOrientation();

   // Block A(e,nb): plus-side trial basis -> current test residual
   ForEachLocalTrialDof(
      kernel_context,
      trial_fe_space,
      [&] ( const auto & trial_dof )
      {
         auto x_minus = MakeZeroElementVector( kernel_context, trial_fe_space );
         auto x_plus = MakeZeroElementVector( kernel_context, trial_fe_space );
         SetLocalDofOnOwnerThread( kernel_context, x_plus, trial_dof, Real(1.0) );
         auto y_minus = MakeZeroElementVector( kernel_context, test_fe_space );

         auto current = BindElementDofs< SemanticSide::Minus >(
            integrands, execution_context, operator_context,
            ReadLocal< TrialName >( x_minus ), AddLocal< TestName >( y_minus ) );
         auto neighbor = BindElementDofs< SemanticSide::Plus >(
            integrands, execution_context, operator_context,
            ReadLocal< TrialName >( x_plus ) );
         std::apply( [&]( const auto &... term )
         {
            ( LocalInteriorFacetIntegrandOperator( kernel_context, current, neighbor,
               operator_context, element_context, face_info, term ), ... );
         }, integrands );

         const auto plus_native_dof = OrientReferenceDofToNative(
            trial_fe_space, trial_dof, plus_orientation );

         AddSparseMatrixEntry(
            kernel_context,
            trial_fe_space,
            test_fe_space,
            element_context.ElementIndex(),
            face_info,
            plus_native_dof,
            y_minus,
            sparse_matrix );
      });
}

template <
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename DomainMesh,
   typename Integrands,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetSparseMatrix(
   KernelContext & kernel_context,
   const ExecutionContext & execution_context,
   const OperatorContext & operator_context,
   const DomainMesh & domain_mesh,
   const GlobalIndex & element_index,
   const Integrands & integrands,
   SparseMatrix & sparse_matrix )
{
   ElementContext element_context{
      element_index,
      domain_mesh.GetCell( element_index )
   };

   InteriorFaceLoop(
      domain_mesh,
      element_index,
      [&] ( const auto & face_info )
      {
         AssembleInteriorFacetSelfSparseMatrix(
            kernel_context,
            execution_context,
            operator_context,
            element_context,
            face_info,
            integrands,
            sparse_matrix );
         if constexpr ( Mode == SparseAssemblyMode::Full )
         {
            AssembleInteriorFacetNeighborSparseMatrix(
               kernel_context,
               execution_context,
               operator_context,
               element_context,
               face_info,
               integrands,
               sparse_matrix );
         }
      } );
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   typename Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleBoundaryFacetSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const DomainMesh & domain_mesh,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = weak_form_traits::only_trial_field_name_v<Integrand>;
   constexpr auto TestName = weak_form_traits::only_test_field_name_v<Integrand>;

   const auto& trial_fe_space =
      weak_form_context.template GetSpace<TrialName>();
   const auto& test_fe_space =
      weak_form_context.template GetSpace<TestName>();

   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index)
   };

   auto apply_boundary = [&](const auto& face, const auto& input, auto& output) {
      auto local = BindElementDofs(integrand, weak_form_context, operator_context,
         ReadLocal<TrialName>(input), AddLocal<TestName>(output));
      std::apply( [&]( const auto &... term )
      {
         ( BoundaryFacetIntegrandOperator( kernel_context, local, operator_context,
            element_context, face, term ), ... );
      }, integrand );
   };
   BoundaryFaceLoop(
      domain_mesh,
      element_index,
      [&] ( auto const & face_info )
      {
         // Compute RHS contribution (with zero input) to separate matrix and RHS terms
         auto x_zero = MakeZeroElementVector(kernel_context, trial_fe_space);
         auto y_rhs = MakeZeroElementVector(kernel_context, test_fe_space);

         apply_boundary(face_info, x_zero, y_rhs);

         // Block A(e,e): minus-side trial basis -> minus-side test residual
         ForEachLocalTrialDof( kernel_context, trial_fe_space, [&] ( const auto & trial_dof )
         {
            auto x_minus = MakeZeroElementVector(kernel_context, trial_fe_space);
            SetLocalDofOnOwnerThread( kernel_context, x_minus, trial_dof, Real(1.0) );

            auto y_minus = MakeZeroElementVector(kernel_context, test_fe_space);

            apply_boundary(face_info, x_minus, y_minus);

            // Subtract RHS contribution to get pure matrix column.
            SubtractLocalDofVector(
               kernel_context,
               test_fe_space,
               y_minus,
               y_rhs );

            AddSparseMatrixEntry(
               kernel_context,
               trial_fe_space,
               test_fe_space,
               element_index,
               trial_dof,
               y_minus,
               sparse_matrix
            );
         });
      }
   );
}

namespace details {

template <
   typename KernelPolicy,
   SparseAssemblyMode Mode = SparseAssemblyMode::Full,
   typename WeakForm,
   typename WeakFormContext,
   typename IntegrationRule,
   typename SparseMatrixType >
void AssembleLocalSparseTarget(
   const WeakForm & weak_form,
   const WeakFormContext & weak_form_context,
   const IntegrationRule & integration_rule,
   SparseMatrixType & sparse_matrix )
{
   if constexpr ( !is_zero< WeakForm > )
   {
      using Signature = form_signature_t< WeakForm >;
      using Requirements = lower_signature_requirements_t< Signature >;
      constexpr auto DomainName = local_facet_assembly_domain_name_v< WeakForm >;
      if constexpr (
         has_nonzero_integrands< WeakForm, DomainName, SemanticEntity::InteriorFacet > ||
         has_nonzero_integrands< WeakForm, DomainName, SemanticEntity::BoundaryFacet > )
      {
         // Per-cell face branches cannot share a block-wide interpolation barrier.
         static_assert(is_unbatched_operator_configuration_allowed_v<KernelPolicy>,
            "Local facet traversal requires BatchSize == 1 because synchronization spans the GPU block.");
      }
      const auto & domain_mesh = GetCellIntegrationDomainMesh( weak_form, weak_form_context );
      using DomainMesh = std::remove_cvref_t< decltype( domain_mesh ) >;
      auto selected_domain = SelectedCellExecutionDomain< DomainName, 0, DomainMesh, false >{ domain_mesh };
      auto selected_context = MakeRestrictedWeakFormContext< Requirements >(
         weak_form_context,
         selected_domain );
      auto operator_context = MakeOperatorContext(
         weak_form,
         selected_context,
         integration_rule );
      auto execution_context = execution_context_detail::BindExecutionContext<
         bound_only_requirements_t< Requirements >,
         KernelPolicy >( selected_context );

      using weak_form_execution_detail::SelectIntegrands;
      auto cell_integrands = SelectIntegrands< DomainName, SemanticEntity::Cell >( weak_form );
      auto interior_integrands = SelectIntegrands< DomainName, SemanticEntity::InteriorFacet >( weak_form );
      auto boundary_integrands = SelectIntegrands< DomainName, SemanticEntity::BoundaryFacet >( weak_form );
      using CellIntegrands = decltype( cell_integrands );
      using InteriorIntegrands = decltype( interior_integrands );
      using BoundaryIntegrands = decltype( boundary_integrands );

      using ExecutionContext = decltype( execution_context );
      using OperatorContext = decltype( operator_context );
      constexpr size_t shared_slots = Max(
         SparseAssemblyRequiredSharedMemory< KernelPolicy, CellIntegrands, ExecutionContext, OperatorContext >(),
         SparseAssemblyRequiredSharedMemory< KernelPolicy, InteriorIntegrands, ExecutionContext, OperatorContext >(),
         SparseAssemblyRequiredSharedMemory< KernelPolicy, BoundaryIntegrands, ExecutionContext, OperatorContext >() );
      using Kernel = KernelContext< KernelPolicy, shared_slots >;
      mesh::CellIterator< KernelPolicy >(
         domain_mesh,
         [=]
         GENDIL_HOST_DEVICE ( GlobalIndex element_index ) mutable
         {
            // NVCC needs unconditional uses to infer captures referenced only
            // inside if constexpr branches. Keep these capture workarounds.
            (void)execution_context;
            (void)operator_context;
            (void)sparse_matrix;
            (void)domain_mesh;
            (void)cell_integrands;
            (void)interior_integrands;
            (void)boundary_integrands;

            GENDIL_SHARED Real shared[ Kernel::shared_memory_block_size ];
            Kernel kernel( shared );
            constexpr auto DomainName = local_facet_assembly_domain_name_v< WeakForm >;
            constexpr bool HasCellIntegrands =
               has_nonzero_integrands< WeakForm, DomainName, SemanticEntity::Cell >;
            constexpr bool HasInteriorIntegrands =
               has_nonzero_integrands< WeakForm, DomainName, SemanticEntity::InteriorFacet >;
            constexpr bool HasBoundaryIntegrands =
               has_nonzero_integrands< WeakForm, DomainName, SemanticEntity::BoundaryFacet >;
            // Share traversal and probes, evaluating each explicit integrand separately.
            if constexpr ( HasCellIntegrands )
            {
               AssembleElementSparseMatrix(
                  kernel,
                  execution_context,
                  operator_context,
                  domain_mesh,
                  element_index,
                  cell_integrands,
                  sparse_matrix );
            }
            if constexpr ( HasInteriorIntegrands )
            {
               AssembleInteriorFacetSparseMatrix< Mode >(
                  kernel,
                  execution_context,
                  operator_context,
                  domain_mesh,
                  element_index,
                  interior_integrands,
                  sparse_matrix );
            }
            if constexpr ( HasBoundaryIntegrands )
            {
               AssembleBoundaryFacetSparseMatrix(
                  kernel,
                  execution_context,
                  operator_context,
                  domain_mesh,
                  element_index,
                  boundary_integrands,
                  sparse_matrix );
            }
         } );
   }
}

} // namespace details

} // namespace gendil
