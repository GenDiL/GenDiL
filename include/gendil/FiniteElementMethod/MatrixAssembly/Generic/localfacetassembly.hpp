// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/sparseassemblyvalidation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/Utilities/KernelContext/batchingeligibility.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template < typename KernelContext, typename FE_Space >
GENDIL_HOST_DEVICE
auto MakeZeroElementVector(
   const KernelContext& kernel_context,
   const FE_Space& /*fe_space*/ )
{
   using FE = typename std::remove_cvref_t<FE_Space>::finite_element_type;
   using ShapeFunctions = typename FE::shape_functions;

   if constexpr ( is_vector_shape_functions_v< ShapeFunctions > )
   {
      constexpr Integer v_dim = ShapeFunctions::vector_dim;
      using dof_shape = typename ShapeFunctions::dof_shape;
      return MakeVectorDofs( kernel_context, dof_shape{}, std::make_index_sequence< v_dim >{} );
   }
   else
   {
      using DofShape = orders_to_num_dofs< typename ShapeFunctions::orders >;
      using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

      return MakeSerialRecursiveArray< Real >( rshape{} );
   }
}

template <typename LHS, typename RHS>
GENDIL_HOST_DEVICE
constexpr bool AreEqual(const LHS& lhs, const RHS& rhs)
{
   return lhs == rhs;
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   FormExpr Integrand,
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
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName = requirements<Integrand>::test_name;

   const auto& trial_fe_space = weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space = weak_form_context.template fe_field<TestName>().space;
   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index) };

   auto element_operator = [&]( const auto& dofs_in, auto& dofs_out )
   {
      GenericCellIntegrandOperator(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         integrand,
         dofs_in,
         dofs_out
      );
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
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContextType,
   typename FaceInfo,
   FormExpr Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetSelfSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContextType & element_context,
   const GlobalIndex & element_index,
   const FaceInfo & face_info,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_fe_space =
      weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space =
      weak_form_context.template fe_field<TestName>().space;
   // Block A(e,e): minus-side trial basis -> minus-side test residual
   ForEachLocalTrialDof(
      kernel_context,
      trial_fe_space,
      [&] ( const auto & trial_dof )
      {
         auto x_minus =
            MakeZeroElementVector(kernel_context, trial_fe_space);
         auto x_plus =
            MakeZeroElementVector(kernel_context, trial_fe_space);
         SetLocalDofOnOwnerThread(
            kernel_context,
            x_minus,
            trial_dof,
            Real(1.0));

         auto y_minus =
            MakeZeroElementVector(kernel_context, test_fe_space);

         LocalInteriorFacetIntegrandOperator(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            face_info,
            integrand,
            x_minus,
            x_plus,
            y_minus);

         AddSparseMatrixEntry(
            kernel_context,
            trial_fe_space,
            test_fe_space,
            element_index,
            trial_dof,
            y_minus,
            sparse_matrix);
      });
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContextType,
   typename FaceInfo,
   FormExpr Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetNeighborSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContextType & element_context,
   const GlobalIndex & element_index,
   const FaceInfo & face_info,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName = requirements<Integrand>::test_name;

   const auto& trial_fe_space =
      weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space =
      weak_form_context.template fe_field<TestName>().space;
   const auto& plus_orientation =
      face_info.PlusSide().GetOrientation();

   // Block A(e,nb): plus-side trial basis -> minus-side test residual
   ForEachLocalTrialDof(
      kernel_context,
      trial_fe_space,
      [&] ( const auto & trial_dof )
      {
         auto x_minus =
            MakeZeroElementVector(kernel_context, trial_fe_space);
         auto x_plus =
            MakeZeroElementVector(kernel_context, trial_fe_space);
         SetLocalDofOnOwnerThread(
            kernel_context,
            x_plus,
            trial_dof,
            Real(1.0));

         auto y_minus =
            MakeZeroElementVector(kernel_context, test_fe_space);

         LocalInteriorFacetIntegrandOperator(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            face_info,
            integrand,
            x_minus,
            x_plus,
            y_minus);

         const auto plus_native_dof =
            OrientReferenceDofToNative(
               trial_fe_space,
               trial_dof,
               plus_orientation);

         AddSparseMatrixEntry(
            kernel_context,
            trial_fe_space,
            test_fe_space,
            element_index,
            face_info,
            plus_native_dof,
            y_minus,
            sparse_matrix);
      });
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   FormExpr Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const DomainMesh & domain_mesh,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index)
   };

   InteriorFaceLoop(
      domain_mesh,
      element_index,
      [&] ( auto const & face_info )
      {
         AssembleInteriorFacetSelfSparseMatrix(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            element_index,
            face_info,
            integrand,
            sparse_matrix);
         AssembleInteriorFacetNeighborSparseMatrix(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            element_index,
            face_info,
            integrand,
            sparse_matrix);
      });
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   FormExpr Integrand,
   typename SparseMatrix >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetElementBlockDiagonalSparseMatrix(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const DomainMesh & domain_mesh,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrix & sparse_matrix )
{
   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index)
   };

   InteriorFaceLoop(
      domain_mesh,
      element_index,
      [&] ( auto const & face_info )
      {
         AssembleInteriorFacetSelfSparseMatrix(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            element_index,
            face_info,
            integrand,
            sparse_matrix);
      });
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename DomainMesh,
   FormExpr Integrand,
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
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_fe_space =
      weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space =
      weak_form_context.template fe_field<TestName>().space;

   ElementContext element_context{
      element_index,
      domain_mesh.GetCell(element_index)
   };

   BoundaryFaceLoop(
      domain_mesh,
      element_index,
      [&] ( auto const & face_info )
      {
         // Compute RHS contribution (with zero input) to separate matrix and RHS terms
         auto x_zero = MakeZeroElementVector(kernel_context, trial_fe_space);
         auto y_rhs = MakeZeroElementVector(kernel_context, test_fe_space);

         LocalBoundaryFacetIntegrandOperator(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            face_info,
            integrand,
            x_zero,
            y_rhs
         );

         // Block A(e,e): minus-side trial basis -> minus-side test residual
         ForEachLocalTrialDof( kernel_context, trial_fe_space, [&] ( const auto & trial_dof )
         {
            auto x_minus = MakeZeroElementVector(kernel_context, trial_fe_space);
            SetLocalDofOnOwnerThread( kernel_context, x_minus, trial_dof, Real(1.0) );

            auto y_minus = MakeZeroElementVector(kernel_context, test_fe_space);

            LocalBoundaryFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               x_minus,
               y_minus
            );

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

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
void GenericLocalFacetAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType& sparse_matrix)
{
   details::ValidateSparseAssemblyExecutionInputs<
      details::SparseAssemblyMode::Full,
      KernelPolicy>(weak_form, wf_ctx);

   constexpr auto DomainName =
      local_facet_assembly_domain_name_v<WeakForm>;
   const auto& domain_mesh =
      GetCellIntegrationDomainMesh(weak_form, wf_ctx);
   using DomainMesh =
      std::remove_cvref_t<decltype(domain_mesh)>;
   auto selected_domain =
      SelectedCellExecutionDomain<
         DomainName,
         0,
         DomainMesh,
         false>{ domain_mesh };

   auto restricted_ctx =
      MakeRestrictedWeakFormContext<WeakForm>(
         wf_ctx,
         Cells<DomainName>{},
         selected_domain);
   auto op_ctx =
      MakeOperatorContext(restricted_ctx, integration_rule);

   constexpr size_t required_shared_mem =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;
   using Context = KernelContext<KernelPolicy, required_shared_mem>;

   mesh::CellIterator<KernelPolicy>(
      domain_mesh,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)restricted_ctx;
         (void)op_ctx;
         (void)weak_form;
         (void)sparse_matrix;
         (void)domain_mesh;

         GENDIL_SHARED Real _shared_mem[Context::shared_memory_block_size];
         Context kernel_ctx(_shared_mem);

         if constexpr (has_cell_contributions_v<WeakForm>)
         {
            AssembleElementSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }

         if constexpr (
            has_interior_facet_contributions_v<WeakForm>)
         {
            AssembleInteriorFacetSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }

         if constexpr (
            has_boundary_facet_contributions_v<WeakForm>)
         {
            AssembleBoundaryFacetSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }
      });
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType>
void AssembleElementBlockDiagonalSparseTarget(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType& sparse_matrix)
{
   details::ValidateSparseAssemblyExecutionInputs<
      details::SparseAssemblyMode::ElementBlockDiagonal,
      KernelPolicy>(weak_form, wf_ctx);

   constexpr auto DomainName =
      local_facet_assembly_domain_name_v<WeakForm>;
   const auto& domain_mesh =
      GetCellIntegrationDomainMesh(weak_form, wf_ctx);
   using DomainMesh =
      std::remove_cvref_t<decltype(domain_mesh)>;
   auto selected_domain =
      SelectedCellExecutionDomain<
         DomainName,
         0,
         DomainMesh,
         false>{ domain_mesh };

   auto restricted_ctx =
      MakeRestrictedWeakFormContext<WeakForm>(
         wf_ctx,
         Cells<DomainName>{},
         selected_domain);
   auto op_ctx =
      MakeOperatorContext(restricted_ctx, integration_rule);

   constexpr size_t required_shared_mem =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;
   using Context = KernelContext<KernelPolicy, required_shared_mem>;

   mesh::CellIterator<KernelPolicy>(
      domain_mesh,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         (void)restricted_ctx;
         (void)op_ctx;
         (void)weak_form;
         (void)sparse_matrix;
         (void)domain_mesh;

         GENDIL_SHARED Real _shared_mem[Context::shared_memory_block_size];
         Context kernel_ctx(_shared_mem);

         if constexpr (has_cell_contributions_v<WeakForm>)
         {
            AssembleElementSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }

         if constexpr (
            has_interior_facet_contributions_v<WeakForm>)
         {
            AssembleInteriorFacetElementBlockDiagonalSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }

         if constexpr (
            has_boundary_facet_contributions_v<WeakForm>)
         {
            AssembleBoundaryFacetSparseMatrix(
               kernel_ctx,
               restricted_ctx,
               op_ctx,
               domain_mesh,
               element_index,
               weak_form,
               sparse_matrix);
         }
      });
}

} // namespace gendil
