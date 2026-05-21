// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/bsrassembly.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

namespace gendil {

template < typename KernelContext, typename FE_Space >
GENDIL_HOST_DEVICE
auto MakeZeroElementVector(const KernelContext& kernel_context, const FE_Space& /*fe_space*/)
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
   FormExpr Integrand,
   typename SparseMatrixType >
GENDIL_HOST_DEVICE
void AssembleElementSparseMatrix(
   const KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrixType & sparse_matrix )
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName = requirements<Integrand>::test_name;

   const auto& trial_fe_space = weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space = weak_form_context.template fe_field<TestName>().space;
   ElementContext element_context{ element_index, trial_fe_space.GetCell(element_index) };

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
   FormExpr Integrand,
   typename SparseMatrixType >
GENDIL_HOST_DEVICE
void AssembleInteriorFacetSparseMatrix(
   const KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrixType & sparse_matrix )
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_fe_space =
      weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space =
      weak_form_context.template fe_field<TestName>().space;

   ElementContext element_context{
      element_index,
      trial_fe_space.GetCell(element_index)
   };

   InteriorFaceLoop(
      trial_fe_space,
      element_index,
      [&] ( auto const & face_info )
      {
         const GlobalIndex plus_element_index = face_info.PlusSide().GetCellIndex();

         // Block A(e,e): minus-side trial basis -> minus-side test residual
         ForEachLocalTrialDof( kernel_context, trial_fe_space, [&] ( const auto & trial_dof )
         {
            auto x_minus = MakeZeroElementVector(kernel_context, trial_fe_space);
            auto x_plus  = MakeZeroElementVector(kernel_context, trial_fe_space);
            SetLocalDofOnOwnerThread( kernel_context, x_minus, trial_dof, Real(1.0) );

            auto y_minus = MakeZeroElementVector(kernel_context, test_fe_space);

            // GenericInteriorFacetWeakFormOperator(
            GenericInteriorFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               x_minus,
               x_plus,
               y_minus
            );

            AddSparseMatrixEntry(
               kernel_context,
               trial_fe_space,
               test_fe_space,
               element_index,
               element_index,
               trial_dof,
               y_minus,
               sparse_matrix
            );
         });

         // Block A(e,nb): plus-side trial basis -> minus-side test residual
         ForEachLocalTrialDof( kernel_context, trial_fe_space, [&] ( const auto & trial_dof )
         {
            auto x_minus = MakeZeroElementVector(kernel_context, trial_fe_space);
            auto x_plus  = MakeZeroElementVector(kernel_context, trial_fe_space);
            SetLocalDofOnOwnerThread( kernel_context, x_plus, trial_dof, Real(1.0) );

            auto y_minus = MakeZeroElementVector(kernel_context, test_fe_space);

            // GenericInteriorFacetWeakFormOperator(
            GenericInteriorFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               x_minus,
               x_plus,
               y_minus
            );

            const auto plus_native_dof =
               OrientReferenceDofToNative(
                  trial_fe_space,
                  trial_dof,
                  face_info.PlusSide().orientation );

            AddSparseMatrixEntry(
               kernel_context,
               trial_fe_space,
               test_fe_space,
               element_index,
               plus_element_index,
               plus_native_dof,
               y_minus,
               sparse_matrix
            );
         });
      } );
}

template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   FormExpr Integrand,
   typename SparseMatrixType >
GENDIL_HOST_DEVICE
void AssembleBoundaryFacetSparseMatrix(
   const KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const GlobalIndex & element_index,
   const Integrand & integrand,
   SparseMatrixType & sparse_matrix )
{
   constexpr auto TrialName = requirements<Integrand>::trial_name;
   constexpr auto TestName  = requirements<Integrand>::test_name;

   const auto& trial_fe_space =
      weak_form_context.template fe_field<TrialName>().space;
   const auto& test_fe_space =
      weak_form_context.template fe_field<TestName>().space;

   ElementContext element_context{
      element_index,
      trial_fe_space.GetCell(element_index)
   };

   BoundaryFaceLoop(
      trial_fe_space,
      element_index,
      [&] ( auto const & face_info )
      {
         // Compute RHS contribution (with zero input) to separate matrix and RHS terms
         auto x_zero = MakeZeroElementVector(kernel_context, trial_fe_space);
         auto y_rhs = MakeZeroElementVector(kernel_context, test_fe_space);

         // GenericBoundaryFacetWeakFormOperator(
         GenericBoundaryFacetIntegrandOperator(
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

            // GenericBoundaryFacetWeakFormOperator(
            GenericBoundaryFacetIntegrandOperator(
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
               element_index,  // Diagonal block only
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
   class SparseMatrixType >
void GenericBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType & sparse_matrix)
{
   auto op_ctx = MakeOperatorContext(wf_ctx, integration_rule);
      
   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericExplicitOperator: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericExplicitOperator: missing TestSpace in integrand.");

   // FE spaces come from wf_ctx via MakeTrialField/MakeTestField
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;

   // Shared memory requirement: for now, bind to the integration rule used by this operator
   constexpr size_t required_shared_mem = required_shared_memory_v<KernelPolicy, IntegrationRule>;

   mesh::CellIterator<KernelPolicy>(
      trial_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];
         KernelContext<KernelPolicy, required_shared_mem> kernel_ctx(_shared_mem);

         AssembleElementSparseMatrix(
            kernel_ctx,
            wf_ctx,
            op_ctx,
            element_index,
            weak_form,
            sparse_matrix
         );
      }
   );
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericBlockDiagonalAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   auto bsr_matrix = MakeBlockDiagonalDGBSRPattern( trial_space );

   GenericBlockDiagonalAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   return bsr_matrix;
}

template<
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   class SparseMatrixType >
void GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   SparseMatrixType & sparse_matrix)
{
   auto op_ctx = MakeOperatorContext(wf_ctx, integration_rule);

   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericAssembly: missing TrialSpace in integrand.");
   static_assert(TestName  != StaticString{"Error"}, "GenericAssembly: missing TestSpace in integrand.");

   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;

   constexpr size_t required_shared_mem =
      required_shared_memory_v<KernelPolicy, IntegrationRule>;

   mesh::CellIterator<KernelPolicy>(
      trial_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         GENDIL_SHARED Real _shared_mem[required_shared_mem];
         KernelContext<KernelPolicy, required_shared_mem> kernel_ctx(_shared_mem);


         if constexpr (has_cell_contributions_v<WeakForm>)
         {
            AssembleElementSparseMatrix(
               kernel_ctx,
               wf_ctx,
               op_ctx,
               element_index,
               weak_form,
               sparse_matrix
            );
         }

         if constexpr (has_interior_facet_contributions_v<WeakForm>)
         {
            AssembleInteriorFacetSparseMatrix(
               kernel_ctx,
               wf_ctx,
               op_ctx,
               element_index,
               weak_form,
               sparse_matrix
            );
         }

         if constexpr (has_boundary_facet_contributions_v<WeakForm>)
         {
            AssembleBoundaryFacetSparseMatrix(
               kernel_ctx,
               wf_ctx,
               op_ctx,
               element_index,
               weak_form,
               sparse_matrix
            );
         }
      }
   );
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
   constexpr auto TrialName = requirements<WeakForm>::trial_name;
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   auto bsr_matrix = MakeDGBSRPattern( trial_space );

   GenericAssembly<KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      bsr_matrix
   );

   #ifdef GENDIL_USE_DEVICE
      // Copy assembled matrix from device to host before returning
      ToHost(bsr_matrix.num_blocks * bsr_matrix.block_rows * bsr_matrix.block_cols, bsr_matrix.values);
   #endif

   return bsr_matrix;
}

}
