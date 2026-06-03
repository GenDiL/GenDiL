// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sgbsrmatrix.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/genericoperator.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/readdofs.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/bsrpattern.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/BSR/localinsertion.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/matrixassemblytype.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/SGBSR/sgbsrgatherscatter.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/KernelContext/kernelcontext.hpp"
#include "gendil/Utilities/KernelContext/kernelcontexttraits.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

template<class WeakForm>
consteval void ValidateSparseLinearAssemblyCoefficientInputs()
{
   static_assert(
      !has_active_trial_coefficient_dependency_v<WeakForm>,
      "Coefficient expression depends on active trial field during sparse "
      "linear assembly. This is nonlinear or ambiguous. Use a supplied frozen "
      "field with a distinct name, for example \"u_lagged\".");
}

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
   KernelContext & kernel_context,
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
   KernelContext & kernel_context,
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
   KernelContext & kernel_context,
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
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( KernelPolicy );

   auto op_ctx = MakeOperatorContext(wf_ctx, integration_rule);
      
   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();

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
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( KernelPolicy );

   auto op_ctx = MakeOperatorContext(wf_ctx, integration_rule);

   using I = std::remove_cvref_t<WeakForm>;
   ValidateSparseLinearAssemblyCoefficientInputs<I>();

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
         // CUDA fix
         (void)wf_ctx;
         (void)op_ctx;
         (void)weak_form;
         (void)sparse_matrix;

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
auto GenericBSRAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
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

template < typename FESpace >
struct IsScalarDGL2Space
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;

   static constexpr bool value =
      std::is_same_v< typename Space::restriction_type, L2Restriction > &&
      !is_vector_shape_functions_v< ShapeFunctions >;
};

template < typename TrialSpace, typename TestSpace >
inline constexpr MatrixAssemblyType default_matrix_assembly_type_v =
   ( IsScalarDGL2Space< TrialSpace >::value &&
     IsScalarDGL2Space< TestSpace >::value )
      ? MatrixAssemblyType::BSR
      : MatrixAssemblyType::SGBSR;


template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule,
   typename Backend >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule,
   Backend backend)
{
   if constexpr ( Type == MatrixAssemblyType::BSR )
   {
      return GenericBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else if constexpr ( Type == MatrixAssemblyType::SGBSR )
   {
      return GenericSGBSRAssembly<KernelPolicy>(
         weak_form,
         wf_ctx,
         integration_rule,
         backend );
   }
   else
   {
      static_assert(
         dependent_false_value_v< Type >,
         "GenericAssembly: COO, CSR, and CSC assembly are reserved but not implemented yet." );
   }
}

template<
   MatrixAssemblyType Type,
   class KernelPolicy,
   class WeakForm,
   class WeakFormContext,
   class IntegrationRule >
auto GenericAssembly(
   const WeakForm& weak_form,
   const WeakFormContext& wf_ctx,
   const IntegrationRule& integration_rule)
{
   return GenericAssembly<Type, KernelPolicy>(
      weak_form,
      wf_ctx,
      integration_rule,
      DefaultBSRBackend{} );
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
   return GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
      weak_form,
      wf_ctx,
      integration_rule );
}

}
