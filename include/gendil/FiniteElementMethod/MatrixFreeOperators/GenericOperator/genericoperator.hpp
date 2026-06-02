// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/operatorcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/elementcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/facetcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/quadraturepointcontext.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatefields.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/quadraturemeasure.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writechannelcontributions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctions.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionschannels.hpp"

namespace gendil {

// ============================================================================
// Pullback-based cell integrand operator
// ============================================================================

/**
 * @brief Cell integrand operator using pullback channels.
 *
 * This function implements the pullback-based evaluation path for cell integrands.
 *
 * **Scope: Cell integrands only**
 * - integrate(cells, u*v) - value only
 * - integrate(cells, dot(grad(u), grad(v))) - gradient only
 * - integrate(cells, u*v + dot(grad(u), grad(v))) - mixed
 *
 * @tparam KernelContext Kernel execution context
 * @tparam WeakFormContext Weak-form context (coefficients, fields, domains)
 * @tparam OperatorContext Operator context (integration rule, quad data)
 * @tparam ElementContext Element context (geometry, DoF mapping)
 * @tparam Integrand Integrand type (Integrand<Cells<...>, Expr>)
 * @tparam ElementDofsIn Input DoF container type
 * @tparam ElementDofsOut Output DoF container type
 *
 * @param kernel_context Kernel execution context
 * @param weak_form_context Weak-form context
 * @param operator_context Operator context
 * @param element_context Element context
 * @param integrand Cell integrand
 * @param dofs_in Input DoFs (trial field)
 * @param dofs_out Output DoFs (test field, accumulated with Add)
 */
template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut >
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperator(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContext & element_context,
   const Integrand & integrand,
   const ElementDofsIn & dofs_in,
   ElementDofsOut & dofs_out )
{
   if constexpr ( CellIntegrand<Integrand> )
   {
      // 1. Interpolate trial fields
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         integrand,
         dofs_in);

      // 2. Get integration rule
      const auto& integration_rule = operator_context.integration_rule();

      // 3. Allocate quadrature container
      auto Du = MakeQuadraturePointContainerFromIntegrand(
         kernel_context,
         weak_form_context,
         integrand,
         integration_rule);

      // 4. Quadrature loop
      QuadraturePointLoop(
         kernel_context,
         integration_rule,
         [&] (const auto& quad_index)
         {
            auto quad_pt_context = MakeQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               integrand,
               quad_index);

            ComputeContributions(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               quad_pt_context,
               fields,
               integrand,
               quad_index,
               Du);
         });

      // 5. Apply test functions
      ApplyAddTestFunctions(
         kernel_context,
         operator_context,
         integrand,
         Du,
         dofs_out);
   }
}

// =============================================================================
// Cell SumFormExpr Dispatch
// =============================================================================
//
// Dispatches SumFormExpr to multiple cell integrands.
//
template <
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename Map,
   typename ElementDofsIn,
   typename ElementDofsOut >
GENDIL_HOST_DEVICE
void GenericCellIntegrandOperator(
   KernelContext & kernel_context,
   const WeakFormContext & weak_form_context,
   const OperatorContext & operator_context,
   const ElementContext & element_context,
   const SumFormExpr<Map> & sum_integrand,
   const ElementDofsIn & dofs_in,
   ElementDofsOut & dofs_out )
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            GenericCellIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               entries.value,
               dofs_in,
               dofs_out
            ),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

// =============================================================================
// Interior Facet Pullback Operator
// =============================================================================
//
// Computes contributions from integrate(interior_facets, expr) using pullback channels.
//
// Pattern: Lower → Interpolate → Quadrature Loop → WriteChannelContributions → Apply Test Functions
//
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename GlobalFaceInfo,
   typename Integrand,
   typename MinusElementDofsIn,
   typename PlusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void GenericInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const GlobalFaceInfo& face_info,
   const Integrand& integrand,
   const MinusElementDofsIn& minus_dofs_in,
   const PlusElementDofsIn& plus_dofs_in,
   MinusElementDofsOut& minus_dofs_out)
{
   if constexpr (InteriorFacetIntegrand<Integrand>)
   {
      // 1. Interpolate trial fields on interior facet (both minus and plus sides)
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         minus_dofs_in,
         plus_dofs_in);

      auto face_context = MakeInteriorFacetContext(weak_form_context, integrand, face_info);

      // 2. Get face integration rule
      // Uses MinusSide() for current element's face
      const auto minus_side = face_info.MinusSide();
      auto face_integration_rule =
         GetFaceIntegrationRule(minus_side, operator_context.facet_integration_rules());

      // 3. Allocate quadrature container
      auto Du = MakeQuadraturePointContainerFromIntegrand(
         kernel_context,
         weak_form_context,
         integrand,
         face_integration_rule);

      // 4. Quadrature loop: evaluate channels and write contributions
      QuadraturePointLoop(
         kernel_context,
         face_integration_rule,
         [&] (const auto& quad_index)
         {
            // Use minus_side which satisfies CellFaceView concept
            // GetReferenceNormal() comes from minus_side
            auto facet_quad_pt_context = MakeFacetQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_context,
               integrand,
               quad_index);

            ComputeContributions(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               facet_quad_pt_context,
               fields,
               integrand,
               quad_index,
               Du);
         }
      );

      // 5. Apply test functions
      ApplyAddTestFunctions(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         Du,
         minus_dofs_out);
   }
}

// =============================================================================
// Interior Facet SumFormExpr Dispatch
// =============================================================================
//
// Dispatches SumFormExpr to multiple interior facet integrands.
//
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename GlobalFaceInfo,
   typename Map,
   typename MinusElementDofsIn,
   typename PlusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void GenericInteriorFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const GlobalFaceInfo& face_info,
   const SumFormExpr<Map>& sum_integrand,
   const MinusElementDofsIn& minus_dofs_in,
   const PlusElementDofsIn& plus_dofs_in,
   MinusElementDofsOut& minus_dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            GenericInteriorFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               entries.value,
               minus_dofs_in,
               plus_dofs_in,
               minus_dofs_out
            ),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

// Interior facet operator wrapper (calls loop over interior faces)
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename TrialSpace,
   typename Integrand,
   typename DofsInView,
   typename MinusElementDofsIn,
   typename MinusElementDofsOut>
GENDIL_HOST_DEVICE
void GenericInteriorFacetOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const Integrand& integrand,
   const DofsInView& dofs_in,              // GLOBAL view for plus-side reading
   const MinusElementDofsIn& minus_dofs_in, // Already read minus-side
   MinusElementDofsOut& minus_dofs_out)
{
   // Loop over interior faces of current element
   InteriorFaceLoop(
      trial_space,
      element_context.element_index,
      [&] (auto const & face_info)
      {
         // Read plus-side DOFs inside face loop (neighbor element)
         auto plus_dofs_in = ReadDofs(
            kernel_context,
            trial_space,
            face_info.PlusSide(),
            dofs_in);

         // Call interior facet integrand operator with both minus/plus DOFs
         GenericInteriorFacetIntegrandOperator(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            face_info,
            integrand,
            minus_dofs_in,
            plus_dofs_in,
            minus_dofs_out);
      }
   );
}

// =============================================================================
// Boundary Facet Pullback Operator
// =============================================================================
//
// Computes contributions from integrate(boundary_facets, expr) using pullback channels.
// Supports ValueChannel and GradientChannel.
//
// Pattern: Lower → Interpolate → Quadrature Loop → ComputeContributions → Apply Test Functions
//
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceInfo,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void GenericBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const FaceInfo& face_info,
   const Integrand& integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   if constexpr (BoundaryFacetIntegrand<Integrand>)
   {
      // 1. Interpolate trial fields on boundary facet (minus side only)
      auto fields = InterpolateFields(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         dofs_in);

      // 2. Get face integration rule
      const auto minus_side = face_info.MinusSide();
      auto face_integration_rule =
         GetFaceIntegrationRule(
            minus_side,
            operator_context.facet_integration_rules());

      // 3. Allocate quadrature container
      auto Du = MakeQuadraturePointContainerFromIntegrand(
         kernel_context,
         weak_form_context,
         integrand,
         face_integration_rule);

      // 4. Quadrature loop: evaluate channels and write contributions
      QuadraturePointLoop(
         kernel_context,
         face_integration_rule,
         [&] (const auto& quad_index)
         {
            // Create FACET context (NOT base QuadraturePointContext)
            auto facet_quad_pt_context = MakeFacetQuadraturePointContext(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               integrand,
               quad_index);

            ComputeContributions(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               facet_quad_pt_context,
               fields,
               integrand,
               quad_index,
               Du);
         });

      // 5. Apply test functions
      ApplyAddTestFunctions(
         kernel_context,
         weak_form_context,
         operator_context,
         face_info,
         integrand,
         Du,
         dofs_out);
   }
}

// =============================================================================
// Boundary Facet SumFormExpr Dispatch
// =============================================================================
//
// Dispatches SumFormExpr to multiple boundary facet integrands.
//
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename FaceInfo,
   typename Map,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void GenericBoundaryFacetIntegrandOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const FaceInfo& face_info,
   const SumFormExpr<Map>& sum_integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   std::apply(
      [&] (auto const & ... entries)
      {
         (
            GenericBoundaryFacetIntegrandOperator(
               kernel_context,
               weak_form_context,
               operator_context,
               element_context,
               face_info,
               entries.value,
               dofs_in,
               dofs_out
            ),
            ...
         );
      },
      sum_integrand.map.entries
   );
}

// Boundary facet operator wrapper (calls loop over boundary faces)
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename TrialSpace,
   typename Integrand,
   typename ElementDofsIn,
   typename ElementDofsOut>
GENDIL_HOST_DEVICE
void GenericBoundaryFacetOperator(
   KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const TrialSpace& trial_space,
   const Integrand& integrand,
   const ElementDofsIn& dofs_in,
   ElementDofsOut& dofs_out)
{
   BoundaryFaceLoop(
      trial_space,
      element_context.element_index,
      [&] (auto const & face_info)
      {
         GenericBoundaryFacetIntegrandOperator(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            face_info,
            integrand,
            dofs_in,
            dofs_out);
      }
   );
}

// ============================================================================
// Pullback GenericExplicitOperator
// ============================================================================

/**
 * @brief Pullback-based explicit operator.
 *
 */
template<
  class KernelPolicy,
  class WeakForm,
  class WeakFormContext,
  class OperatorContext,
  class DofsInView,
  class DofsOutView>
void GenericExplicitOperator(
  const WeakForm& weak_form,
  const WeakFormContext& wf_ctx,
  const OperatorContext& op_ctx,
  const DofsInView& dofs_in,
  DofsOutView& dofs_out)
{
   GENDIL_REQUIRE_BATCH_SIZE_ONE_FOR_UNAUDITED_OPERATOR(
      KernelPolicy,
      "GenericExplicitOperator" );

   using I = std::remove_cvref_t<WeakForm>;

   constexpr auto TrialName = requirements<I>::trial_name;
   constexpr auto TestName  = requirements<I>::test_name;

   static_assert(TrialName != StaticString{"Error"}, "GenericExplicitOperator: missing TrialSpace in weak_form.");
   static_assert(TestName  != StaticString{"Error"}, "GenericExplicitOperator: missing TestSpace in weak_form.");

   // FE spaces come from wf_ctx via MakeTrialField/MakeTestField
   const auto& trial_space = wf_ctx.template fe_field<TrialName>().space;
   const auto& test_space  = wf_ctx.template fe_field<TestName>().space;
   using IntegrationRule = decltype(op_ctx.integration_rule());

   // Shared memory requirement
   constexpr size_t required_shared_mem = required_shared_memory_v<KernelPolicy, IntegrationRule>;

   mesh::CellIterator<KernelPolicy>(
      trial_space,
      [=] GENDIL_HOST_DEVICE (GlobalIndex element_index) mutable
      {
         GENDIL_SHARED Real _shared_mem[ required_shared_mem ];
         KernelContext<KernelPolicy, required_shared_mem> kernel(_shared_mem);

         // Read local dofs once
         auto u_elem = ReadDofs(kernel, trial_space, element_index, dofs_in);

         // Local accumulation buffer
         using VType = decltype(ReadDofs(kernel, test_space, element_index, dofs_out));
         VType v_elem{};

         ElementContext element_context{ element_index, trial_space.GetCell( element_index ) };

         // Cell terms - use pullback path
         if constexpr (has_cell_contributions_v<WeakForm>)
         {
            GenericCellIntegrandOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               weak_form,
               u_elem,
               v_elem
            );
         }

         // Interior facet terms - use pullback path
         if constexpr (has_interior_facet_contributions_v<WeakForm>)
         {
            GenericInteriorFacetOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               weak_form,
               dofs_in,   // global input view, needed to read plus-side dofs
               u_elem,    // minus-side dofs already read
               v_elem
            );
         }

         // Boundary facet terms - use pullback path
         if constexpr (has_boundary_facet_contributions_v<WeakForm>)
         {
            GenericBoundaryFacetOperator(
               kernel,
               wf_ctx,
               op_ctx,
               element_context,
               trial_space,
               weak_form,
               u_elem,    // minus-side dofs already read
               v_elem
            );
         }

         // Write once
         WriteDofs(kernel, test_space, element_index, v_elem, dofs_out);
      }
   );
}

// ============================================================================
// Pullback GenericOperator class
// ============================================================================

/**
 * @brief Pullback-based generic operator.
 *
 * High-level operator wrapper that uses pullback channels for cell integrands.
 */
template<
   class KernelPolicy,
   class IntegrationRule,
   class WeakForm,
   class WeakFormContext>
class PullbackGenericOperator
{
   WeakForm weak_form;
   WeakFormContext wf_ctx;
   IntegrationRule ir;

   using OperatorContext = decltype(MakeOperatorContext(wf_ctx, ir));
   OperatorContext op_ctx;

public:
   PullbackGenericOperator(WeakForm wf,
                           WeakFormContext ctx,
                           IntegrationRule ir_)
      : weak_form(std::move(wf))
      , wf_ctx(std::move(ctx))
      , ir(ir_)
      , op_ctx(MakeOperatorContext(wf_ctx, ir))
   {}

   template<class input, class output>
   void operator()(const input& dofs_vector_in, output& dofs_vector_out) const
   {
      constexpr auto TrialName = requirements<WeakForm>::trial_name;
      constexpr auto TestName  = requirements<WeakForm>::test_name;
      auto dofs_in = MakeReadOnlyElementTensorView< KernelPolicy >( wf_ctx.template fe_field<TrialName>().space, dofs_vector_in );
      auto dofs_out = MakeWriteOnlyElementTensorView< KernelPolicy >( wf_ctx.template fe_field<TestName>().space, dofs_vector_out );

      GenericExplicitOperator<KernelPolicy>(
         weak_form, wf_ctx, op_ctx, dofs_in, dofs_out
      );
   }
};

/**
 * @brief Factory for pullback-based GenericOperator
 *
 * **Usage:**
 * ```cpp
 * auto pb_op = MakePullbackGenericOperator<KernelPolicy>(
 *    weak_form, wf_ctx, integration_rule);
 * pb_op(input_vector, output_vector);
 * ```
 */
template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakePullbackGenericOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakeGenericOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

template<
  class KernelPolicy,
  class IntegrationRule,
  class WeakFormTuple,
  class WeakFormContext>
auto MakeWeakFormResidualOperator(
   WeakFormTuple weak_form,
   WeakFormContext wf_ctx,
   IntegrationRule ir)
{
   return PullbackGenericOperator<KernelPolicy, IntegrationRule, WeakFormTuple, WeakFormContext>(
      std::move(weak_form),
      std::move(wf_ctx),
      ir
   );
}

} // namespace gendil
