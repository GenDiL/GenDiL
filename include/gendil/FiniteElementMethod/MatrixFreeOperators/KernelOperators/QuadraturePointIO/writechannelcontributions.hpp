// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/quadraturemeasure.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writequadraturelocalvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymappingtranspose.hpp"

namespace gendil
{

// =============================================================================
// WriteChannelContributions: Cell channel application
// =============================================================================
//
// **Purpose:**
//    Evaluates PullbackResult channel coefficient expressions at a cell
//    quadrature point and writes them to the existing quadrature contribution
//    container (Du.values and/or Du.gradients).
//
// **Coordinate Frame Convention (CRITICAL):**
//    GradientChannel stores physical-gradient coefficients produced by weak-form
//    pullback. This function applies ApplyMappingTranspose to convert physical
//    gradients to reference-frame gradients before writing to Du.gradients.
//
//    Transformation: grad_ref = inv_J^T * grad_phys
//
//    This matches the current temporary InnerExpr/DotExpr specialization behavior
//    (see inner.hpp, dot.hpp).
//
//    Du.gradients expects reference-frame coefficients.
//    ApplyGradientTestFunctions expects reference-frame inputs.
//
// **Channel Behavior:**
//    If ValueChannel is present:
//      1. Evaluate value coefficient expression at quadrature point
//      2. Write to Du.values via WriteQuadratureLocalValues
//
//    If GradientChannel is present:
//      1. Evaluate gradient coefficient expression at quadrature point (physical frame)
//      2. Apply ApplyMappingTranspose using quad_pt_context.inv_J_mesh
//      3. Write reference-frame coefficient to Du.gradients via WriteQuadratureLocalValues
//
//    Both channels can be written independently in a single call.
//
// **Example usage (future GenericOperator integration):**
//    ```cpp
//    auto channels = pullback(u * v + dot(grad(u), grad(v)), ScalarIdentity{});
//    // channels contains: ValueChannel → u, GradientChannel → grad(u)
//
//    QuadraturePointLoop(kernel_context, integration_rule, [&](const auto& quad_index) {
//       auto quad_pt_context = MakeQuadPtContext(..., quad_index);
//       WriteChannelContributions(
//          kernel_context, weak_form_context, operator_context,
//          element_context, quad_pt_context, fields, channels, quad_index, Du);
//    });
//
//    // Then dispatch test functions based on channel presence:
//    if constexpr (has ValueChannel && has GradientChannel)
//       ApplyValuesAndGradientTestFunctions<true>(..., Du.values, Du.gradients, dofs_out);
//    else if constexpr (has ValueChannel)
//       ApplyTestFunctions<true>(..., Du.values, dofs_out);
//    else if constexpr (has GradientChannel)
//       ApplyGradientTestFunctions<true>(..., Du.gradients, dofs_out);
//    ```
//
// =============================================================================

/**
 * @brief Write PullbackResult channel contributions to cell quadrature container.
 *
 * Evaluates ValueChannel and/or GradientChannel coefficient expressions at the
 * given quadrature point and writes them to Du.values and/or Du.gradients.
 *
 * CRITICAL: GradientChannel coefficients are evaluated in physical frame, then
 * mapped to reference frame via ApplyMappingTranspose before writing to Du.gradients.
 *
 * @tparam KernelContext Kernel execution context
 * @tparam WeakFormContext Weak-form context (coefficients, fields)
 * @tparam OperatorContext Operator context (integration rule)
 * @tparam ElementContext Element context (geometry, DoF mapping)
 * @tparam QuadPtContext Quadrature point context (inv_J_mesh, weights, coords)
 * @tparam Fields Interpolated trial fields
 * @tparam Channels PullbackResult type (compile-time channel map)
 * @tparam QuadIndex Quadrature point index type (TensorIndex)
 * @tparam DuType Quadrature container type (InterpolatedField)
 *
 * @param kernel_context Kernel execution context
 * @param weak_form_context Weak-form context
 * @param operator_context Operator context
 * @param element_context Element context
 * @param quad_pt_context Quadrature point context (provides inv_J_mesh for mapping)
 * @param fields Interpolated trial fields
 * @param channels PullbackResult with ValueChannel and/or GradientChannel
 * @param quad_index Current quadrature point index
 * @param Du Quadrature container with .values and/or .gradients fields
 *
 * @pre Du must be allocated with appropriate .values and/or .gradients fields
 *      matching the channels present in the PullbackResult.
 * @pre This is a cell-domain operation. Do NOT use for facet integrands without
 *      separate facet-specific channel-application design.
 *
 * @post If ValueChannel present: Du.values contains evaluated value coefficients
 * @post If GradientChannel present: Du.gradients contains reference-frame gradient coefficients
 */
template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename QuadPtContext,
   typename Fields,
   typename Channels,
   typename QuadIndex,
   typename DuType>
GENDIL_HOST_DEVICE
void WriteChannelContributions(
   const KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields,
   const Channels& channels,
   const QuadIndex& quad_index,
   DuType& Du)
{
   using C = std::remove_cvref_t<Channels>;

   // ValueChannel: evaluate coefficient expression, write to Du.values
   if constexpr (C::template contains<ValueChannel>())
   {
      const auto& value_coeff_expr = channels.template get<ValueChannel>();

      auto value_coeff_q = value_coeff_expr(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields);

      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         value_coeff_q,
         Du.values);
   }

   // GradientChannel: evaluate (physical frame), map to reference frame, write to Du.gradients
   if constexpr (C::template contains<GradientChannel>())
   {
      const auto& grad_coeff_expr = channels.template get<GradientChannel>();

      auto grad_coeff_q = grad_coeff_expr(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields);

      // CRITICAL: Map physical → reference frame
      // This transformation is required because:
      //   - GradientChannel stores physical-gradient coefficients (from pullback)
      //   - Du.gradients expects reference-frame coefficients
      //   - ApplyGradientTestFunctions expects reference-frame inputs
      // Matches current temporary InnerExpr/DotExpr specialization behavior.
      if constexpr (requires { quad_pt_context.inv_J_mesh; })
      {
         ApplyMappingTranspose(quad_pt_context.inv_J_mesh, grad_coeff_q);
      }
      else if constexpr (requires { quad_pt_context.inv_J_mesh_minus; })
      {
         // For interior facets, use the minus side's inv_J_mesh for mapping
         ApplyMappingTranspose(quad_pt_context.inv_J_mesh_minus, grad_coeff_q);
      }
      else
      {
         static_assert(requires { quad_pt_context.inv_J_mesh; },
            "GradientChannel mapping requires QuadPtContext with inv_J_mesh field");
      }

      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         grad_coeff_q,
         Du.gradients);
   }
}

// ============================================================================
// Form-level lowering to pullback channels
// ============================================================================

/**
 * @brief Lower cell integrand to pullback channels (form-level lowering).
 *
 * For cell integrands `integrate(cells, field_expr)`, this function:
 * 1. Unwraps the inner FieldExpr from the Integrand wrapper
 * 2. Constructs the integration measure seed (quad_pt_context.weight * det_J)
 * 3. Calls FieldExpr pullback with the measure seed
 *
 * This preserves the Integrand::operator() semantics which multiplies the
 * field expression result by the measure during evaluation.
 *
 * @tparam DomainName Cell domain name (e.g., "mesh1")
 * @tparam Expr Inner FieldExpr type
 * @param integrand Cell integrand wrapper
 * @return PullbackResult with channels including integration measure
 */
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto LowerCellIntegrandToPullbackChannels(const Integrand<Cells<DomainName>, Expr>& integrand)
{
   const auto& field_expr = integrand.expr;  // Unwrap
   QuadratureMeasureExpr measure;            // Create measure seed
   return pullback(field_expr, measure);     // FieldExpr pullback with measure
}

// =============================================================================
// Boundary Facet Integrand Lowering
// =============================================================================
//
// Lower boundary facet integrand to pullback expression:
//   integrate(boundary_facets, expr) -> pullback(expr, FacetQuadratureMeasureExpr{})
//
// The measure expression evaluates to: weight * det_J_facet
// where det_J_facet is computed via ComputeFacetGeometry in MakeFacetQuadraturePointContext.
//
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto LowerBoundaryFacetIntegrandToPullbackChannels(
   const Integrand<BoundaryFacets<DomainName>, Expr>& integrand)
{
   const auto& field_expr = integrand.expr;  // Extract field expression (e.g., u * v)
   FacetQuadratureMeasureExpr measure;       // Facet measure seed
   return pullback(field_expr, measure);
}

// =============================================================================
// Interior Facet Integrand Lowering
// =============================================================================
//
// Lower interior facet integrand to pullback expression:
//   integrate(interior_facets, expr) -> pullback(expr, FacetQuadratureMeasureExpr{})
//
// The measure expression evaluates to: weight * det_J_facet
// Same measure as boundary facets (no distinction needed).
//
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto LowerInteriorFacetIntegrandToPullbackChannels(
   const Integrand<InteriorFacets<DomainName>, Expr>& integrand)
{
   const auto& field_expr = integrand.expr;  // Extract field expression (e.g., jump(u) * jump(v))
   FacetQuadratureMeasureExpr measure;       // Facet measure seed (same as boundary)
   return pullback(field_expr, measure);
}

template<
   typename KernelContext,
   typename WeakFormContext,
   typename OperatorContext,
   typename ElementContext,
   typename QuadPtContext,
   typename Fields,
   typename Integrand,
   typename QuadIndex,
   typename DuType>
GENDIL_HOST_DEVICE
void ComputeContributions(
   const KernelContext& kernel_context,
   const WeakFormContext& weak_form_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields,
   const Integrand& integrand,
   const QuadIndex& quad_index,
   DuType& Du)
{
   if constexpr ( CellIntegrand<Integrand> )
   {
      auto channels = LowerCellIntegrandToPullbackChannels(integrand);

      WriteChannelContributions(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields,
         channels,
         quad_index,
         Du);
   }
   else if constexpr( BoundaryFacetIntegrand<Integrand> )
   {
      auto channels = LowerBoundaryFacetIntegrandToPullbackChannels(integrand);

      WriteChannelContributions(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields,
         channels,
         quad_index,
         Du);
   }
   else if constexpr( InteriorFacetIntegrand<Integrand> )
   {
      auto channels = LowerInteriorFacetIntegrandToPullbackChannels(integrand);

      WriteChannelContributions(
         kernel_context,
         weak_form_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields,
         channels,
         quad_index,
         Du);
   }
}

} // namespace gendil
