// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/pullback.hpp"
#include "gendil/FiniteElementMethod/WeakForm/quadraturemeasure.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/writequadraturelocalvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymappingtranspose.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil
{

namespace pullback_detail
{

// Check the physical direction axis before mapping or writing. Rank-only DSL
// shapes do not encode a callback's fixed vector/matrix extents.
template<class Value, Integer Dim>
GENDIL_HOST_DEVICE
constexpr void ValidateGradientDimension(const Value&, const TensorIndex<Dim>&)
{
   if constexpr (static_num_cols_v<Value> != std::dynamic_extent)
   {
      static_assert(static_num_cols_v<Value> == Dim,
         "Gradient channel direction count must match the physical domain dimension.");
   }
   else if constexpr (static_extent_v<Value> != std::dynamic_extent)
   {
      static_assert(static_extent_v<Value> == Dim,
         "Gradient channel direction count must match the physical domain dimension.");
   }
}


template<class Sink>
inline constexpr bool has_value_sink_v = []
{
   if constexpr (requires(Sink& sink) { sink.values; })
   {
      return !std::is_same_v<
         std::remove_cvref_t<decltype(std::declval<Sink&>().values)>, Empty>;
   }
   else { return false; }
}();

template<class Sink>
inline constexpr bool has_gradient_sink_v = []
{
   if constexpr (requires(Sink& sink) { sink.gradients; })
   {
      return !std::is_same_v<
         std::remove_cvref_t<decltype(std::declval<Sink&>().gradients)>, Empty>;
   }
   else { return false; }
}();

// A pullback's integration measure and gradient mapping must use geometry for
// the same entity. A physical boundary has an explicit marker; one selected
// side of an interior face is not a physical boundary.
template<PullbackEntity Entity, class QuadPtContext>
inline constexpr bool quadrature_context_compatible_v = []
{
   if constexpr (Entity == PullbackEntity::Cell)
   {
      return requires(const QuadPtContext& context)
      {
         context.weight;
         context.det_J;
         context.inv_J_mesh;
      } && !requires(const QuadPtContext& context) { context.det_J_facet; };
   }
   else if constexpr (Entity == PullbackEntity::BoundaryFacet)
   {
      return requires(const QuadPtContext& context)
      {
         requires is_physical_boundary_v<QuadPtContext>;
         context.weight;
         context.det_J_facet;
         context.inv_J_mesh;
      };
   }
   else
   {
      return requires(const QuadPtContext& context)
      {
         context.weight;
         context.det_J_facet;
         context.inv_J_mesh_minus;
         context.MinusSide();
         context.PlusSide();
      };
   }
}();

} // namespace pullback_detail

template<class Channels, class Sink>
inline constexpr bool current_row_channel_sink_compatible_v =
   pullback_channels_match_rows_v<Channels, PullbackRows::Current> &&
   (!std::remove_cvref_t<Channels>::template Contains<ValueChannel>() ||
      pullback_detail::has_value_sink_v<Sink>) &&
   (!std::remove_cvref_t<Channels>::template Contains<GradientChannel>() ||
      pullback_detail::has_gradient_sink_v<Sink>);

template<class Channels, class MinusSink, class PlusSink>
inline constexpr bool global_interior_channel_sinks_compatible_v =
   pullback_channels_match_rows_v<Channels, PullbackRows::MinusPlus> &&
   (!std::remove_cvref_t<Channels>::template Contains<ValueMinusChannel>() ||
      pullback_detail::has_value_sink_v<MinusSink>) &&
   (!std::remove_cvref_t<Channels>::template Contains<ValuePlusChannel>() ||
      pullback_detail::has_value_sink_v<PlusSink>) &&
   (!std::remove_cvref_t<Channels>::template Contains<GradientMinusChannel>() ||
      pullback_detail::has_gradient_sink_v<MinusSink>) &&
   (!std::remove_cvref_t<Channels>::template Contains<GradientPlusChannel>() ||
      pullback_detail::has_gradient_sink_v<PlusSink>);

// Allocate only the value/gradient storage requested by the current-row pullback.
template <
   typename KernelContext,
   typename IntegrationRule,
   typename TestSpace,
   PullbackEntity Entity,
   typename Channels >
   requires ( Entity != PullbackEntity::GlobalInteriorFacet )
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContainerForSpace(
   const KernelContext & kernel_context,
   const IntegrationRule & integration_rule,
   const TestSpace & test_space,
   const EntityPullbackResult<Entity, Channels> & /*pullback*/ )
{
   constexpr bool NeedValues = Channels::template Contains<ValueChannel>();
   constexpr bool NeedGradients = Channels::template Contains<GradientChannel>();

   if constexpr ( NeedValues || NeedGradients )
   {
      return MakeQuadraturePointContainerForSpace<NeedValues, NeedGradients>(
         kernel_context,
         integration_rule,
         GetMinusFacetFieldVolumeSpace(test_space));
   }
   else
   {
      return InterpolatedField<Empty, Empty>{};
   }
}

// A canonical face has distinct output rows. An unused side needs no storage
// and does not inspect that side's finite element space.
template <
   SemanticSide Side,
   typename KernelContext,
   typename IntegrationRule,
   typename TestSpace,
   typename Channels >
   requires ( Side == SemanticSide::Minus || Side == SemanticSide::Plus )
GENDIL_HOST_DEVICE
auto MakeQuadraturePointContainerForSpace(
   const KernelContext & kernel_context,
   const IntegrationRule & integration_rule,
   const TestSpace & test_space,
   const EntityPullbackResult<PullbackEntity::GlobalInteriorFacet, Channels> & /*pullback*/ )
{
   using Value = std::conditional_t<Side == SemanticSide::Minus, ValueMinusChannel, ValuePlusChannel>;
   using Gradient = std::conditional_t<Side == SemanticSide::Minus, GradientMinusChannel, GradientPlusChannel>;
   constexpr bool NeedValues = Channels::template Contains<Value>();
   constexpr bool NeedGradients = Channels::template Contains<Gradient>();

   if constexpr ( NeedValues || NeedGradients )
   {
      if constexpr ( Side == SemanticSide::Minus )
      {
         return MakeQuadraturePointContainerForSpace<NeedValues, NeedGradients>(
            kernel_context,
            integration_rule,
            GetMinusFacetFieldVolumeSpace(test_space));
      }
      else
      {
         return MakeQuadraturePointContainerForSpace<NeedValues, NeedGradients>(
            kernel_context,
            integration_rule,
            GetPlusFacetFieldVolumeSpace(test_space));
      }
   }
   else
   {
      return InterpolatedField<Empty, Empty>{};
   }
}

/**
 * @brief Compute current-row contributions at one quadrature point.
 *
 * Evaluates the supplied pullback and fills the requested value/gradient
 * entries in contributions. Gradient coefficients are mapped from physical
 * to reference coordinates for subsequent test-function contraction.
 * Cells, physical boundaries, and local interior facets share this operation.
 * Construct the pullback once before entering the quadrature loop.
 */
template<
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename ElementContext,
   typename QuadPtContext,
   typename Fields,
   PullbackEntity Entity,
   typename PullbackChannels,
   typename QuadIndex,
   typename Contributions>
GENDIL_HOST_DEVICE
void ComputeContributions(
   const KernelContext& kernel_context,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const QuadPtContext& quad_pt_context,
   const Fields& fields,
   const EntityPullbackResult<Entity, PullbackChannels>& pullback,
   const QuadIndex& quad_index,
   Contributions& contributions)
{
   using C = EntityPullbackResult<Entity, PullbackChannels>;
   static_assert(Entity == PullbackEntity::Cell ||
                 Entity == PullbackEntity::BoundaryFacet ||
                 Entity == PullbackEntity::LocalInteriorFacet,
      "Current-row contributions require a cell, boundary, or local-interior pullback.");
   static_assert(pullback_detail::quadrature_context_compatible_v<Entity, QuadPtContext>,
      "Pullback entity must match the quadrature point context.");
   static_assert(current_row_channel_sink_compatible_v<C, Contributions>,
      "Current-row pullback channels require matching value/gradient sink storage.");

   // ValueChannel: evaluate coefficient expression, write to contributions.values
   if constexpr (C::template Contains<ValueChannel>())
   {
      const auto& value_coeff_expr = pullback.template Get<ValueChannel>();

      auto value_coeff_q = value_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields);

      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         value_coeff_q,
         contributions.values);
   }

   // GradientChannel: evaluate (physical frame), map to reference frame, write to contributions.gradients
   if constexpr (C::template Contains<GradientChannel>())
   {
      const auto& grad_coeff_expr = pullback.template Get<GradientChannel>();

      auto grad_coeff_q = grad_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         quad_pt_context,
         fields);

      pullback_detail::ValidateGradientDimension(grad_coeff_q, quad_index);

      // Test-gradient contraction consumes reference-frame coefficients.
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
         contributions.gradients);
   }
}

/**
 * @brief Compute separate minus/plus contributions at an interior quadrature point.
 *
 * Each gradient coefficient uses its own side's inverse Jacobian. Only the
 * value/gradient entries present in the supplied pullback are evaluated.
 */
template<
   typename KernelContext,
   typename ExecutionContext,
   typename OperatorContext,
   typename ElementContext,
   typename FacetQuadPtContext,
   typename Fields,
   PullbackEntity Entity,
   typename PullbackChannels,
   typename QuadIndex,
   typename MinusContributions,
   typename PlusContributions>
GENDIL_HOST_DEVICE
void ComputeContributions(
   const KernelContext& kernel_context,
   const ExecutionContext& execution_context,
   const OperatorContext& operator_context,
   const ElementContext& element_context,
   const FacetQuadPtContext& facet_quad_pt_context,
   const Fields& fields,
   const EntityPullbackResult<Entity, PullbackChannels>& pullback,
   const QuadIndex& quad_index,
   MinusContributions& minus_contributions,
   PlusContributions& plus_contributions)
{
   using C = EntityPullbackResult<Entity, PullbackChannels>;
   static_assert(Entity == PullbackEntity::GlobalInteriorFacet,
      "Two-row contributions require a global-interior pullback.");
   static_assert(pullback_detail::quadrature_context_compatible_v<Entity, FacetQuadPtContext>,
      "Pullback entity must match the quadrature point context.");
   static_assert(global_interior_channel_sinks_compatible_v<C, MinusContributions, PlusContributions>,
      "Canonical interior pullback channels require matching minus/plus sink storage.");

   if constexpr (C::template Contains<ValueMinusChannel>())
   {
      const auto& value_coeff_expr =
         pullback.template Get<ValueMinusChannel>();
      auto value_coeff_q = value_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         facet_quad_pt_context,
         fields);
      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         value_coeff_q,
         minus_contributions.values);
   }

   if constexpr (C::template Contains<ValuePlusChannel>())
   {
      const auto& value_coeff_expr =
         pullback.template Get<ValuePlusChannel>();
      auto value_coeff_q = value_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         facet_quad_pt_context,
         fields);
      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         value_coeff_q,
         plus_contributions.values);
   }

   if constexpr (C::template Contains<GradientMinusChannel>())
   {
      const auto& grad_coeff_expr =
         pullback.template Get<GradientMinusChannel>();
      auto grad_coeff_q = grad_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         facet_quad_pt_context,
         fields);
      pullback_detail::ValidateGradientDimension(grad_coeff_q, quad_index);
      ApplyMappingTranspose(
         facet_quad_pt_context.MinusSide().inv_J_mesh,
         grad_coeff_q);
      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         grad_coeff_q,
         minus_contributions.gradients);
   }

   if constexpr (C::template Contains<GradientPlusChannel>())
   {
      const auto& grad_coeff_expr =
         pullback.template Get<GradientPlusChannel>();
      auto grad_coeff_q = grad_coeff_expr(
         kernel_context,
         execution_context,
         operator_context,
         element_context,
         facet_quad_pt_context,
         fields);
      pullback_detail::ValidateGradientDimension(grad_coeff_q, quad_index);
      ApplyMappingTranspose(
         facet_quad_pt_context.PlusSide().inv_J_mesh,
         grad_coeff_q);
      WriteQuadratureLocalValues(
         kernel_context,
         quad_index,
         grad_coeff_q,
         plus_contributions.gradients);
   }
}

// ============================================================================
// Integrand pullback construction
// ============================================================================

/**
 * @brief Construct cell test-value and test-gradient coefficient expressions.
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
 * @return Cell pullback with the integration measure included; coefficient
 * expressions are evaluated later at quadrature points.
 */
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto MakeCellPullback(const Integrand<Cells<DomainName>, Expr>& integrand)
{
   const auto& field_expr = integrand.expr;  // Unwrap
   QuadratureMeasureExpr measure;            // Create measure seed
   return MakeEntityPullbackResult<PullbackEntity::Cell>(
      pullback(field_expr, measure));
}

// =============================================================================
// Boundary Facet Pullback Construction
// =============================================================================
//
// Construct boundary test coefficients with the facet integration measure:
//   integrate(boundary_facets, expr) -> pullback(expr, FacetQuadratureMeasureExpr{})
//
// The measure expression evaluates to: weight * det_J_facet
// where det_J_facet is computed via ComputeFacetGeometry in MakeFacetQuadraturePointContext.
//
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto MakeBoundaryFacetPullback(
   const Integrand<BoundaryFacets<DomainName>, Expr>& integrand)
{
   const auto& field_expr = integrand.expr;  // Extract field expression (e.g., u * v)
   FacetQuadratureMeasureExpr measure;       // Facet measure seed
   return MakeEntityPullbackResult<PullbackEntity::BoundaryFacet>(
      pullback_boundary(field_expr, measure));
}

// =============================================================================
// Interior Facet Pullback Construction
// =============================================================================
//
// Construct current-row test coefficients with the facet integration measure:
//   integrate(interior_facets, expr) -> pullback(expr, FacetQuadratureMeasureExpr{})
//
// The measure expression evaluates to: weight * det_J_facet
// Same measure as boundary facets (no distinction needed).
//
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto MakeLocalInteriorFacetPullback(
   const Integrand<InteriorFacets<DomainName>, Expr>& integrand)
{
   static_assert(
      !has_unqualified_interior_test_trace_v<Expr>,
      "Interior facet test dependencies must use explicit trace syntax. "
      "Use minus(v), plus(v), jump(v), average(v), minus(grad(v)), "
      "plus(grad(v)), jump(grad(v)), or average(grad(v)); unqualified v or "
      "grad(v) is invalid on interior facets.");
   static_assert(
      !has_unqualified_side_dependent_inputs_v<Expr>,
      "Interior facet trial/coefficient finite-element dependencies must use "
      "explicit trace syntax. Use minus(expr), plus(expr), jump(expr), "
      "average(expr), or a trace-aware operator such as upwind(...).");

   const auto& field_expr = integrand.expr;  // Extract field expression (e.g., jump(u) * jump(v))
   FacetQuadratureMeasureExpr measure;       // Facet measure seed (same as boundary)
   return MakeEntityPullbackResult<PullbackEntity::LocalInteriorFacet>(
      pullback(field_expr, measure));
}

// Construct separate minus/plus test coefficients for canonical facet traversal.
template<StaticString DomainName, FieldExpr Expr>
GENDIL_HOST_DEVICE
auto MakeGlobalInteriorFacetPullback(
   const Integrand<InteriorFacets<DomainName>, Expr>& integrand)
{
   static_assert(
      !has_unqualified_interior_test_trace_v<Expr>,
      "Interior facet test dependencies must use explicit trace syntax. "
      "Use minus(v), plus(v), jump(v), average(v), minus(grad(v)), "
      "plus(grad(v)), jump(grad(v)), or average(grad(v)); unqualified v or "
      "grad(v) is invalid on interior facets.");
   static_assert(
      !has_unqualified_side_dependent_inputs_v<Expr>,
      "Interior facet trial/coefficient finite-element dependencies must use "
      "explicit trace syntax. Use minus(expr), plus(expr), jump(expr), "
      "average(expr), or a trace-aware operator such as upwind(...).");

   const auto& field_expr = integrand.expr;
   FacetQuadratureMeasureExpr measure;
   return MakeEntityPullbackResult<PullbackEntity::GlobalInteriorFacet>(
      pullback_global_interior(field_expr, measure));
}

} // namespace gendil
