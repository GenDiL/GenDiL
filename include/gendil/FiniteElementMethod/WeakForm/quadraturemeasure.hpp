// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testlineartraits.hpp"

namespace gendil
{

/**
 * @brief Quadrature measure expression for cell integrands
 *
 * Evaluates to: quad_pt_context.weight * quad_pt_context.det_J
 *
 * This scalar, test-free FieldExpr represents the integration measure
 * (quadrature weight × Jacobian determinant) used as a seed in form-lowering
 * to pullback channels.
 *
 * **Usage in form lowering:**
 * ```cpp
 * // Cell integrand: integrate(cells, field_expr)
 * auto channels = pullback(field_expr, QuadratureMeasureExpr{});
 * // Channels now include measure, matching Integrand::operator() semantics
 * ```
 *
 * **Trait summary:**
 * - FieldShape: Scalar
 * - Test-linear: No (test-free)
 * - Requirements: No trial/test spaces
 */
struct QuadratureMeasureExpr : FieldBase
{
   template<
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields>
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const QuadPtContext& quad_pt_context,
      const Fields&) const
   {
      return quad_pt_context.weight * quad_pt_context.det_J;
   }
};

// =============================================================================
// Trait specializations for QuadratureMeasureExpr
// =============================================================================

/**
 * @brief FieldShape trait: QuadratureMeasureExpr is Scalar.
 */
template<>
struct field_shape_impl<QuadratureMeasureExpr>
{
   static constexpr FieldShape value = FieldShape::Scalar;
};

/**
 * @brief Test-linearity trait: QuadratureMeasureExpr is test-free (not test-linear).
 *
 * The measure does not depend on test functions.
 */
template<>
struct test_linearity<QuadratureMeasureExpr>
{
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

/**
 * @brief Requirements trait: QuadratureMeasureExpr has no trial/test space requirements.
 *
 * It only accesses quad_pt_context, not field values or test functions.
 */
template<>
struct requirements<QuadratureMeasureExpr>
{
   static constexpr auto trial_name = StaticString("None");
   static constexpr auto test_name  = StaticString("None");
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr OperatorMask test_mask  = OperatorMask::None;
};

// =============================================================================
// FacetQuadratureMeasureExpr
// =============================================================================

/**
 * @brief Quadrature measure expression for facet integrands
 *
 * Evaluates to: quad_pt_context.weight * quad_pt_context.det_J_facet
 *
 * **Important:** This expression expects FacetQuadraturePointContext, not base QuadraturePointContext.
 * It uses the cleaned-up facet geometry computed by MakeFacetQuadraturePointContext.
 *
 * **Usage:** Works for both boundary and interior facets.
 *
 * **Trait summary:**
 * - FieldShape: Scalar
 * - Test-linear: No (test-free)
 * - Requirements: No trial/test spaces
 */
struct FacetQuadratureMeasureExpr : FieldBase
{
   template<
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename FacetQuadPtContext,
      typename Fields>
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext&,
      const WeakFormContext&,
      const OperatorContext&,
      const ElementContext&,
      const FacetQuadPtContext& facet_quad_pt_context,
      const Fields&) const
   {
      if constexpr (requires { facet_quad_pt_context.det_J_facet; })
      {
         // Use cleaned-up facet measure: weight * det_J_facet
         return facet_quad_pt_context.weight * facet_quad_pt_context.det_J_facet;
      }
      else
      {
         const Real det_J_facet = facet_quad_pt_context.det_J_facet_minus;
         // Fallback for old two-sided contexts: use the current/minus measure.
         return facet_quad_pt_context.weight * det_J_facet;
      }
   }
};

// Trait specializations
template<>
struct field_shape_impl<FacetQuadratureMeasureExpr>
{
   static constexpr FieldShape value = FieldShape::Scalar;
};

template<>
struct test_linearity<FacetQuadratureMeasureExpr>
{
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<>
struct requirements<FacetQuadratureMeasureExpr>
{
   static constexpr auto trial_name = StaticString("None");
   static constexpr auto test_name  = StaticString("None");
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr OperatorMask test_mask  = OperatorMask::None;
};

} // namespace gendil
