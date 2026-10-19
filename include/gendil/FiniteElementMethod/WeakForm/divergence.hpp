// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/Utilities/tensorproductdata.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/coefficient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/finiteelementfield.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>

namespace gendil
{
namespace weak_form_detail
{
template<class E>
inline constexpr bool is_divergence_operand_v = is_vector_field_reference_v<E>;
template<StaticString Name>
inline constexpr bool is_divergence_operand_v<FiniteElementField<Name>> = true;

// Gradient-channel fallback: grad(component, direction) is already physical.
// Keep this contraction below the semantic node so other lowerers can still
// recognize divergence without reconstructing it from arbitrary algebra.
struct DivergenceTrace
{
   template<typename T, Integer Components, Integer Dim>
   GENDIL_HOST_DEVICE
   T operator()(const SerialRecursiveArray<T, Components, Dim>& gradient) const
   {
      static_assert(Components == Dim,
         "div requires one vector component per physical dimension.");
      T result{};
      for (Integer d = 0; d < Dim; ++d) { result += gradient(d, d); }
      return result;
   }

   template<typename Gradient>
   GENDIL_HOST_DEVICE
   Real operator()(const Gradient&) const
   {
      static_assert(dependent_false_v<Gradient>,
         "div requires a vector field with a matrix-valued gradient.");
      return 0;
   }
};
} // namespace weak_form_detail

template<FieldExpr Expr>
struct DivergenceExpr : FieldBase
{
   static_assert(weak_form_detail::is_divergence_operand_v<Expr>,
      "div requires a named vector trial, test, or prescribed finite-element field.");
   Expr expr;

   GENDIL_HOST_DEVICE
   explicit DivergenceExpr(const Expr& field) : expr(field) {}

   template<class KernelContext, class WeakFormContext, class OperatorContext,
            class ElementContext, class QuadPtContext, class Fields>
   GENDIL_HOST_DEVICE
   auto operator()(const KernelContext& kernel_context, const WeakFormContext&,
                   const OperatorContext&, const ElementContext&,
                   const QuadPtContext& quad_pt_context, const Fields& fields) const
   {
      if constexpr (is_test_linear_v<Expr>)
      {
         static_assert(dependent_false_v<KernelContext>,
            "Test-space divergence must be lowered through pullback before numerical evaluation.");
      }
      else
      {
         return weak_form_detail::DivergenceTrace{}(
            ReadFieldGradientAtQuadraturePoint<Expr::name>(
               kernel_context, quad_pt_context, fields));
      }
   }
};

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const DivergenceExpr<Expr>& divergence)
{
   return os << "div(" << divergence.expr << ")";
}

/**
 * @brief Express the physical divergence of a named vector finite-element field.
 *
 * Use div(u)*div(v) for a cell grad-div term, or
 * average(div(u))*dot(jump(v), Normal{}) for an interior normal coupling.
 * The result is the sum of derivatives du_i/dx_i. The field must have one
 * component per physical dimension. Take divergence before averaging or
 * selecting a facet trace, for example average(div(u)).
 *
 * Trial/test fields use VectorTrialSpace and VectorTestSpace. Prescribed fields
 * use FiniteElementField bound through MakeFiniteElementField; div(w)*div(v)
 * then gives an input-free load. The selected space must have one component
 * per physical dimension, checked when the operator is constructed.
 *
 * The returned DivergenceExpr retains the operation's identity through semantic
 * analysis and affine extraction. Evaluation contracts the existing physical
 * gradient, and test pullback writes a diagonal physical-gradient coefficient.
 * No additional derivative interpolation or quadrature storage is required.
 * Arbitrary coefficients, compound operands, and derivative nesting are not
 * admitted. The explicit inner(grad(field), identity) encoding remains usable
 * as an algebraic fallback for trial/test fields.
 */
template<FieldExpr Field>
   requires weak_form_detail::is_divergence_operand_v<Field>
GENDIL_HOST_DEVICE
auto div(const Field& field)
{
   return DivergenceExpr<Field>{field};
}
} // namespace gendil
