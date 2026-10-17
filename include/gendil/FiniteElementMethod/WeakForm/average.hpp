// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"
#include "gendil/Utilities/arraytraits.hpp"

namespace gendil
{

// =============================================================================
// Helper: Elementwise Averaging for Scalar and Vector-Valued Expressions
// =============================================================================
//
// AverageValues computes 0.5 * (minus_value + plus_value) for:
//   - Scalar types: uses operator+
//   - std::array types: elementwise averaging
//
// vector-valued coefficient in expressions like:
//     dot(average(grad(u)), normal) * jump(v)
//

namespace details
{

// Scalar case: use operator+
template <typename T>
GENDIL_HOST_DEVICE
constexpr std::enable_if_t<!is_std_array_v<T>, T>
AverageValues(const T& minus_value, const T& plus_value)
{
   return 0.5 * (minus_value + plus_value);
}

// std::array case: elementwise averaging
template <typename T, std::size_t N>
GENDIL_HOST_DEVICE
constexpr std::array<T, N> AverageValues(
   const std::array<T, N>& minus_value,
   const std::array<T, N>& plus_value)
{
   std::array<T, N> result;
   for (std::size_t i = 0; i < N; ++i)
   {
      result[i] = 0.5 * (minus_value[i] + plus_value[i]);
   }
   return result;
}

} // namespace details

template < FieldExpr Expr>
struct AverageExpr : FieldBase
{
   Expr expr;

   GENDIL_HOST_DEVICE
   AverageExpr(const Expr& expr_)
      : expr(expr_)
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const ElementContext & element_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      static_assert(is_test_free_v<Expr>,
         "average(expr) numerical evaluation requires a test-free expression; lower test expressions through pullback.");
      static_assert(!is_physical_boundary_v<QuadPtContext>,
         "Boundary average(expr) must be lowered through boundary pullback before numerical evaluation.");

      // Pullback consumes test averages and boundary identities. Only interior
      // input/coefficient averages survive into quadrature-point evaluation.
      if constexpr (is_test_free_v<Expr> &&
                    !is_physical_boundary_v<QuadPtContext>)
      {
         auto minus_value = minus(expr)(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields
         );

         auto plus_value = plus(expr)(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields
         );

         return details::AverageValues(minus_value, plus_value);
      }
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const AverageExpr<Expr>& average)
{
   return os << "{{ " << average.expr << " }}";
}

template < FieldExpr Expr >
GENDIL_HOST_DEVICE
auto average(const Expr& expr)
{
   return AverageExpr<Expr>(expr);
}

} // namespace gendil
