// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testlineartraits.hpp"

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

// Type trait to detect std::array
template <typename T>
struct is_std_array : std::false_type {};

template <typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template <typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;

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
      if constexpr ( is_side_evaluable_v<Expr> )
      {
         // Side-evaluable average: 0.5 * (minus + plus). This covers trial
         // traces, coefficient expressions, and explicit FE-field expressions.
         auto minus_value = expr(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context.MinusSide(),
            fields.minus_fields
         );

         auto plus_value = expr(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context.PlusSide(),
            fields.plus_fields
         );

         return details::AverageValues(minus_value, plus_value);
      }
      else
      {
         // Test-side average: 0.5 * minus (only current side)
         return 0.5 * expr(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields.minus_fields
         );
      }
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const AverageExpr<Expr>& average)
{
   return os << "{{ " << average.expr << " }}";
}

template < FieldExpr Expr >
auto average(const Expr& expr)
{
   return AverageExpr<Expr>(expr);
}

} // namespace gendil
