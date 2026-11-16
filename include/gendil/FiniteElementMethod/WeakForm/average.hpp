// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/arrayarithmetic.hpp"
#include "gendil/Algebra/matrixvectorops.hpp"
#include "gendil/Algebra/staticvectorops.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"

namespace gendil
{

namespace details
{

// Use shared arithmetic without widening floating-point entries merely for
// the internal factor 1/2. Integral entries need a real factor to retain
// fractional averages instead of narrowing back into the operand's type.
template <typename T>
GENDIL_HOST_DEVICE
constexpr auto AverageValues(const T& minus_value, const T& plus_value)
{
   using Scalar = std::conditional_t<StaticMatrix<T>, matrix_value_t<T>,
      std::conditional_t<StaticVector<T>, vector_value_t<T>, T>>;
   using Scale = std::conditional_t<std::is_floating_point_v<Scalar>, Scalar, Real>;
   return Scale{0.5} * (minus_value + plus_value);
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
