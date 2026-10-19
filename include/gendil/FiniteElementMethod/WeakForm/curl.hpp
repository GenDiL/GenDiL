// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/tensorproductdata.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/coefficient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/finiteelementfield.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include <type_traits>

namespace gendil
{
namespace weak_form_detail
{
template<class E>
inline constexpr bool is_curl_operand_v = is_vector_field_reference_v<E>;
template<StaticString Name>
inline constexpr bool is_curl_operand_v<FiniteElementField<Name>> = true;

// Interpret the existing physical gradient without adding a derivative channel.
// Its vector-field layout is gradient(component, physical direction).
struct CurlFromGradient
{
   template<typename Gradient>
   GENDIL_HOST_DEVICE
   auto operator()(const Gradient& gradient) const
   {
      if constexpr (StaticMatrix<Gradient>)
      {
         constexpr auto Components = static_num_rows_v<Gradient>;
         constexpr auto Dim = static_num_cols_v<Gradient>;
         static_assert(Components == 3 && Dim == 3,
            "curl requires a 3-by-3 physical vector-field gradient.");
         if constexpr (Components == 3 && Dim == 3)
         {
            using T = matrix_value_t<Gradient>;
            SerialRecursiveArray<T, 3> result{};
            result(0) = static_cast<T>(GetValue(gradient, 2, 1)) -
                        static_cast<T>(GetValue(gradient, 1, 2));
            result(1) = static_cast<T>(GetValue(gradient, 0, 2)) -
                        static_cast<T>(GetValue(gradient, 2, 0));
            result(2) = static_cast<T>(GetValue(gradient, 1, 0)) -
                        static_cast<T>(GetValue(gradient, 0, 1));
            return result;
         }
      }
      else
      {
         static_assert(dependent_false_v<Gradient>,
            "curl requires a vector field with a matrix-valued gradient.");
      }
   }
};
} // namespace weak_form_detail

template<FieldExpr Expr>
struct CurlExpr : FieldBase
{
   static_assert(weak_form_detail::is_curl_operand_v<Expr>,
      "curl requires a named vector trial, test, or prescribed finite-element field.");
   Expr expr;

   GENDIL_HOST_DEVICE
   explicit CurlExpr(const Expr& field) : expr(field) {}

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
            "Test-space curl must be lowered through pullback before numerical evaluation.");
      }
      else
      {
         return weak_form_detail::CurlFromGradient{}(
            ReadFieldGradientAtQuadraturePoint<Expr::name>(
               kernel_context, quad_pt_context, fields));
      }
   }
};

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const CurlExpr<Expr>& expression)
{
   return os << "curl(" << expression.expr << ")";
}

/**
 * @brief Express the three-dimensional curl of a named vector field.
 *
 * The result is the vector (dy(u_z) - dz(u_y), dz(u_x) - dx(u_z),
 * dx(u_y) - dy(u_x)). Its vector shape is known when the expression is
 * formed; operator construction validates a three-dimensional space with
 * exactly three vector components.
 *
 * Use dot(curl(u), curl(v)) for a curl-curl form. Prescribed FiniteElementField
 * values may also be differentiated. Take curl before selecting a trace, for example
 * average(curl(u)); compound operands and derivative nesting are unsupported.
 * Two-dimensional rotation operators are deferred to separately named DSL functions.
 */
template<FieldExpr Field>
   requires weak_form_detail::is_curl_operand_v<Field>
GENDIL_HOST_DEVICE
auto curl(const Field& field)
{
   return CurlExpr<Field>{field};
}
} // namespace gendil
