// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>
#include <utility>

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/tensorproductdata.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"

namespace gendil
{
namespace weak_form_detail
{
// Read each off-diagonal pair once and fill both output entries. Convert
// integral entries before averaging so halves are retained without overflow
// in integer addition/subtraction. Floating-point inputs retain their precision.
template<bool Skew, class Matrix>
GENDIL_HOST_DEVICE constexpr auto MatrixSymmetryPart(const Matrix& matrix)
{
   static_assert(StaticMatrix<Matrix>,
      "sym and skew require a readable static matrix with real arithmetic entries.");
   if constexpr (StaticMatrix<Matrix>)
   {
      constexpr Integer size = static_num_rows_v<Matrix>;
      static_assert(size == static_num_cols_v<Matrix>,
         "sym and skew require a square matrix.");
      if constexpr (size == static_num_cols_v<Matrix>)
      {
         using Scalar = matrix_value_t<Matrix>;
         using Entry = std::conditional_t<std::is_floating_point_v<Scalar>, Scalar, double>;
         SerialRecursiveArray<Entry, size, size> result{};
         for (Integer i = 0; i < size; ++i)
         {
            if constexpr (!Skew)
               result(i, i) = static_cast<Entry>(static_cast<Scalar>(GetValue(matrix, i, i)));
            for (Integer j = i + 1; j < size; ++j)
            {
               const Entry a = Entry(.5)*static_cast<Entry>(static_cast<Scalar>(GetValue(matrix, i, j)));
               const Entry b = Entry(.5)*static_cast<Entry>(static_cast<Scalar>(GetValue(matrix, j, i)));
               const Entry value = Skew ? a - b : a + b;
               result(i, j) = value;
               result(j, i) = Skew ? -value : value;
            }
         }
         return result;
      }
   }
}
} // namespace weak_form_detail

template<FieldExpr Expr>
struct SymExpr : FieldBase
{
   static_assert(MatrixExpr<Expr>, "sym requires a matrix-shaped expression.");
   Expr expr;

   GENDIL_HOST_DEVICE constexpr explicit SymExpr(const Expr& operand) : expr(operand) {}

   template<class... Args>
   GENDIL_HOST_DEVICE constexpr auto operator()(Args&&... args) const
   {
      const auto& value = expr(std::forward<Args>(args)...);
      return weak_form_detail::MatrixSymmetryPart<false>(value);
   }
};

template<FieldExpr Expr>
struct SkewExpr : FieldBase
{
   static_assert(MatrixExpr<Expr>, "skew requires a matrix-shaped expression.");
   Expr expr;

   GENDIL_HOST_DEVICE constexpr explicit SkewExpr(const Expr& operand) : expr(operand) {}

   template<class... Args>
   GENDIL_HOST_DEVICE constexpr auto operator()(Args&&... args) const
   {
      const auto& value = expr(std::forward<Args>(args)...);
      return weak_form_detail::MatrixSymmetryPart<true>(value);
   }
};

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const SymExpr<Expr>& value)
{ return os << "sym(" << value.expr << ")"; }

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const SkewExpr<Expr>& value)
{ return os << "skew(" << value.expr << ")"; }

/**
 * @brief Take the symmetric part of a square matrix expression.
 *
 * sym(A) represents (A + transpose(A))/2. For example,
 * inner(sym(grad(u)), sym(grad(v))) pairs symmetric vector gradients.
 * The operand is evaluated once and the result owns its entries in a
 * SerialRecursiveArray. Float and long-double entries retain their types;
 * integral entries produce double entries so half-integer values are preserved.
 * Readable proxies follow their inferred scalar type. Only real arithmetic
 * static matrices are supported; rectangular values are rejected.
 * Test pullback applies the same projection to the incoming matrix seed.
 */
template<MatrixExpr Expr>
GENDIL_HOST_DEVICE constexpr auto sym(const Expr& expr)
{ return SymExpr<Expr>{expr}; }

/**
 * @brief Take the skew-symmetric part (A - transpose(A))/2 of a square matrix.
 *
 * skew(A) has zero diagonal and opposite off-diagonal entries. Like sym(A),
 * it evaluates its operand once, returns owning storage, and applies the same
 * projection to test seeds under the real Frobenius inner product.
 */
template<MatrixExpr Expr>
GENDIL_HOST_DEVICE constexpr auto skew(const Expr& expr)
{ return SkewExpr<Expr>{expr}; }

/**
 * @brief Express the small-strain tensor of a named vector trial/test field.
 *
 * sym_grad(u) is sym(grad(u)), with grad(u)(i,j) = du_i/dx_j.
 * The field must have one component per spatial dimension. Apply facet
 * selectors outside the helper, for example average(sym_grad(u)). Prescribed
 * fields, compound differentiation, and nested derivatives remain unsupported.
 */
template<FieldExpr Field>
   requires is_vector_field_reference_v<Field>
auto sym_grad(const Field& field)
{ return sym(grad(field)); }

/// Express the skew part of a named vector trial/test field's gradient as
/// skew(grad(field)). The same operand and dimension rules as sym_grad apply.
template<FieldExpr Field>
   requires is_vector_field_reference_v<Field>
auto skew_grad(const Field& field)
{ return skew(grad(field)); }

} // namespace gendil
