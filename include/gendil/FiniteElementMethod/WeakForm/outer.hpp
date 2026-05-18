// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/Utilities/MathHelperFunctions/outer.hpp"

namespace gendil
{

// =============================================================================
// OuterExpr
// =============================================================================

/**
 * @brief Outer product expression for vector-valued SIPDG tensor jumps.
 *
 * OuterExpr represents the outer product of two vector-valued expressions:
 *    outer(a, b)[c][j] = a[c] * b[j]
 *
 * **Orientation:** Component-first indexing, matching grad(U) tensor layout.
 *    grad(U)(component, direction) = ∂U_component/∂x_direction
 *
 * **Target use cases (vector SIPDG):**
 *    outer(jump(V), normal) - tensor jump for interior consistency
 *    outer(V, normal)       - tensor jump for boundary consistency
 *
 * **Shape requirements:**
 *    - LHS must be Vector-shaped (field components)
 *    - RHS must be Vector-shaped (spatial vector, e.g., Normal)
 *    - Result is Matrix-shaped
 *
 * **Test-linearity (strict XOR rule):**
 *    - Exactly one operand test-linear → OuterExpr is test-linear
 *    - Both operands test-linear → NonlinearInTest (quadratic, invalid)
 *    - Both operands test-free → OuterExpr is test-free
 *
 * @tparam LHS Left-hand side vector expression (field components)
 * @tparam RHS Right-hand side vector expression (spatial vector)
 */
template<FieldExpr LHS, FieldExpr RHS>
struct OuterExpr : FieldBase
{
   static_assert(is_valid_outer_expr_v<LHS, RHS>,
      "OuterExpr requires Vector×Vector operands. "
      "Both operands must have FieldShape::Vector.");

   LHS lhs;
   RHS rhs;

   OuterExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template<typename... Args>
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const
   {
      // Evaluate both operands to get vectors
      auto lhs_val = lhs(std::forward<Args>(args)...);
      auto rhs_val = rhs(std::forward<Args>(args)...);

      // Outer product: result[c][j] = lhs_val[c] * rhs_val[j]
      // Returns SerialRecursiveArray<Real, NumComp, Dim>
      return Outer(lhs_val, rhs_val);
   }
};

/**
 * @brief Prints OuterExpr in human-readable form.
 */
template<FieldExpr LHS, FieldExpr RHS>
std::ostream& operator<<(std::ostream& os, const OuterExpr<LHS, RHS>& expr)
{
   return os << "outer(" << expr.lhs << ", " << expr.rhs << ")";
}

/**
 * @brief Factory function for creating OuterExpr.
 *
 * Usage:
 *    outer(jump(U), normal)  // Interior tensor jump
 *    outer(V, normal)        // Boundary tensor jump
 *
 * @tparam LHS Left-hand side vector expression
 * @tparam RHS Right-hand side vector expression
 * @return OuterExpr instance
 */
template<FieldExpr LHS, FieldExpr RHS>
auto outer(const LHS& lhs, const RHS& rhs)
{
   return OuterExpr<LHS, RHS>(lhs, rhs);
}

} // namespace gendil
