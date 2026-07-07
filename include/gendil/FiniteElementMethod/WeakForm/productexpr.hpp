// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productkind.hpp"
#include "gendil/Algebra/matrixvectorops.hpp"

namespace gendil
{

/**
 * @brief Expression representing algebraic product of two field expressions.
 *
 * ProductExpr classifies the product operation by ProductKind and determines
 * the result FieldShape at compile time. Supported product kinds:
 *
 * - ScalarTimes: Scalar multiplication (S×S, S×V, V×S, S×M, M×S)
 *   Result shape is the non-scalar operand, or Scalar if both are Scalar.
 *
 * - MatVec: Matrix-vector multiplication (M×V)
 *   Result shape is Vector.
 *
 * - MatMat: Matrix-matrix multiplication (M×M)
 *   Result shape is Matrix.
 *
 * Unsupported products (Vector×Vector, Vector×Matrix) are rejected at
 * compile time with a clear error message. Use dot(), inner(), outer(),
 * or hadamard() for explicit vector/matrix operations.
 *
 * @tparam LHS Left-hand side field expression
 * @tparam RHS Right-hand side field expression
 */
template<FieldExpr LHS, FieldExpr RHS>
struct ProductExpr : FieldBase
{
   LHS lhs;
   RHS rhs;

   // Deduce ProductKind from operand types
   static constexpr ProductKind product_kind =
      product_kind_v<LHS, RHS>;

   // Deduce result FieldShape from operand types
   static constexpr FieldShape field_shape =
      product_field_shape_v<LHS, RHS>;

   // Ensure product is supported
   static_assert(product_kind != ProductKind::Unsupported,
      "Unsupported ProductExpr. Vector×Vector and Vector×Matrix products "
      "are ambiguous. Use dot(a,b), inner(A,B), outer(a,b), or hadamard(a,b) "
      "to specify the intended operation explicitly.");

   // Kind-specific validation
   static_assert(
      (product_kind == ProductKind::ScalarTimes) ==
         (ScalarExpr<LHS> || ScalarExpr<RHS>),
      "ScalarTimes requires at least one scalar operand");

   static_assert(
      (product_kind == ProductKind::MatVec) ==
         (MatrixExpr<LHS> && VectorExpr<RHS>),
      "MatVec requires Matrix × Vector");

   static_assert(
      (product_kind == ProductKind::MatMat) ==
         (MatrixExpr<LHS> && MatrixExpr<RHS>),
      "MatMat requires Matrix × Matrix");

   GENDIL_HOST_DEVICE
   constexpr ProductExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template<typename... Args>
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const
   {
      // Ensure product is supported (should have been caught at construction)
      static_assert(product_kind != ProductKind::Unsupported,
         "ProductExpr::operator() called for unsupported ProductKind.");

      // Evaluate operands and apply product
      // ScalarTimes, MatVec, and MatMat all use operator* on evaluated values
      auto lhs_q = lhs(std::forward<Args>(args)...);
      auto rhs_q = rhs(std::forward<Args>(args)...);
      return lhs_q * rhs_q;
   }
};

// =============================================================================
// Stream output for ProductExpr
// =============================================================================

/**
 * @brief Stream output for ProductExpr.
 *
 * Prints the ProductExpr in infix notation with parentheses.
 */
template<FieldExpr LHS, FieldExpr RHS>
std::ostream& operator<<(std::ostream& os, const ProductExpr<LHS, RHS>& expr)
{
   return os << "(" << expr.lhs << " * " << expr.rhs << ")";
}

// =============================================================================
// ProductExpr operator* overload
// =============================================================================

/**
 * @brief ScalarTimes, MatVec, and MatMat operator* → ProductExpr
 *
 * Handles ProductKind::ScalarTimes products:
 * - Scalar × Scalar → Scalar (u × v, μ × v, v × v)
 * - Scalar × Vector → Vector (μ × grad(v), u × grad(v))
 * - Vector × Scalar → Vector (grad(v) × μ, grad(v) × u)
 * - Scalar × Matrix → Matrix (μ × grad(vᵥ), u × grad(vᵥ))
 * - Matrix × Scalar → Matrix (grad(vᵥ) × μ, grad(vᵥ) × u)
 *
 * Handles ProductKind::MatMat products:
 * - Matrix × Matrix → Matrix (A × A, A × grad(vᵥ), grad(vᵥ) × A)
 *
 * Handles ProductKind::MatVec products:
 * - Matrix × Vector → Vector (A × β, A × Normal{}, grad(V) × Normal{})
 *
 * Examples:
 * - μ × v, v × μ (scalar coefficient × test field)
 * - u × v, v × u (trial × test)
 * - μ × grad(v) (scalar coefficient × gradient)
 * - β × μ, A × μ (vector/matrix coefficient × scalar multiplier)
 * - A × A (matrix coefficient × matrix coefficient)
 * - grad(vᵥ) × grad(vᵥ) (gradient × gradient)
 *
 * Constraint: is_productexpr_syntax_candidate_v ensures this overload
 * and MultFieldExpr operator* are mutually exclusive.
 *
 * ProductExpr is the single weak-form node for supported matrix-vector,
 * matrix-matrix, and scalar-multiplication products.
 */
template<class LHS, class RHS>
   requires is_productexpr_syntax_candidate_v<LHS, RHS>
GENDIL_HOST_DEVICE
constexpr auto operator*(const LHS& lhs, const RHS& rhs)
{
   return ProductExpr<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>{lhs, rhs};
}

} // namespace gendil
