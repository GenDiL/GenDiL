// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/prelude.hpp"

namespace gendil {

/**
 * @brief Expression representing rank-2 × rank-1 matrix-vector product.
 *
 * Semantics: M[M, N] × v[N] → result[M]
 *
 * For vector finite element gradients:
 *   grad(u)[NumComp, Dim] * v[Dim] → result[NumComp]
 *
 * NOT for rank-1 × rank-1 contraction (use DotExpr/InnerExpr for scalar gradients).
 *
 * Example usage:
 *   Vector FE: grad(u) has shape [NumComp, Dim], Normal has shape [Dim]
 *   MatVecExpr<GradientExpr<TrialSpace<"u">>, Normal>(grad(u), Normal{})
 *   evaluates to result[NumComp] = sum_i grad(u)[a, i] * Normal[i]
 *
 * @tparam MatrixExpr Expression evaluating to rank-2 tensor [M, N]
 * @tparam VectorExpr Expression evaluating to rank-1 tensor [N]
 */
template<typename MatrixExpr, typename VectorExpr>
struct MatVecExpr : FieldBase {
   MatrixExpr matrix_expr;
   VectorExpr vector_expr;

   constexpr MatVecExpr(const MatrixExpr& mat, const VectorExpr& vec)
      : matrix_expr(mat), vector_expr(vec) {}

   template<typename... Args>
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const {
      auto mat_val = matrix_expr(std::forward<Args>(args)...);
      auto vec_val = vector_expr(std::forward<Args>(args)...);

      // Contract: mat[i,j] * vec[j] → result[i]
      //
      // For SerialRecursiveArray<T, M, N> and SerialRecursiveArray<T, N>:
      //   result[i] = sum_j mat(i, j) * vec(j)
      //
      // This is a numeric contraction on concrete values, not DSL-level dot() which
      // would construct another expression node.

      using MatType = decltype(mat_val);
      using VecType = decltype(vec_val);

      // Static check: matrix must be rank-2, vector must be rank-1
      // For SerialRecursiveArray, rank is the number of template parameters
      //
      // This check only instantiates when operator() is evaluated, not during classification.
      static_assert(MatType::rank == 2,
         "MatVecExpr represents rank-2 × rank-1 matrix-vector multiplication. "
         "The matrix expression evaluates to rank-1 (scalar gradient). "
         "For scalar gradients, use dot(grad(u), Normal{}) instead of grad(u) * Normal{}.");
      static_assert(VecType::rank == 1,
         "MatVecExpr requires rank-1 vector. "
         "The vector expression does not evaluate to rank-1.");

      // Extract dimensions
      constexpr Integer M = MatType::template dimension<0>;
      constexpr Integer N = MatType::template dimension<1>;
      constexpr Integer N_vec = VecType::template dimension<0>;

      static_assert(N == N_vec,
         "MatVecExpr dimension mismatch: matrix columns must match vector length.");

      // Perform contraction: result[i] = sum_j mat(i, j) * vec(j)
      using ResultType = SerialRecursiveArray<Real, M>;
      ResultType result;

      for (Integer i = 0; i < M; ++i) {
         result(i) = Real{0};
         for (Integer j = 0; j < N; ++j) {
            result(i) += mat_val(i, j) * vec_val(j);
         }
      }

      return result;
   }
};

/**
 * @brief Syntax-level operator* for GradientExpr<VectorField> * Normal → MatVecExpr
 *
 * DSL Policy: grad(VectorTestSpace<"v">) * Normal{} means matrix-vector multiplication,
 * where VectorTestSpace declares a vector-valued field so grad has shape [NumComp, Dim].
 *
 * For scalar fields, use: dot(grad(TestSpace<"v">), Normal{})
 *
 * Constraint: E must be a vector field reference (VectorTrialSpace or VectorTestSpace).
 * Plain TrialSpace/TestSpace are scalar-valued and do not match this overload.
 *
 * @tparam E Vector field reference (VectorTrialSpace<Name> or VectorTestSpace<Name>)
 */
template<FieldExpr E>
   requires is_vector_field_reference_v<E>
constexpr auto operator*(const GradientExpr<E>& grad, const Normal& n) {
   return MatVecExpr<GradientExpr<E>, Normal>(grad, n);
}

} // namespace gendil
