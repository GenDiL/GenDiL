// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticvector.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <type_traits>

namespace gendil
{

// Forward declaration to avoid circular dependency
struct FieldBase;

// Concept: rank-2 static matrix (SFINAE-safe)
template<class M>
concept static_matrix = requires {
   { static_num_rows_v<M> } -> std::convertible_to<Integer>;
   { static_num_cols_v<M> } -> std::convertible_to<Integer>;
   static_num_rows_v<M> != std::dynamic_extent;
   static_num_cols_v<M> != std::dynamic_extent;
};

// Concept: rank-1 static vector (SFINAE-safe)
template<class V>
concept static_vector_type = requires {
   { static_extent_v<V> } -> std::convertible_to<Integer>;
   static_extent_v<V> != std::dynamic_extent;
};

// Matrix-vector multiplication: y = A * x
// Supports: SerialRecursiveArray<T, M, N> * std::array<T, N>
//          SerialRecursiveArray<T, M, N> * SerialRecursiveArray<T, N>
// Result type: SerialRecursiveArray<T, M>
//
// Constrained to concrete value types only (excludes FieldExpr templates).
//
// The fundamental type exclusion prevents this operator from participating in
// scalar/index arithmetic such as integral_constant * Integer inside layout
// offset calculations (e.g., FixedStridedLayout::Offset stride multiplication).
template<typename Mat, typename Vec>
requires static_matrix<Mat> && static_vector_type<Vec> &&
         (static_num_cols_v<Mat> == static_extent_v<Vec>) &&
         (!std::derived_from<std::remove_cvref_t<Mat>, FieldBase>) &&
         (!std::derived_from<std::remove_cvref_t<Vec>, FieldBase>) &&
         (!std::is_fundamental_v<std::remove_cvref_t<Mat>>) &&
         (!std::is_fundamental_v<std::remove_cvref_t<Vec>>)
GENDIL_HOST_DEVICE
auto operator*(const Mat& A, const Vec& v)
{
   constexpr Integer Rows = static_num_rows_v<Mat>;
   constexpr Integer Cols = static_num_cols_v<Mat>;

   using MatValue = std::remove_cvref_t<decltype(matrix_access(A, 0, 0))>;
   using VecValue = std::remove_cvref_t<decltype(vector_access(v, 0))>;
   using ResultValue = decltype(std::declval<MatValue>() * std::declval<VecValue>());

   SerialRecursiveArray<ResultValue, Rows> result;
   for (Integer i = 0; i < Rows; ++i) {
      result(i) = ResultValue{0};
      for (Integer j = 0; j < Cols; ++j) {
         result(i) += matrix_access(A, i, j) * vector_access(v, j);
      }
   }
   return result;
}

} // namespace gendil
