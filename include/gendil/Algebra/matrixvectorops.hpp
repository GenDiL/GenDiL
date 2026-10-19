// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticvector.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

// Forward declaration to avoid circular dependency
struct FieldBase;

// Concept: rank-2 static matrix (SFINAE-safe)
template<class M>
concept static_matrix = requires {
   { static_num_rows_v<M> } -> std::convertible_to<Integer>;
   { static_num_cols_v<M> } -> std::convertible_to<Integer>;
   requires static_num_rows_v<M> != std::dynamic_extent;
   requires static_num_cols_v<M> != std::dynamic_extent;
};

// Concept: rank-1 static vector (SFINAE-safe)
template<class V>
concept static_vector_type = requires {
   { static_extent_v<V> } -> std::convertible_to<Integer>;
   requires static_extent_v<V> != std::dynamic_extent;
};

// Scalar scaling uses the same owning matrix storage as vector gradients, so
// a nested-array coefficient can enter the mapping and quadrature-write paths.
template<typename Scalar, typename Entry, size_t Rows, size_t Cols>
   requires (std::is_arithmetic_v<Scalar> && std::is_arithmetic_v<Entry>)
GENDIL_HOST_DEVICE
auto operator*(const Scalar& scalar,
               const std::array<std::array<Entry, Cols>, Rows>& matrix)
   -> SerialRecursiveArray<std::remove_cvref_t<decltype(
      std::declval<Scalar>() * std::declval<Entry>())>, Rows, Cols>
{
   using Result = std::remove_cvref_t<decltype(std::declval<Scalar>() * std::declval<Entry>())>;
   SerialRecursiveArray<Result, Rows, Cols> result;
   for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Cols; ++j) { result(i, j) = scalar * matrix[i][j]; }
   return result;
}

template<typename Entry, typename Scalar, size_t Rows, size_t Cols>
   requires (std::is_arithmetic_v<Entry> && std::is_arithmetic_v<Scalar>)
GENDIL_HOST_DEVICE
auto operator*(const std::array<std::array<Entry, Cols>, Rows>& matrix,
               const Scalar& scalar)
   -> SerialRecursiveArray<std::remove_cvref_t<decltype(
      std::declval<Entry>() * std::declval<Scalar>())>, Rows, Cols>
{
   using Result = std::remove_cvref_t<decltype(std::declval<Entry>() * std::declval<Scalar>())>;
   SerialRecursiveArray<Result, Rows, Cols> result;
   for (size_t i = 0; i < Rows; ++i)
      for (size_t j = 0; j < Cols; ++j) { result(i, j) = matrix[i][j] * scalar; }
   return result;
}

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

   using MatValue =
      typename vector_element_type<
         std::remove_cvref_t<decltype(matrix_access(A, 0, 0))>>::type;
   using VecValue =
      typename vector_element_type<
         std::remove_cvref_t<decltype(vector_access(v, 0))>>::type;
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
