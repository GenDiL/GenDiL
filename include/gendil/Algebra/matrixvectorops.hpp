// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

// Forward declaration to avoid circular dependency
struct FieldBase;

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

// Multiply compatible fixed-size matrix/vector values through their const
// accessors. The result owns SerialRecursiveArray<ProductScalar, Rows> storage,
// using the promotion of the inferred scalar product. Weak-form expressions
// use their separate symbolic multiplication overloads.
template<typename Mat, typename Vec>
requires StaticMatrix<Mat> && StaticVector<Vec> &&
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

   using MatValue = matrix_value_t<Mat>;
   using VecValue = vector_value_t<Vec>;
   using ResultValue = decltype(std::declval<MatValue>() * std::declval<VecValue>());

   SerialRecursiveArray<ResultValue, Rows> result;
   for (Integer i = 0; i < Rows; ++i) {
      result(i) = ResultValue{0};
      for (Integer j = 0; j < Cols; ++j) {
         result(i) += static_cast<MatValue>(GetValue(A, i, j)) *
                      static_cast<VecValue>(GetValue(v, j));
      }
   }
   return result;
}

} // namespace gendil
