// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/arrayarithmetic.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

// Forward declaration to avoid circular dependency
struct FieldBase;

// Matrix scaling uses owning recursive storage, including for read-only and
// proxy operands. Native recursive-array overloads retain their contracts.
template<class Scalar, StaticMatrix Matrix>
   requires (std::is_arithmetic_v<Scalar> &&
             array_arithmetic_detail::class_operand<Matrix> &&
             !array_arithmetic_detail::is_recursive_array<std::remove_cvref_t<Matrix>>)
GENDIL_HOST_DEVICE
constexpr auto operator*(const Scalar& scalar, const Matrix& matrix)
{
   using Entry = std::remove_cvref_t<decltype(
      std::declval<Scalar>() * std::declval<matrix_value_t<Matrix>>())>;
   using Result = array_arithmetic_detail::result_t<Entry, Matrix, false>;
   return array_arithmetic_detail::ApplyEntrywise<Result>(
      matrix, [&](auto value) { return scalar * value; });
}

template<StaticMatrix Matrix, class Scalar>
   requires (std::is_arithmetic_v<Scalar> &&
             array_arithmetic_detail::class_operand<Matrix> &&
             !array_arithmetic_detail::is_recursive_array<std::remove_cvref_t<Matrix>>)
GENDIL_HOST_DEVICE
constexpr auto operator*(const Matrix& matrix, const Scalar& scalar)
{
   using Entry = std::remove_cvref_t<decltype(
      std::declval<matrix_value_t<Matrix>>() * std::declval<Scalar>())>;
   using Result = array_arithmetic_detail::result_t<Entry, Matrix, false>;
   return array_arithmetic_detail::ApplyEntrywise<Result>(
      matrix, [&](auto value) { return value * scalar; });
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
