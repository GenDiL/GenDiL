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

// Matrix composition accepts any compatible const-readable static matrices.
// Its owning result follows scalar promotion, independently of input storage.
template<StaticMatrix Left, StaticMatrix Right>
   requires (static_num_cols_v<Left> == static_num_rows_v<Right> &&
             (array_arithmetic_detail::class_operand<Left> ||
              array_arithmetic_detail::class_operand<Right>) &&
             !std::derived_from<std::remove_cvref_t<Left>, FieldBase> &&
             !std::derived_from<std::remove_cvref_t<Right>, FieldBase>)
GENDIL_HOST_DEVICE
auto operator*(const Left& left, const Right& right)
{
   using LeftValue = matrix_value_t<Left>;
   using RightValue = matrix_value_t<Right>;
   using ResultValue = decltype(std::declval<LeftValue>() * std::declval<RightValue>());
   constexpr Integer Rows = static_num_rows_v<Left>;
   constexpr Integer Inner = static_num_cols_v<Left>;
   constexpr Integer Columns = static_num_cols_v<Right>;

   SerialRecursiveArray<ResultValue, Rows, Columns> result;
   ConstexprLoop<Rows, Columns>([&](auto row, auto column)
   {
      const auto i = static_cast<GlobalIndex>(row);
      const auto j = static_cast<GlobalIndex>(column);
      ResultValue sum{};
      ConstexprLoop<Inner>([&](auto index)
      {
         const auto k = static_cast<GlobalIndex>(index);
         sum += static_cast<LeftValue>(GetValue(left, i, k)) *
                static_cast<RightValue>(GetValue(right, k, j));
      });
      result(i, j) = sum;
   });
   return result;
}

} // namespace gendil
