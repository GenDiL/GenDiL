// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Algebra/accessors.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

/// Negate a value using its native unary minus, or negate each scalar entry of
/// a readable static vector or matrix into owning array storage. Fallback
/// entries preserve ordinary unary-minus promotions, including short to int.
template<class Value>
   requires (requires(Value&& value) { -std::forward<Value>(value); } ||
             StaticVector<Value> || StaticMatrix<Value>)
GENDIL_HOST_DEVICE constexpr auto NegateValue(Value&& value)
{
   if constexpr (requires { -std::forward<Value>(value); })
   {
      return -std::forward<Value>(value);
   }
   else if constexpr (StaticMatrix<Value>)
   {
      using Scalar = matrix_value_t<Value>;
      using Entry = std::remove_cvref_t<decltype(-std::declval<Scalar>())>;
      constexpr auto Rows = static_num_rows_v<Value>;
      constexpr auto Columns = static_num_cols_v<Value>;
      std::array<std::array<Entry, Columns>, Rows> result{};
      const auto& entries = value;
      for (std::size_t i = 0; i < Rows; ++i)
         for (std::size_t j = 0; j < Columns; ++j)
            result[i][j] = -static_cast<Scalar>(GetValue(entries, i, j));
      return result;
   }
   else
   {
      using Scalar = vector_value_t<Value>;
      using Entry = std::remove_cvref_t<decltype(-std::declval<Scalar>())>;
      std::array<Entry, static_extent_v<Value>> result{};
      const auto& entries = value;
      for (std::size_t i = 0; i < result.size(); ++i)
         result[i] = -static_cast<Scalar>(GetValue(entries, i));
      return result;
   }
}

} // namespace gendil
