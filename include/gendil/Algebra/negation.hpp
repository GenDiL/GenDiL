// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/Algebra/arrayarithmetic.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

/// Negate a value using its native unary minus, or negate each scalar entry of
/// a readable static vector or matrix into owning storage. Standard arrays
/// retain standard storage; other fallback operands produce recursive arrays.
/// Entries preserve ordinary unary-minus promotions, including short to int.
template<class Value>
   requires (requires(Value&& value) { -std::forward<Value>(value); } ||
             StaticVector<Value> || StaticMatrix<Value>)
GENDIL_HOST_DEVICE constexpr auto NegateValue(Value&& value)
{
   if constexpr (requires { -std::forward<Value>(value); })
   {
      return -std::forward<Value>(value);
   }
   else
   {
      using Scalar = array_arithmetic_detail::static_value_t<Value>;
      using Entry = std::remove_cvref_t<decltype(-std::declval<Scalar>())>;
      using Result = array_arithmetic_detail::result_t<Entry, Value>;
      return array_arithmetic_detail::ApplyEntrywise<Result>(
         value, [](auto entry) { return -entry; });
   }
}

} // namespace gendil
