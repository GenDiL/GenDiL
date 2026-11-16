// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/arrayarithmetic.hpp"

namespace gendil
{

// Scale any readable static vector into owning storage. Native recursive-array
// overloads retain their existing arithmetic and custom-element contracts.
template<class Scalar, StaticVector Vector>
   requires (std::is_arithmetic_v<Scalar> &&
             array_arithmetic_detail::class_operand<Vector> &&
             !array_arithmetic_detail::is_recursive_array<std::remove_cvref_t<Vector>>)
GENDIL_HOST_DEVICE
constexpr auto operator*(const Scalar& scalar, const Vector& vector)
{
   using Entry = std::remove_cvref_t<decltype(
      std::declval<Scalar>() * std::declval<vector_value_t<Vector>>())>;
   using Result = array_arithmetic_detail::result_t<Entry, Vector>;
   return array_arithmetic_detail::ApplyEntrywise<Result>(
      vector, [&](auto value) { return scalar * value; });
}

template<StaticVector Vector, class Scalar>
   requires (std::is_arithmetic_v<Scalar> &&
             array_arithmetic_detail::class_operand<Vector> &&
             !array_arithmetic_detail::is_recursive_array<std::remove_cvref_t<Vector>>)
GENDIL_HOST_DEVICE
constexpr auto operator*(const Vector& vector, const Scalar& scalar)
{
   using Entry = std::remove_cvref_t<decltype(
      std::declval<vector_value_t<Vector>>() * std::declval<Scalar>())>;
   using Result = array_arithmetic_detail::result_t<Entry, Vector>;
   return array_arithmetic_detail::ApplyEntrywise<Result>(
      vector, [&](auto value) { return value * scalar; });
}

} // namespace gendil
