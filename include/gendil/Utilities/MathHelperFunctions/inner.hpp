// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <type_traits>
#include <utility>

namespace gendil
{

// Value-level inner product (Frobenius) for rank-2 SerialRecursiveArray
template<typename T, Integer M, Integer N>
GENDIL_HOST_DEVICE
T Inner(const SerialRecursiveArray<T, M, N>& A, const SerialRecursiveArray<T, M, N>& B)
{
   T result = T{0};
   for (Integer i = 0; i < M; ++i) {
      for (Integer j = 0; j < N; ++j) {
         result += A(i, j) * B(i, j);
      }
   }
   return result;
}

// Mixed matrix storage contracts matching entries without transposing either
// operand. Arithmetic products determine the accumulation type.
template<typename L, typename R, Integer M, Integer N>
   requires (std::is_arithmetic_v<L> && std::is_arithmetic_v<R>)
GENDIL_HOST_DEVICE
auto Inner(const SerialRecursiveArray<L, M, N>& lhs,
           const std::array<std::array<R, N>, M>& rhs)
   -> std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>
{
   using Result = std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>;
   Result result{0};
   for (Integer i = 0; i < M; ++i)
      for (Integer j = 0; j < N; ++j) { result += lhs(i, j) * rhs[i][j]; }
   return result;
}

template<typename L, typename R, Integer M, Integer N>
   requires (std::is_arithmetic_v<L> && std::is_arithmetic_v<R>)
GENDIL_HOST_DEVICE
auto Inner(const std::array<std::array<L, N>, M>& lhs,
           const SerialRecursiveArray<R, M, N>& rhs)
   -> std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>
{
   using Result = std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>;
   Result result{0};
   for (Integer i = 0; i < M; ++i)
      for (Integer j = 0; j < N; ++j) { result += lhs[i][j] * rhs(i, j); }
   return result;
}

template<typename L, typename R, Integer M, Integer N>
   requires (std::is_arithmetic_v<L> && std::is_arithmetic_v<R>)
GENDIL_HOST_DEVICE
constexpr auto Inner(const std::array<std::array<L, N>, M>& lhs,
                     const std::array<std::array<R, N>, M>& rhs)
   -> std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>
{
   using Result = std::remove_cvref_t<decltype(std::declval<L>() * std::declval<R>())>;
   Result result{0};
   for (Integer i = 0; i < M; ++i)
      for (Integer j = 0; j < N; ++j) { result += lhs[i][j] * rhs[i][j]; }
   return result;
}

} // namespace gendil
