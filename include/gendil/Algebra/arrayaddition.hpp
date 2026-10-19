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

namespace array_addition_detail
{
template<class T>
inline constexpr bool is_array = false;

template<class T, std::size_t N>
inline constexpr bool is_array<std::array<T, N>> = true;

template<class T>
inline constexpr bool is_recursive_array = false;

template<class T, class... Dims>
inline constexpr bool is_recursive_array<RecursiveArray<T, Dims...>> = true;

template<class L, class R>
using scalar_sum_t = std::remove_cvref_t<decltype(
   std::declval<const L&>() + std::declval<const R&>())>;

// Only a nested std::array with exactly these extents exposes a scalar leaf.
template<class Array, std::size_t... Dims>
struct array_leaf {};

template<class T>
   requires (!is_array<T> && !is_recursive_array<T>)
struct array_leaf<T> { using type = T; };

template<class T, std::size_t N, std::size_t... Dims>
struct array_leaf<std::array<T, N>, N, Dims...>
   : array_leaf<std::remove_cv_t<T>, Dims...> {};

template<class Array, std::size_t... Dims>
using array_leaf_t = typename array_leaf<std::remove_cv_t<Array>, Dims...>::type;

template<class T>
GENDIL_HOST_DEVICE
constexpr const T& array_at(const T& value) { return value; }

template<class T, std::size_t N, class... Indices>
GENDIL_HOST_DEVICE
constexpr decltype(auto) array_at(const std::array<T, N>& value,
                                std::size_t first, Indices... rest)
{
   return array_at(value[first], rest...);
}
} // namespace array_addition_detail

// Mixed storage adds matching scalar entries and returns an owning recursive
// array in either operand order, as matrix-vector multiplication does.
template<class L, class R, std::size_t N, std::size_t... Dims>
GENDIL_HOST_DEVICE
auto operator+(const std::array<L, N>& lhs,
               const SerialRecursiveArray<R, N, Dims...>& rhs)
   -> SerialRecursiveArray<array_addition_detail::scalar_sum_t<
      array_addition_detail::array_leaf_t<L, Dims...>, R>, N, Dims...>
{
   using Value = array_addition_detail::scalar_sum_t<
      array_addition_detail::array_leaf_t<L, Dims...>, R>;
   SerialRecursiveArray<Value, N, Dims...> result;
   UnitLoop<N, Dims...>([&](auto... indices)
   {
      result(indices...) = array_addition_detail::array_at(lhs, indices...) +
                           rhs(indices...);
   });
   return result;
}

template<class L, class R, std::size_t N, std::size_t... Dims>
GENDIL_HOST_DEVICE
auto operator+(const SerialRecursiveArray<L, N, Dims...>& lhs,
               const std::array<R, N>& rhs)
   -> SerialRecursiveArray<array_addition_detail::scalar_sum_t<
      L, array_addition_detail::array_leaf_t<R, Dims...>>, N, Dims...>
{
   using Value = array_addition_detail::scalar_sum_t<
      L, array_addition_detail::array_leaf_t<R, Dims...>>;
   SerialRecursiveArray<Value, N, Dims...> result;
   UnitLoop<N, Dims...>([&](auto... indices)
   {
      result(indices...) = lhs(indices...) +
                           array_addition_detail::array_at(rhs, indices...);
   });
   return result;
}

namespace array_addition_detail
{
// Deduce nested results without invoking an array operator during its own
// constraint checking. Mismatched inner extents have no result type.
template<class L, class R, bool ArrayPair = is_array<L> && is_array<R>, class = void>
struct sum_result {};

template<class L, class R>
struct sum_result<L, R, false, std::void_t<scalar_sum_t<L, R>>>
{ using type = scalar_sum_t<L, R>; };

template<class L, class R, std::size_t N>
struct sum_result<std::array<L, N>, std::array<R, N>, true,
                  std::void_t<typename sum_result<
                     std::remove_cv_t<L>, std::remove_cv_t<R>>::type>>
{
   using type = std::array<typename sum_result<
      std::remove_cv_t<L>, std::remove_cv_t<R>>::type, N>;
};
} // namespace array_addition_detail

/** Add equally shaped arrays componentwise, including nested arrays.
 * The result owns its entries and preserves std::array storage. Entry types
 * follow their addition result. Outside gendil, use `using gendil::operator+`.
 * Include this header before defining templates that add standard arrays.
 */
template<class L, class R, std::size_t N>
GENDIL_HOST_DEVICE
constexpr auto operator+(const std::array<L, N>& lhs, const std::array<R, N>& rhs)
   -> typename array_addition_detail::sum_result<
      std::array<L, N>, std::array<R, N>>::type
{
   using Result = typename array_addition_detail::sum_result<
      std::array<L, N>, std::array<R, N>>::type;
   return [&]<std::size_t... I>(std::index_sequence<I...>)
   {
      return Result{(lhs[I] + rhs[I])...};
   }(std::make_index_sequence<N>{});
}

} // namespace gendil
