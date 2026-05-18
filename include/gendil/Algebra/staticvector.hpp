// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include <array>
#include <span>
#include <type_traits>
#include <utility>
#include <iterator>
#include <concepts>
#include <cstddef>

namespace gendil
{

template<class T>
struct static_extent : std::integral_constant<Integer, std::dynamic_extent> {};

template<class T, Integer N>
struct static_extent<T[N]> : std::integral_constant<Integer, N> {};

template<class T, Integer N>
struct static_extent<std::array<T, N>> : std::integral_constant<Integer, N> {};

// Rank-1 RecursiveArray (use actual underlying type, not alias)
template<class T, Integer N>
struct static_extent<RecursiveArray<T, StaticDimension<N>>> : std::integral_constant<Integer, N> {};

template<class T>
inline constexpr Integer static_extent_v =
   static_extent<std::remove_cvref_t<T>>::value;

// Static number of rows (SFINAE-safe with default)
template<class T>
struct static_num_rows : std::integral_constant<Integer, std::dynamic_extent> {};

// Specialization for actual RecursiveArray type with StaticDimension
template<class T, Integer M, Integer N>
struct static_num_rows<RecursiveArray<T, StaticDimension<M>, StaticDimension<N>>> : std::integral_constant<Integer, M> {};

template<class T, Integer M, Integer N>
struct static_num_rows<std::array<std::array<T, N>, M>> : std::integral_constant<Integer, M> {};

template<class T>
inline constexpr Integer static_num_rows_v = static_num_rows<std::remove_cvref_t<T>>::value;

// Static number of columns (SFINAE-safe with default)
template<class T>
struct static_num_cols : std::integral_constant<Integer, std::dynamic_extent> {};

// Specialization for actual RecursiveArray type with StaticDimension
template<class T, Integer M, Integer N>
struct static_num_cols<RecursiveArray<T, StaticDimension<M>, StaticDimension<N>>> : std::integral_constant<Integer, N> {};

template<class T, Integer M, Integer N>
struct static_num_cols<std::array<std::array<T, N>, M>> : std::integral_constant<Integer, N> {};

template<class T>
inline constexpr Integer static_num_cols_v = static_num_cols<std::remove_cvref_t<T>>::value;

// Helper to get element type, handling RecursiveArray rank-0 wrapper
template<class T>
struct vector_element_type { using type = T; };

template<class T>
struct vector_element_type<RecursiveArray<T>> { using type = T; };

template<class R>
using vector_value_t =
   typename vector_element_type<
      std::remove_cvref_t<decltype(std::declval<R&>()[0])>
   >::type;

template<class R, class T>
concept static_vector =
   (static_extent_v<std::remove_cvref_t<R>> != std::dynamic_extent) &&
   std::same_as<vector_value_t<R>, T> &&
   requires(R&& r, GlobalIndex i) {
      //   { std::size(r) } -> std::convertible_to<Integer>;
      { r[i] };
   };

template < typename T, static_vector<T> VecType >
GENDIL_HOST_DEVICE
constexpr auto operator*( const T & scalar, const VecType & vec )
{
   VecType result;
   ConstexprLoop< static_extent_v<VecType> >( [&] ( auto i )
   {
      result[i] = scalar * vec[i];
   });
   return result;
}

} // namespace gendil
