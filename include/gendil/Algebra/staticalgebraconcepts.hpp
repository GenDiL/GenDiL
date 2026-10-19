// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include <array>
#include <span>
#include <type_traits>
#include <utility>
#include <concepts>

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

namespace static_algebra_detail
{
template<class T>
concept ArithmeticScalar = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template<class T>
concept HasArithmeticValueType = requires {
   typename T::value_type;
   requires ArithmeticScalar<typename T::value_type>;
};

template<class T>
concept HasArithmeticElementType = requires {
   typename T::element_type;
   requires ArithmeticScalar<typename T::element_type>;
};

template<class T>
using arithmetic_type = std::conditional_t<ArithmeticScalar<T>,
                                         std::remove_cvref_t<T>, void>;

// Scalar metadata describes readable proxies. Without it, infer from the
// const accessor, unwrapping RecursiveArray's rank-zero scalar wrapper.
template<class R, bool Matrix>
auto ScalarValueType()
{
   using Container = std::remove_cvref_t<R>;
   if constexpr (HasArithmeticValueType<Container>)
      return std::type_identity<std::remove_cvref_t<typename Container::value_type>>{};
   else if constexpr (HasArithmeticElementType<Container>)
      return std::type_identity<std::remove_cvref_t<typename Container::element_type>>{};
   // Check dimensions before probing access: an unrelated variadic callable
   // may have a deduced return type whose body is invalid for index arguments.
   else if constexpr (Matrix && static_num_rows_v<R> != std::dynamic_extent &&
                      static_num_cols_v<R> != std::dynamic_extent)
   {
      if constexpr (requires(const std::remove_reference_t<R>& r) {
                       GetValue(r, GlobalIndex{}, GlobalIndex{});
                    })
      {
         using Entry = typename vector_element_type<std::remove_cvref_t<decltype(
            GetValue(std::declval<const std::remove_reference_t<R>&>(),
                          GlobalIndex{}, GlobalIndex{}))>>::type;
         return std::type_identity<arithmetic_type<Entry>>{};
      }
      else
         return std::type_identity<void>{};
   }
   else if constexpr (!Matrix && static_extent_v<R> != std::dynamic_extent &&
                      static_num_rows_v<R> == std::dynamic_extent &&
                      static_num_cols_v<R> == std::dynamic_extent)
   {
      if constexpr (requires(const std::remove_reference_t<R>& r) {
                       GetValue(r, GlobalIndex{});
                    })
      {
         using Entry = typename vector_element_type<std::remove_cvref_t<decltype(
            GetValue(std::declval<const std::remove_reference_t<R>&>(),
                          GlobalIndex{}))>>::type;
         return std::type_identity<arithmetic_type<Entry>>{};
      }
      else
         return std::type_identity<void>{};
   }
   else
      return std::type_identity<void>{};
}
} // namespace static_algebra_detail

// The scalar represented by an entry, or void when no arithmetic type can be
// inferred. Metadata takes precedence: value_type, then element_type.
template<class V>
using vector_value_t = typename decltype(
   static_algebra_detail::ScalarValueType<V, false>())::type;

template<class M>
using matrix_value_t = typename decltype(
   static_algebra_detail::ScalarValueType<M, true>())::type;

// A fixed-size, const-readable vector of arithmetic scalars. T is a conversion
// target, not a computation type; writes and arithmetic have separate contracts.
template<class V, class T = vector_value_t<V>>
concept StaticVector =
   (static_extent_v<V> != std::dynamic_extent) &&
   (static_num_rows_v<V> == std::dynamic_extent) &&
   (static_num_cols_v<V> == std::dynamic_extent) &&
   static_algebra_detail::ArithmeticScalar<T> &&
   requires(const std::remove_reference_t<V>& v, GlobalIndex i) {
      { GetValue(v, i) } -> std::convertible_to<T>;
   };

// A fixed-size, const-readable matrix with scalar entries. Row containers are
// not scalar entries, so higher-rank arrays do not satisfy this contract.
template<class M, class T = matrix_value_t<M>>
concept StaticMatrix =
   (static_num_rows_v<M> != std::dynamic_extent) &&
   (static_num_cols_v<M> != std::dynamic_extent) &&
   static_algebra_detail::ArithmeticScalar<T> &&
   requires(const std::remove_reference_t<M>& m, GlobalIndex i, GlobalIndex j) {
      { GetValue(m, i, j) } -> std::convertible_to<T>;
   };

} // namespace gendil
