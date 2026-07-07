// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace gendil {

template<class T>
struct is_tuple : std::false_type {};

template<class... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template<class T>
inline constexpr bool is_tuple_v =
   is_tuple<std::remove_cvref_t<T>>::value;

template<class T>
struct tuple_size_or_zero : std::integral_constant<size_t, 0> {};

template<class... Ts>
struct tuple_size_or_zero<std::tuple<Ts...>>
   : std::integral_constant<size_t, sizeof...(Ts)> {};

template<class T>
inline constexpr size_t tuple_size_or_zero_v =
   tuple_size_or_zero<std::remove_cvref_t<T>>::value;

} // namespace gendil
