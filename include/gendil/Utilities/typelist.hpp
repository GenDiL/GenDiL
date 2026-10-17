// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <type_traits>

namespace gendil
{
//-------------------------------------------
// Tiny typelist utilities
//-------------------------------------------
template<class... Ts> struct type_list {};

template<class A, class B> struct concat;
template<class... As, class... Bs>
struct concat<type_list<As...>, type_list<Bs...>> { using type = type_list<As..., Bs...>; };
template<class A, class B>
using concat_t = typename concat<A,B>::type;

template<class T, class List> struct contains;
template<class T> struct contains<T, type_list<>> : std::false_type {};
template<class T, class U, class... Rest>
struct contains<T, type_list<U, Rest...>>
  : std::bool_constant<std::is_same_v<T,U> || contains<T, type_list<Rest...>>::value> {};

template<class List> struct unique;
template<> struct unique<type_list<>> { using type = type_list<>; };
template<class T, class... Rest>
struct unique<type_list<T, Rest...>> {
  using tail_u = typename unique<type_list<Rest...>>::type;
  using type = std::conditional_t<
    contains<T, tail_u>::value,
    tail_u,
    concat_t<type_list<T>, tail_u>
  >;
};
template<class List>
using unique_t = typename unique<List>::type;

template<class... Lists>
struct concat_many;

template<>
struct concat_many<> { using type = type_list<>; };

template<class List>
struct concat_many<List> { using type = List; };

template<class First, class Second, class... Rest>
struct concat_many<First, Second, Rest...>
{
   using type = typename concat_many<concat_t<First, Second>, Rest...>::type;
};

template<class... Lists>
using concat_many_t = typename concat_many<Lists...>::type;

} // namespace gendil
