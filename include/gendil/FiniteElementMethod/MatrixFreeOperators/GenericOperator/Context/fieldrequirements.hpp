// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/typelist.hpp"

#include <type_traits>

namespace gendil
{

// Field access requirements describe named reads and writes without requiring
// finite-element expressions, quadrature, or a particular algorithm.
enum class FieldStorageRole { Input, Bound };
enum class FieldSide { Current, Minus, Plus };

template<StaticString Name, FieldStorageRole Role,
         FieldSide Side = FieldSide::Current>
struct FieldRead
{
   static constexpr auto name = Name;
   static constexpr auto storage_role = Role;
   static constexpr auto side = Side;
};

template<StaticString Name, FieldSide Side = FieldSide::Current>
struct FieldWrite
{
   static constexpr auto name = Name;
   static constexpr auto side = Side;
};

template<class Reads, class Writes>
struct FieldAccessRequirements
{
   using reads = Reads;
   using writes = Writes;
};

namespace field_requirements_detail
{
template<class List> struct ReadRolesCompatible;
template<> struct ReadRolesCompatible<type_list<>> : std::true_type {};
template<class First, class... Rest>
struct ReadRolesCompatible<type_list<First, Rest...>>
   : std::bool_constant<
      (((First::name != Rest::name) || (First::storage_role == Rest::storage_role)) && ...) &&
      ReadRolesCompatible<type_list<Rest...>>::value> {};

template<class List> struct bound_reads;
template<class... Fields>
struct bound_reads<type_list<Fields...>>
{
   using type = concat_many_t<std::conditional_t<
      Fields::storage_role == FieldStorageRole::Bound,
      type_list<Fields>, type_list<>>...>;
};
}

// Assembly and local probes bind persistent sources once, before supplying
// their active input and output tensors inside the executing kernel.
template<class Requirements>
using bound_only_requirements_t = FieldAccessRequirements<
   typename field_requirements_detail::bound_reads<typename Requirements::reads>::type,
   type_list<>>;

} // namespace gendil
