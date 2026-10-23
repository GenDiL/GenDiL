// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

// Named requirement carriers shared by symbolic analysis and operator requirement lowering.
// This header does not depend on weak-form expression definitions.
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/typelist.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/operatormask.hpp"
#include <type_traits>

namespace gendil
{

// ---------------------------------------------------------------------------
// Named-field requirement model
// ---------------------------------------------------------------------------

enum class NamedFieldProvenance : unsigned
{
   None                    = 0u,
   CoefficientInput        = 1u,
   FiniteElementExpression = 2u,
   ActiveTrial             = 4u,
   ActiveTest              = 8u
};

constexpr NamedFieldProvenance operator+(NamedFieldProvenance a, NamedFieldProvenance b)
{
   return NamedFieldProvenance(unsigned(a) | unsigned(b));
}

constexpr bool has_provenance(NamedFieldProvenance provenance, NamedFieldProvenance query)
{
   return (unsigned(provenance) & unsigned(query)) != 0u;
}

template<StaticString Name, OperatorMask Mask, NamedFieldProvenance Provenance>
struct NamedFieldRequirement
{
   static constexpr auto name = Name;
   static constexpr OperatorMask mask = Mask;
   static constexpr NamedFieldProvenance provenance = Provenance;
};

template<class Req>
using named_field_requirement_name_t = NameTag<Req::name>;

template<class LHS, class RHS>
struct merge_named_field_requirement
{
   static_assert(LHS::name == RHS::name,
      "Named field requirements can only be merged when they have the same name.");

   using type = NamedFieldRequirement<
      LHS::name,
      LHS::mask + RHS::mask,
      LHS::provenance + RHS::provenance>;
};

template<class LHS, class RHS>
using merge_named_field_requirement_t =
   typename merge_named_field_requirement<LHS, RHS>::type;

template<class Req, class List>
struct merge_requirement_into_list;

template<class Req>
struct merge_requirement_into_list<Req, type_list<>>
{
   using type = type_list<Req>;
};

template<bool SameName, class Req, class Head, class TailList>
struct merge_requirement_into_nonempty_list;

template<class Req, class Head, class... Tail>
struct merge_requirement_into_nonempty_list<true, Req, Head, type_list<Tail...>>
{
   using type = type_list<merge_named_field_requirement_t<Req, Head>, Tail...>;
};

template<class Req, class Head, class... Tail>
struct merge_requirement_into_nonempty_list<false, Req, Head, type_list<Tail...>>
{
   using merged_tail = typename merge_requirement_into_list<Req, type_list<Tail...>>::type;
   using type = concat_t<type_list<Head>, merged_tail>;
};

template<class Req, class Head, class... Tail>
struct merge_requirement_into_list<Req, type_list<Head, Tail...>>
   : merge_requirement_into_nonempty_list<(Req::name == Head::name), Req, Head, type_list<Tail...>>
{};

template<class List>
struct merge_named_field_requirement_list;

template<>
struct merge_named_field_requirement_list<type_list<>>
{
   using type = type_list<>;
};

template<class Req, class... Rest>
struct merge_named_field_requirement_list<type_list<Req, Rest...>>
{
   using merged_rest = typename merge_named_field_requirement_list<type_list<Rest...>>::type;
   using type = typename merge_requirement_into_list<Req, merged_rest>::type;
};

template<class List>
using merge_named_field_requirement_list_t =
   typename merge_named_field_requirement_list<List>::type;

template<class List, NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance;

template<NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance<type_list<>, Provenance>
{
   using type = type_list<>;
};

template<class Req, class... Rest, NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance<type_list<Req, Rest...>, Provenance>
{
   using filtered_rest =
      typename filter_requirements_by_provenance<type_list<Rest...>, Provenance>::type;

   using type = std::conditional_t<
      has_provenance(Req::provenance, Provenance),
      concat_t<type_list<Req>, filtered_rest>,
      filtered_rest>;
};

template<class List, NamedFieldProvenance Provenance>
using filter_requirements_by_provenance_t =
   typename filter_requirements_by_provenance<List, Provenance>::type;

template<class List>
struct named_field_requirement_names;

template<>
struct named_field_requirement_names<type_list<>>
{
   using type = type_list<>;
};

template<class Req, class... Rest>
struct named_field_requirement_names<type_list<Req, Rest...>>
{
   using rest_names = typename named_field_requirement_names<type_list<Rest...>>::type;
   using type = unique_t<concat_t<type_list<NameTag<Req::name>>, rest_names>>;
};

template<class List>
using named_field_requirement_names_t =
   typename named_field_requirement_names<List>::type;

template<class List, StaticString Name>
struct contains_named_field_requirement;

template<StaticString Name>
struct contains_named_field_requirement<type_list<>, Name> : std::false_type {};

template<class Req, class... Rest, StaticString Name>
struct contains_named_field_requirement<type_list<Req, Rest...>, Name>
   : std::bool_constant<
        (Req::name == Name) ||
        contains_named_field_requirement<type_list<Rest...>, Name>::value>
{};

template<class List, StaticString Name>
inline constexpr bool contains_named_field_requirement_v =
   contains_named_field_requirement<List, Name>::value;

template<class List, StaticString Name>
struct find_named_field_requirement;

template<StaticString Name>
struct find_named_field_requirement<type_list<>, Name>
{
   using type = void;
};

template<bool SameName, class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty;

template<class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty<true, Req, RestList, Name>
{
   using type = Req;
};

template<class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty<false, Req, RestList, Name>
{
   using type = typename find_named_field_requirement<RestList, Name>::type;
};

template<class Req, class... Rest, StaticString Name>
struct find_named_field_requirement<type_list<Req, Rest...>, Name>
   : find_named_field_requirement_nonempty<(Req::name == Name), Req, type_list<Rest...>, Name>
{};

template<class List, StaticString Name>
using find_named_field_requirement_t =
   typename find_named_field_requirement<List, Name>::type;

template<class List>
struct is_empty_type_list : std::false_type {};

template<>
struct is_empty_type_list<type_list<>> : std::true_type {};

template<class List>
inline constexpr bool is_empty_type_list_v = is_empty_type_list<List>::value;

} // namespace gendil
