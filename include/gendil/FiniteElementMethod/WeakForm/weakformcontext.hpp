// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

template<class T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// -----------------------------------------------------------------------------
// Key tags
// -----------------------------------------------------------------------------

struct WeakFormKeyTag {};
struct FieldKeyTag : WeakFormKeyTag {};
struct FiniteElementFieldKeyTag : WeakFormKeyTag {};
struct DomainKeyTag : WeakFormKeyTag {};

// -----------------------------------------------------------------------------
// Keys
// -----------------------------------------------------------------------------

template<StaticString Name>
struct FieldKey : FieldKeyTag
{
   static constexpr auto name = Name;
};

template<StaticString Name>
struct FiniteElementFieldKey : FiniteElementFieldKeyTag
{
   static constexpr auto name = Name;
};

template<StaticString Name>
struct DomainKey : DomainKeyTag
{
   static constexpr auto name = Name;
};

// -----------------------------------------------------------------------------
// Traits
// No partial specialization on FieldKey<N>/DomainKey<N>/... anymore.
// -----------------------------------------------------------------------------

template<class Key>
struct is_field_key
   : std::bool_constant<std::is_base_of_v<FieldKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_field_key_v = is_field_key<Key>::value;

template<class Key>
struct is_fe_field_key
   : std::bool_constant<std::is_base_of_v<FiniteElementFieldKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_fe_field_key_v = is_fe_field_key<Key>::value;

template<class Key>
struct is_domain_key
   : std::bool_constant<std::is_base_of_v<DomainKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_domain_key_v = is_domain_key<Key>::value;

template<class Key>
struct is_weak_form_context_key
   : std::bool_constant<
        std::is_base_of_v<WeakFormKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_weak_form_context_key_v =
   is_weak_form_context_key<Key>::value;

// -----------------------------------------------------------------------------
// Entry/key introspection
// -----------------------------------------------------------------------------

template<class E, class = void>
struct has_key_type : std::false_type {};

template<class E>
struct has_key_type<E, std::void_t<typename uncvref_t<E>::key_type>> : std::true_type {};

template<class E>
inline constexpr bool has_key_type_v = has_key_type<E>::value;

template<class E, class = void>
struct entry_key
{
   using type = void;
};

template<class E>
struct entry_key<E, std::void_t<typename uncvref_t<E>::key_type>>
{
   using type = uncvref_t<typename uncvref_t<E>::key_type>;
};

template<class E>
using entry_key_t = typename entry_key<E>::type;

// -----------------------------------------------------------------------------
// Views / factories
// -----------------------------------------------------------------------------

template<class T>
constexpr decltype(auto) to_view(T&& x) noexcept
{
   return std::forward<T>(x);
}

template<StaticString Name, class T>
constexpr auto MakeField(T&& x)
{
   auto v = to_view(std::forward<T>(x));
   using V = uncvref_t<decltype(v)>;
   return Entry<FieldKey<Name>, V>{ static_cast<V>(v) };
}

template<class SpaceView, class DofsView>
struct FiniteElementFieldView
{
   SpaceView space;
   DofsView  dofs;
};

template<StaticString Name, class FiniteElementSpace, class Field>
constexpr auto MakeFiniteElementField(FiniteElementSpace&& fe_space, Field&& u)
{
   auto fes_v = to_view(std::forward<FiniteElementSpace>(fe_space));
   using FESV = uncvref_t<decltype(fes_v)>;

   auto u_v = to_view(std::forward<Field>(u));
   using UV = uncvref_t<decltype(u_v)>;

   using View = FiniteElementFieldView<FESV, UV>;

   return Entry<FiniteElementFieldKey<Name>, View>{
      View{ static_cast<FESV>(fes_v), static_cast<UV>(u_v) }
   };
}

template<StaticString Name, class FiniteElementSpace>
constexpr auto MakeTrialField(FiniteElementSpace&& fe_space)
{
  return MakeFiniteElementField<Name>(std::forward<FiniteElementSpace>(fe_space), Empty{});
}

template<StaticString Name, class FiniteElementSpace>
constexpr auto MakeTestField(FiniteElementSpace&& fe_space)
{
  return MakeFiniteElementField<Name>(std::forward<FiniteElementSpace>(fe_space), Empty{});
}

template<StaticString Name, class T>
constexpr auto MakeDomain(T&& x)
{
   auto v = to_view(std::forward<T>(x));
   using V = uncvref_t<decltype(v)>;
   return Entry<DomainKey<Name>, V>{ static_cast<V>(v) };
}

// -----------------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------------

template<class DomainsMap, class FEFieldsMap, class FieldsMap>
struct WeakFormContext
{
   DomainsMap  domains;
   FEFieldsMap fe_fields;
   FieldsMap   fields;

   template<StaticString Name>
   static consteval bool has_domain()
   {
      return DomainsMap::template contains<DomainKey<Name>>();
   }

   template<StaticString Name>
   static consteval bool has_fe_field()
   {
      return FEFieldsMap::template contains<FiniteElementFieldKey<Name>>();
   }

   template<StaticString Name>
   static consteval bool has_field()
   {
      return FieldsMap::template contains<FieldKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) domain() &
   {
      static_assert(has_domain<Name>(), "WeakFormContext: missing required domain.");
      return domains.template get<DomainKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) domain() const &
   {
      static_assert(has_domain<Name>(), "WeakFormContext: missing required domain.");
      return domains.template get<DomainKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) fe_field() &
   {
      static_assert(has_fe_field<Name>(), "WeakFormContext: missing required finite element field.");
      return fe_fields.template get<FiniteElementFieldKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) fe_field() const &
   {
      static_assert(has_fe_field<Name>(), "WeakFormContext: missing required finite element field.");
      return fe_fields.template get<FiniteElementFieldKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) field() &
   {
      static_assert(has_field<Name>(), "WeakFormContext: missing required field.");
      return fields.template get<FieldKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) field() const &
   {
      static_assert(has_field<Name>(), "WeakFormContext: missing required field.");
      return fields.template get<FieldKey<Name>>();
   }
};

// -----------------------------------------------------------------------------
// Uniqueness
// -----------------------------------------------------------------------------

template<class... Ts>
struct all_unique;

template<>
struct all_unique<> : std::true_type {};

template<class T, class... Rest>
struct all_unique<T, Rest...>
   : std::bool_constant<
        (!std::is_same_v<T, Rest> && ...) &&
        all_unique<Rest...>::value> {};

// -----------------------------------------------------------------------------
// Partition helpers
// -----------------------------------------------------------------------------

template<class E>
constexpr auto as_field_tuple(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_field_key_v<Key>) {
      return std::tuple{ std::forward<E>(e) };
   } else {
      return std::tuple{};
   }
}

template<class E>
constexpr auto as_fe_field_tuple(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_fe_field_key_v<Key>) {
    return std::tuple{ std::forward<E>(e) };
  } else {
    return std::tuple{};
  }
}

template<class E>
constexpr auto as_domain_tuple(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_domain_key_v<Key>) {
      return std::tuple{ std::forward<E>(e) };
   } else {
      return std::tuple{};
   }
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template<class... Entries>
constexpr auto MakeWeakFormContext(Entries&&... es)
{
   static_assert(
      (has_key_type_v<Entries> && ...),
      "MakeWeakFormContext: every argument must be an Entry-like type exposing key_type.");

   static_assert(
      (is_weak_form_context_key_v<entry_key_t<Entries>> && ...),
      "MakeWeakFormContext: keys must derive from WeakFormKeyTag.");

   static_assert(
      all_unique<entry_key_t<Entries>...>::value,
                  "MakeWeakFormContext: duplicate key provided.");

   auto fields_t    = std::tuple_cat(as_field_tuple(std::forward<Entries>(es))...);
   auto domains_t   = std::tuple_cat(as_domain_tuple(std::forward<Entries>(es))...);
   auto fe_fields_t = std::tuple_cat(as_fe_field_tuple(std::forward<Entries>(es))...);

   auto fields_map    = tuple_to_map(std::move(fields_t));
   auto domains_map   = tuple_to_map(std::move(domains_t));
   auto fe_fields_map = tuple_to_map(std::move(fe_fields_t));

   return WeakFormContext<
      decltype(domains_map),
      decltype(fe_fields_map),
      decltype(fields_map)>{
         std::move(domains_map),
         std::move(fe_fields_map),
         std::move(fields_map)
   };
}

} // namespace gendil
