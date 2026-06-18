// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

// This header is the compile-time registry used by weak forms and matrix-free
// operators. User-facing factory functions create typed Entry objects, and
// MakeWeakFormContext normalizes those entries into separate maps for:
//
// - ordinary coefficient fields;
// - finite-element fields with their backing DoF storage;
// - cell/local-facet integration topology;
// - global interior-face integration topology;
// - global boundary-face integration topology.
//
// The important architectural rule is that public integration-domain entries
// are normalized once, before map construction. The as_*_tuple helpers near the
// bottom of the file are deliberately dumb filters over already-normalized
// entries; they should not grow topology-expansion logic.

template<class T>
using uncvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

// -----------------------------------------------------------------------------
// Key tags
// -----------------------------------------------------------------------------

// All context keys derive from WeakFormKeyTag so MakeWeakFormContext can reject
// accidental non-context entries early. The derived tags partition entries into
// the independent maps stored by WeakFormContext.
struct WeakFormKeyTag {};
struct FieldKeyTag : WeakFormKeyTag {};
struct FiniteElementFieldKeyTag : WeakFormKeyTag {};
struct DomainKeyTag : WeakFormKeyTag {};
struct InteriorFaceDomainKeyTag : WeakFormKeyTag {};
struct BoundaryFaceDomainKeyTag : WeakFormKeyTag {};
struct IntegrationDomainKeyTag : WeakFormKeyTag {};

// -----------------------------------------------------------------------------
// Keys
// -----------------------------------------------------------------------------

template<StaticString Name>
struct FieldKey : FieldKeyTag
{
   // Named non-FE coefficient field, e.g. a scalar parameter or q-point data.
   static constexpr auto name = Name;
};

template<StaticString Name>
struct FiniteElementFieldKey : FiniteElementFieldKeyTag
{
   // Named finite-element field. The stored value carries both a finite
   // element space binding and its algebraic backing storage.
   static constexpr auto name = Name;
};

template<StaticString Name>
struct DomainKey : DomainKeyTag
{
   // Internal normalized cell integration-topology key. Public code should
   // register full parent domains with MakeIntegrationDomain<Name>(...);
   // restricted-context adapters may still create DomainKey entries directly.
   static constexpr auto name = Name;
};

template<StaticString Name>
struct InteriorFaceDomainKey : InteriorFaceDomainKeyTag
{
   // Internal normalized global interior-face topology key. Future public
   // restricted-domain APIs should be parent-domain based, not direct
   // InteriorFaceDomainKey construction.
   static constexpr auto name = Name;
};

template<StaticString Name>
struct BoundaryFaceDomainKey : BoundaryFaceDomainKeyTag
{
   // Internal normalized global boundary-face topology key. Future public
   // restricted-domain APIs should be parent-domain based, not direct
   // BoundaryFaceDomainKey construction.
   static constexpr auto name = Name;
};

template<StaticString Name>
struct IntegrationDomainKey : IntegrationDomainKeyTag
{
   // Public normalization key produced by MakeIntegrationDomain<Name>(...).
   // It is expanded into internal cell/face topology keys before maps are
   // built.
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
struct is_interior_face_domain_key
   : std::bool_constant<std::is_base_of_v<InteriorFaceDomainKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_interior_face_domain_key_v =
   is_interior_face_domain_key<Key>::value;

template<class Key>
struct is_boundary_face_domain_key
   : std::bool_constant<std::is_base_of_v<BoundaryFaceDomainKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_boundary_face_domain_key_v =
   is_boundary_face_domain_key<Key>::value;

template<class Key>
struct is_integration_domain_key
   : std::bool_constant<std::is_base_of_v<IntegrationDomainKeyTag, uncvref_t<Key>>> {};

template<class Key>
inline constexpr bool is_integration_domain_key_v =
   is_integration_domain_key<Key>::value;

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
struct has_value_type : std::false_type {};

template<class E>
struct has_value_type<E, std::void_t<typename uncvref_t<E>::value_type>>
   : std::true_type {};

template<class E>
inline constexpr bool has_value_type_v = has_value_type<E>::value;

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

// Factory summary:
// - MakeField<Name>(x) stores ordinary non-FE data.
// - MakeFiniteElementField<Name>(space, dofs) stores a named FE field.
// - MakeTrialField/MakeTestField register empty FE-field entries used for the
//   active trial/test space contract.
// - MakeIntegrationDomain<Name>(space) stores a public topology request that is
//   normalized later by MakeWeakFormContext.

template<class T>
constexpr decltype(auto) to_view(T&& x) noexcept
{
   // GenDiL mesh/FES/field objects stored in weak-form contexts are
   // lightweight view-like values. The context stores them by value for
   // device/lambda capture. Ownership and destructor-based resource
   // management remain outside this context layer.
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
   // `space` describes how the DoF storage is interpreted. For mixed spaces,
   // restricted contexts select the compatible homogeneous or face-space batch
   // before operator-context construction.
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
constexpr auto MakeIntegrationDomain(T&& x)
{
   // Homogeneous finite element spaces are the singleton cell/local-facet case.
   // Mixed/domain-decomposed finite element spaces may also provide global face
   // topology. Homogeneous global face execution must be modeled with a
   // singleton domain-decomposed space containing explicit face FES entries.
   auto v = to_view(std::forward<T>(x));
   using V = uncvref_t<decltype(v)>;

   if constexpr (is_mixed_finite_element_space_v<V> ||
                 is_cell_finite_element_space_v<V>)
   {
      using Domain = IntegrationDomain<V>;
      return Entry<IntegrationDomainKey<Name>, Domain>{
         Domain{ static_cast<V>(v) } };
   }
   else
   {
      static_assert(
         dependent_false_v<V>,
         "MakeIntegrationDomain<Name>: expected a homogeneous "
         "FiniteElementSpace or a MixedFiniteElementSpace built with "
         "MakeMixedFiniteElementSpace(...).");
   }
}

// -----------------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------------

// WeakFormContext stores normalized entries in separate maps. This keeps
// lookups simple and name-specific: Cells<Name> query domains, global
// InteriorFacets<Name> query interior_face_domains, and global
// BoundaryFacets<Name> query boundary_face_domains.
template<
   class DomainsMap,
   class InteriorFaceDomainsMap,
   class BoundaryFaceDomainsMap,
   class FEFieldsMap,
   class FieldsMap>
struct WeakFormContext
{
   DomainsMap             domains;
   InteriorFaceDomainsMap interior_face_domains;
   BoundaryFaceDomainsMap boundary_face_domains;
   FEFieldsMap            fe_fields;
   FieldsMap              fields;

   template<StaticString Name>
   static consteval bool has_domain()
   {
      return DomainsMap::template contains<DomainKey<Name>>();
   }

   template<StaticString Name>
   static consteval bool has_interior_face_domain()
   {
      return InteriorFaceDomainsMap::template contains<InteriorFaceDomainKey<Name>>();
   }

   static consteval bool has_any_interior_face_domain()
   {
      return std::tuple_size_v<
         decltype(std::declval<InteriorFaceDomainsMap>().entries)> != 0;
   }

   template<StaticString Name>
   static consteval bool has_boundary_face_domain()
   {
      return BoundaryFaceDomainsMap::template contains<BoundaryFaceDomainKey<Name>>();
   }

   static consteval bool has_any_boundary_face_domain()
   {
      return std::tuple_size_v<
         decltype(std::declval<BoundaryFaceDomainsMap>().entries)> != 0;
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
   constexpr decltype(auto) interior_face_domain() &
   {
      static_assert(
         has_interior_face_domain<Name>(),
         "WeakFormContext: missing required interior face domain.");
      return interior_face_domains.template get<InteriorFaceDomainKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) interior_face_domain() const &
   {
      static_assert(
         has_interior_face_domain<Name>(),
         "WeakFormContext: missing required interior face domain.");
      return interior_face_domains.template get<InteriorFaceDomainKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) boundary_face_domain() &
   {
      static_assert(
         has_boundary_face_domain<Name>(),
         "WeakFormContext: missing required boundary face domain.");
      return boundary_face_domains.template get<BoundaryFaceDomainKey<Name>>();
   }

   template<StaticString Name>
   constexpr decltype(auto) boundary_face_domain() const &
   {
      static_assert(
         has_boundary_face_domain<Name>(),
         "WeakFormContext: missing required boundary face domain.");
      return boundary_face_domains.template get<BoundaryFaceDomainKey<Name>>();
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

template<class T>
struct is_empty_finite_element_field_value : std::false_type {};

template<class SpaceView>
struct is_empty_finite_element_field_value<
   FiniteElementFieldView<SpaceView, Empty>> : std::true_type {};

template<class T>
inline constexpr bool is_empty_finite_element_field_value_v =
   is_empty_finite_element_field_value<uncvref_t<T>>::value;

// Duplicate-key policy after normalization:
// - ordinary duplicate keys are rejected;
// - explicit identical trial/test registrations are allowed when both entries
//   are empty FE fields with the same value type.
//
// The type-identity check is intentional for this case because the static field
// name participates in the key/type machinery.
template<
   class LHS,
   class RHS,
   bool HasEntryShape =
      has_key_type_v<LHS> &&
      has_key_type_v<RHS> &&
      has_value_type_v<LHS> &&
      has_value_type_v<RHS>>
struct context_entry_pair_is_compatible : std::true_type {};

template<class LHS, class RHS>
struct context_entry_pair_is_compatible<LHS, RHS, true>
{
   using LKey = entry_key_t<LHS>;
   using RKey = entry_key_t<RHS>;
   using L = uncvref_t<LHS>;
   using R = uncvref_t<RHS>;

   static constexpr bool value =
      !std::is_same_v<LKey, RKey> ||
      (
         // Duplicate empty FE-field entries with identical value type are
         // allowed to support explicit identical trial/test registration. The
         // static field name participates in the key/type, so type identity is
         // the intended check here.
         is_fe_field_key_v<LKey> &&
         std::is_same_v<typename L::value_type, typename R::value_type> &&
         is_empty_finite_element_field_value_v<typename L::value_type>
      );
};

template<class... Entries>
struct context_entries_are_compatible : std::true_type {};

template<class FirstEntry, class... RestEntries>
struct context_entries_are_compatible<FirstEntry, RestEntries...>
   : std::bool_constant<
        (context_entry_pair_is_compatible<FirstEntry, RestEntries>::value && ...) &&
        context_entries_are_compatible<RestEntries...>::value> {};

template<class Tuple>
struct context_entry_tuple_is_compatible;

template<class... Entries>
struct context_entry_tuple_is_compatible<std::tuple<Entries...>>
   : context_entries_are_compatible<Entries...> {};

// -----------------------------------------------------------------------------
// Context entry normalization helpers
// -----------------------------------------------------------------------------

// The normalization pipeline is intentionally two-stage:
//
// 1. ExpandWeakFormContextEntry expands public integration-domain entries into
//    internal cell/interior-face/boundary-face topology entries.
// 2. The as_*_tuple helpers below only filter those already-normalized entries
//    into the final maps.
//
// This keeps topology policy centralized and prevents later map construction
// from depending on the original user-facing entry shape.

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
constexpr auto ExpandIntegrationDomainEntry(E&& e)
{
   // Public integration domains always create a cell/local-facet domain. Mixed
   // spaces additionally create internal global face-domain entries when the
   // mixed space actually contains face-space batches.
   using Key = entry_key_t<E>;
   using Integration = uncvref_t<decltype(e.value)>;
   using Space = uncvref_t<decltype(std::declval<Integration>().space)>;

   using CellDomain = CellIntegrationDomain<Space>;
   auto cell_domain = Entry<DomainKey<Key::name>, CellDomain>{
      CellDomain{ static_cast<Space>(e.value.space) } };

   if constexpr (is_mixed_finite_element_space_v<Space>)
   {
      if constexpr (Space::num_interior_face_spaces > 0 &&
                    Space::num_boundary_face_spaces > 0)
      {
         using InteriorDomain = InteriorFaceIntegrationDomain<Space>;
         using BoundaryDomain = BoundaryFaceIntegrationDomain<Space>;
         return std::tuple{
            cell_domain,
            Entry<InteriorFaceDomainKey<Key::name>, InteriorDomain>{
               InteriorDomain{ static_cast<Space>(e.value.space) } },
            Entry<BoundaryFaceDomainKey<Key::name>, BoundaryDomain>{
               BoundaryDomain{ static_cast<Space>(e.value.space) } } };
      }
      else if constexpr (Space::num_interior_face_spaces > 0)
      {
         using InteriorDomain = InteriorFaceIntegrationDomain<Space>;
         return std::tuple{
            cell_domain,
            Entry<InteriorFaceDomainKey<Key::name>, InteriorDomain>{
               InteriorDomain{ static_cast<Space>(e.value.space) } } };
      }
      else if constexpr (Space::num_boundary_face_spaces > 0)
      {
         using BoundaryDomain = BoundaryFaceIntegrationDomain<Space>;
         return std::tuple{
            cell_domain,
            Entry<BoundaryFaceDomainKey<Key::name>, BoundaryDomain>{
               BoundaryDomain{ static_cast<Space>(e.value.space) } } };
      }
      else
      {
         return std::tuple{ cell_domain };
      }
   }
   else
   {
      // Homogeneous integration domains are cell/local-facet topology only;
      // they do not imply global face topology. Global face singleton cases
      // must use a singleton domain-decomposed FES with face FES entries.
      return std::tuple{ cell_domain };
   }
}

template<class E>
constexpr auto ExpandWeakFormContextEntry(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_integration_domain_key_v<Key>) {
      return ExpandIntegrationDomainEntry(std::forward<E>(e));
   } else {
      return std::tuple{ std::forward<E>(e) };
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

template<class E>
constexpr auto as_interior_face_domain_tuple(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_interior_face_domain_key_v<Key>) {
      return std::tuple{ std::forward<E>(e) };
   } else {
      return std::tuple{};
   }
}

template<class E>
constexpr auto as_boundary_face_domain_tuple(E&& e)
{
   using Key = entry_key_t<E>;
   if constexpr (is_boundary_face_domain_key_v<Key>) {
      return std::tuple{ std::forward<E>(e) };
   } else {
      return std::tuple{};
   }
}

template<class... ExistingEntries, class EntryT>
constexpr auto insert_fe_field_or_validate(
   StaticMap<ExistingEntries...> map,
   EntryT&& entry)
{
   using E = uncvref_t<EntryT>;
   using Key = typename E::key_type;

   if constexpr (StaticMap<ExistingEntries...>::template contains<Key>())
   {
      using ExistingEntry = find_entry_t<Key, ExistingEntries...>;
      static_assert(
         std::is_same_v<typename ExistingEntry::value_type, typename E::value_type> &&
         is_empty_finite_element_field_value_v<typename E::value_type>,
         "MakeWeakFormContext: duplicate finite element field keys are allowed "
         "only for matching empty trial/test field registrations.");
      return map;
   }
   else
   {
      return append(std::move(map), std::forward<EntryT>(entry));
   }
}

template<class Map>
constexpr auto make_fe_field_map_allowing_identical_duplicates(Map map)
{
   return map;
}

template<class Map, class FirstEntry, class... RestEntries>
constexpr auto make_fe_field_map_allowing_identical_duplicates(
   Map map,
   FirstEntry&& first_entry,
   RestEntries&&... rest_entries)
{
   auto next_map = insert_fe_field_or_validate(
      std::move(map),
      std::forward<FirstEntry>(first_entry));
   return make_fe_field_map_allowing_identical_duplicates(
      std::move(next_map),
      std::forward<RestEntries>(rest_entries)...);
}

template<class Tuple>
constexpr auto tuple_to_fe_field_map_allowing_identical_duplicates(Tuple&& t)
{
   return std::apply(
      [] (auto&&... entries)
      {
         return make_fe_field_map_allowing_identical_duplicates(
            make_map(),
            std::forward<decltype(entries)>(entries)...);
      },
      std::forward<Tuple>(t));
}

// -----------------------------------------------------------------------------
// Factory
// -----------------------------------------------------------------------------

template<class... Entries>
constexpr auto MakeWeakFormContext(Entries&&... es)
{
   // MakeWeakFormContext materializes normalized_entries before filtering them
   // into several maps. That avoids forwarding/lifetime surprises because the
   // same normalized tuple is inspected multiple times.
   static_assert(
      ((has_key_type_v<Entries> && has_value_type_v<Entries>) && ...),
      "MakeWeakFormContext: every argument must be an Entry-like type exposing "
      "key_type and value_type.");

   static_assert(
      (is_weak_form_context_key_v<entry_key_t<Entries>> && ...),
      "MakeWeakFormContext: keys must derive from WeakFormKeyTag.");

   auto normalized_entries =
      std::tuple_cat(ExpandWeakFormContextEntry(std::forward<Entries>(es))...);
   using NormalizedEntries = std::remove_cvref_t<decltype(normalized_entries)>;

   // Run duplicate checks after normalization so conflicts introduced by a
   // public integration-domain expansion are diagnosed before map construction.
   static_assert(
      context_entry_tuple_is_compatible<NormalizedEntries>::value,
      "MakeWeakFormContext: duplicate keys are not allowed after integration "
      "domain normalization, except duplicate identical finite element field "
      "entries such as matching trial/test field registrations.");

   auto fields_t = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_field_tuple(entries)...);
      },
      normalized_entries);
   auto domains_t = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_domain_tuple(entries)...);
      },
      normalized_entries);
   auto interior_face_domains_t = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_interior_face_domain_tuple(entries)...);
      },
      normalized_entries);
   auto boundary_face_domains_t = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_boundary_face_domain_tuple(entries)...);
      },
      normalized_entries);
   auto fe_fields_t = std::apply(
      [] (const auto&... entries)
      {
         return std::tuple_cat(as_fe_field_tuple(entries)...);
      },
      normalized_entries);

   auto fields_map = tuple_to_map(std::move(fields_t));
   auto domains_map = tuple_to_map(std::move(domains_t));
   auto interior_face_domains_map =
      tuple_to_map(std::move(interior_face_domains_t));
   auto boundary_face_domains_map =
      tuple_to_map(std::move(boundary_face_domains_t));
   auto fe_fields_map =
      tuple_to_fe_field_map_allowing_identical_duplicates(std::move(fe_fields_t));

   return WeakFormContext<
      decltype(domains_map),
      decltype(interior_face_domains_map),
      decltype(boundary_face_domains_map),
      decltype(fe_fields_map),
      decltype(fields_map)>{
         std::move(domains_map),
         std::move(interior_face_domains_map),
         std::move(boundary_face_domains_map),
         std::move(fe_fields_map),
         std::move(fields_map)
   };
}

} // namespace gendil
