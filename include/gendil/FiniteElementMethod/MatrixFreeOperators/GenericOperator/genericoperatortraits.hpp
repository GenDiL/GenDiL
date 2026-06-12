// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"

#include <type_traits>

namespace gendil {

// Static analysis for the matrix-free generic operator traversal strategy.
//
// Dispatch is intentionally topology-driven:
// - no global face domains in the weak-form context selects the local,
//   cell-owned facet traversal;
// - at least one global face domain selects the global face traversal.
//
// Once global traversal is selected, hybrid local/global facet execution is not
// allowed. Every facet term that appears in the form must have the matching
// named global face domain in the context. Cell terms do not participate in this
// validation.

// True when the context provides any global face topology. This does not inspect
// the form and does not consider KernelPolicy batching; batching only restricts
// whether the selected traversal strategy is valid.
template<class WFContext>
inline constexpr bool has_global_face_domains_v =
   std::remove_cvref_t<WFContext>::has_any_interior_face_domain() ||
   std::remove_cvref_t<WFContext>::has_any_boundary_face_domain();

// Name-aware validation for InteriorFacets<Name> terms under global traversal.
// The default is true so non-interior terms, including cell and boundary terms,
// do not affect the interior-face requirement.
template<class Form, class WFContext>
struct interior_facet_terms_have_domains : std::true_type {};

// An interior facet integrand requires the matching
// MakeInteriorFaceDomain<Name>(...) entry in the weak-form context.
template<StaticString DomainName, FieldExpr Expr, class WFContext>
struct interior_facet_terms_have_domains<
   Integrand<InteriorFacets<DomainName>, Expr>,
   WFContext>
   : std::bool_constant<
        std::remove_cvref_t<WFContext>::template
           has_interior_face_domain<DomainName>()> {};

template<class Map, class WFContext>
struct interior_facet_terms_have_domains<SumFormExpr<Map>, WFContext>
   : interior_facet_terms_have_domains<Map, WFContext> {};

template<class FirstEntry, class... RestEntries, class WFContext>
struct interior_facet_terms_have_domains<
   StaticMap<FirstEntry, RestEntries...>,
   WFContext>
{
   static constexpr bool value =
      interior_facet_terms_have_domains<FirstEntry, WFContext>::value &&
      interior_facet_terms_have_domains<
         StaticMap<RestEntries...>,
         WFContext>::value;
};

template<class Entry, class WFContext>
struct interior_facet_terms_have_domains<StaticMap<Entry>, WFContext>
   : interior_facet_terms_have_domains<Entry, WFContext> {};

template<class Key, class T, class WFContext>
struct interior_facet_terms_have_domains<Entry<Key, T>, WFContext>
   : interior_facet_terms_have_domains<T, WFContext> {};

template<class Form, class WFContext>
inline constexpr bool interior_facet_terms_have_domains_v =
   interior_facet_terms_have_domains<
      std::remove_cvref_t<Form>,
      std::remove_cvref_t<WFContext>>::value;

// Name-aware validation for BoundaryFacets<Name> terms under global traversal.
// The default is true so non-boundary terms, including cell and interior terms,
// do not affect the boundary-face requirement.
template<class Form, class WFContext>
struct boundary_facet_terms_have_domains : std::true_type {};

// A boundary facet integrand requires the matching
// MakeBoundaryFaceDomain<Name>(...) entry in the weak-form context.
template<StaticString DomainName, FieldExpr Expr, class WFContext>
struct boundary_facet_terms_have_domains<
   Integrand<BoundaryFacets<DomainName>, Expr>,
   WFContext>
   : std::bool_constant<
        std::remove_cvref_t<WFContext>::template
           has_boundary_face_domain<DomainName>()> {};

template<class Map, class WFContext>
struct boundary_facet_terms_have_domains<SumFormExpr<Map>, WFContext>
   : boundary_facet_terms_have_domains<Map, WFContext> {};

template<class FirstEntry, class... RestEntries, class WFContext>
struct boundary_facet_terms_have_domains<
   StaticMap<FirstEntry, RestEntries...>,
   WFContext>
{
   static constexpr bool value =
      boundary_facet_terms_have_domains<FirstEntry, WFContext>::value &&
      boundary_facet_terms_have_domains<
         StaticMap<RestEntries...>,
         WFContext>::value;
};

template<class Entry, class WFContext>
struct boundary_facet_terms_have_domains<StaticMap<Entry>, WFContext>
   : boundary_facet_terms_have_domains<Entry, WFContext> {};

template<class Key, class T, class WFContext>
struct boundary_facet_terms_have_domains<Entry<Key, T>, WFContext>
   : boundary_facet_terms_have_domains<T, WFContext> {};

template<class Form, class WFContext>
inline constexpr bool boundary_facet_terms_have_domains_v =
   boundary_facet_terms_have_domains<
      std::remove_cvref_t<Form>,
      std::remove_cvref_t<WFContext>>::value;

// Global traversal is valid only when every facet term in the form has its
// matching named global face domain. A form with only cell terms satisfies this
// requirement even if the context provides global face domains.
template<class Form, class WFContext>
inline constexpr bool global_facet_domain_requirements_satisfied_v =
   interior_facet_terms_have_domains_v<Form, WFContext> &&
   boundary_facet_terms_have_domains_v<Form, WFContext>;

// High-level strategy selector used by GenericExplicitOperator. Keep this based
// only on context-provided topology so batch_size never implicitly selects the
// global path.
template<class WFContext>
inline constexpr bool use_global_facets_operator_v =
   has_global_face_domains_v<WFContext>;

} // namespace gendil
