// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/domain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

// Domain detection traits - detect which domain types appear in a weak form

template<typename Domain>
struct is_cells_domain : std::false_type {};

template<StaticString Name>
struct is_cells_domain<Cells<Name>> : std::true_type {};

template<typename Domain>
struct is_interior_facets_domain : std::false_type {};

template<StaticString Name>
struct is_interior_facets_domain<InteriorFacets<Name>> : std::true_type {};

template<typename Domain>
struct is_boundary_facets_domain : std::false_type {};

template<StaticString Name>
struct is_boundary_facets_domain<BoundaryFacets<Name>> : std::true_type {};

// Check if a weak form contains contributions from a specific domain type

template<typename Form>
struct has_cell_contributions : std::false_type {};

template<typename Domain, FieldExpr Expr>
struct has_cell_contributions<Integrand<Domain, Expr>> : is_cells_domain<Domain> {};


template<typename Form>
inline constexpr bool has_cell_contributions_v = has_cell_contributions<Form>::value;

// Interior facets

template<typename Form>
struct has_interior_facet_contributions : std::false_type {};

template<typename Domain, FieldExpr Expr>
struct has_interior_facet_contributions<Integrand<Domain, Expr>> : is_interior_facets_domain<Domain> {};


template<typename Form>
inline constexpr bool has_interior_facet_contributions_v = has_interior_facet_contributions<Form>::value;

// Boundary facets

template<typename Form>
struct has_boundary_facet_contributions : std::false_type {};

template<typename Domain, FieldExpr Expr>
struct has_boundary_facet_contributions<Integrand<Domain, Expr>> : is_boundary_facets_domain<Domain> {};


template<typename Form>
inline constexpr bool has_boundary_facet_contributions_v = has_boundary_facet_contributions<Form>::value;


template<class... Terms>
struct has_cell_contributions<std::tuple<Terms...>>
   : std::bool_constant<(has_cell_contributions<std::remove_cvref_t<Terms>>::value || ...)> {};
template<class Terms, class Symbols>
struct has_cell_contributions<SumFormExpr<Terms, Symbols>> : has_cell_contributions<Terms> {};


template<class... Terms>
struct has_interior_facet_contributions<std::tuple<Terms...>>
   : std::bool_constant<(has_interior_facet_contributions<std::remove_cvref_t<Terms>>::value || ...)> {};
template<class Terms, class Symbols>
struct has_interior_facet_contributions<SumFormExpr<Terms, Symbols>> : has_interior_facet_contributions<Terms> {};


template<class... Terms>
struct has_boundary_facet_contributions<std::tuple<Terms...>>
   : std::bool_constant<(has_boundary_facet_contributions<std::remove_cvref_t<Terms>>::value || ...)> {};
template<class Terms, class Symbols>
struct has_boundary_facet_contributions<SumFormExpr<Terms, Symbols>> : has_boundary_facet_contributions<Terms> {};

} // namespace gendil
