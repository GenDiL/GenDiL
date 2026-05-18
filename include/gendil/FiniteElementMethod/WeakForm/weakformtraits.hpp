// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

enum class OperatorMask : unsigned
{
   None = 0u,
   Values = 1u,
   Gradients = 2u
};

constexpr OperatorMask operator+(OperatorMask a, OperatorMask b) {
   return OperatorMask(unsigned(a) | unsigned(b));
}

constexpr bool need_values(OperatorMask m) { return (unsigned(m) & 1u) != 0u; }
constexpr bool need_gradients(OperatorMask m) { return (unsigned(m) & 2u) != 0u; }

template<StaticString LHSName, StaticString RHSName>
struct MergeName
{
   static_assert(
      dependent_false_value_v<LHSName, RHSName>,
      "requirements: multiple distinct trial/test spaces detected.");
};

template < StaticString Name >
struct MergeName<Name, Name>
{
   static constexpr decltype(Name) value = Name;
};

template < StaticString LHSName >
struct MergeName<LHSName, "Error">
{
   static constexpr decltype(LHSName) value = LHSName;
};

template < StaticString RHSName >
struct MergeName<"Error", RHSName>
{
   static constexpr decltype(RHSName) value = RHSName;
};

template <>
struct MergeName<"Error", "Error">
{
   static constexpr StaticString value = StaticString{"Error"};
};

template<StaticString LHSName, StaticString RHSName>
consteval auto merge_name()
{
   constexpr auto error = StaticString{"Error"};

   if constexpr (LHSName == error) {
      return RHSName;
   } else if constexpr (RHSName == error) {
      return LHSName;
   } else {
      static_assert(LHSName == RHSName,
         "requirements: multiple distinct trial/test spaces detected.");
      return LHSName;
   }
}

struct NoRequirements
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr StaticString trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr StaticString test_name = StaticString{"Error"};
};

// Forward declaration of requirements template
template< typename Expr>
struct requirements;

// Leaf requirements - defined first so they're complete when helper base classes are instantiated

template < StaticString Name, FieldShape Shape >
struct requirements< TrialSpace<Name, Shape> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::Values;
   static constexpr decltype(Name) trial_name = Name;
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr auto test_name = StaticString{"Error"};
};

template < StaticString Name, FieldShape Shape >
struct requirements< GradientExpr<TrialSpace<Name, Shape>> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::Gradients;
   static constexpr decltype(Name) trial_name = Name;
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr auto test_name = StaticString{"Error"};
};

template < StaticString Name, FieldShape Shape >
struct requirements< TestSpace<Name, Shape> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr auto trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::Values;
   static constexpr decltype(Name) test_name = Name;
};

template < StaticString Name, FieldShape Shape >
struct requirements< GradientExpr<TestSpace<Name, Shape>> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr auto trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::Gradients;
   static constexpr decltype(Name) test_name = Name;
};

template < StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs >
struct requirements< Coefficient<Name, Shape, Fn, Inputs...> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr StaticString trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr StaticString test_name = StaticString{"Error"};
};

template < StaticString Name >
struct requirements< FiniteElementField<Name> >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr auto trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr auto test_name = StaticString{"Error"};
};

template <>
struct requirements< ScaleExpr >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr auto trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr auto test_name = StaticString{"Error"};
};

template <>
struct requirements< Normal >
{
   static constexpr OperatorMask trial_mask = OperatorMask::None;
   static constexpr auto trial_name = StaticString{"Error"};
   static constexpr OperatorMask test_mask = OperatorMask::None;
   static constexpr auto test_name = StaticString{"Error"};
};

// Helper base classes for derived requirements - defined after leaf requirements to avoid incomplete type issues

// Helper for unary expressions (forward single child's requirements)
template < typename Expr >
struct UnaryExprRequirements
{
   static constexpr OperatorMask trial_mask = requirements<Expr>::trial_mask;
   static constexpr auto trial_name = requirements<Expr>::trial_name;
   static constexpr OperatorMask test_mask = requirements<Expr>::test_mask;
   static constexpr auto test_name = requirements<Expr>::test_name;
};

// Helper for binary expressions (merge requirements from two children)
template < typename LHSExpr, typename RHSExpr >
struct BinaryExprRequirements
{
   static constexpr OperatorMask trial_mask =
      requirements<LHSExpr>::trial_mask + requirements<RHSExpr>::trial_mask;
   static constexpr auto trial_name =
      merge_name<requirements<LHSExpr>::trial_name, requirements<RHSExpr>::trial_name>();
   static constexpr OperatorMask test_mask =
      requirements<LHSExpr>::test_mask + requirements<RHSExpr>::test_mask;
   static constexpr auto test_name =
      merge_name<requirements<LHSExpr>::test_name, requirements<RHSExpr>::test_name>();
};

// Helper for n-ary expressions (recursively merge requirements)

template < typename... Expr >
struct NAryExprRequirements; 

template < typename Expr >
struct NAryExprRequirements<Expr> : UnaryExprRequirements<Expr>
{};

template < typename FirstExpr, typename SecondExpr >
struct NAryExprRequirements<FirstExpr, SecondExpr> : BinaryExprRequirements<FirstExpr, SecondExpr>
{};

template < typename FirstExpr, typename... TailExpr >
struct NAryExprRequirements< FirstExpr, TailExpr... >
{
   using RestReqs = NAryExprRequirements<TailExpr...>;
   static constexpr OperatorMask trial_mask =
      requirements<FirstExpr>::trial_mask + RestReqs::trial_mask;
   static constexpr auto trial_name =
      merge_name<requirements<FirstExpr>::trial_name, RestReqs::trial_name>();
   static constexpr OperatorMask test_mask =
      requirements<FirstExpr>::test_mask + RestReqs::test_mask;
   static constexpr auto test_name =
      merge_name<requirements<FirstExpr>::test_name, RestReqs::test_name>();
};

// Derived requirements - inherit from helper base classes

template < typename Domain, FieldExpr Expr >
struct requirements< Integrand<Domain, Expr> > : UnaryExprRequirements<Expr>
{};

template < FieldExpr Expr >
struct requirements< NegExpr<Expr> > : UnaryExprRequirements<Expr>
{};

template < FieldExpr HeadExpr, FieldExpr... TailExpr >
struct requirements< SumExpr<HeadExpr, TailExpr...> > : NAryExprRequirements<HeadExpr, TailExpr...>
{};

template < FieldExpr Expr >
struct requirements< SumExpr<Expr> > : UnaryExprRequirements<Expr>
{};

template<class Key, class T>
struct requirements< Entry<Key, T> > : UnaryExprRequirements<T>
{};

template < typename FirstEntry, typename ... RestEntries >
struct requirements< StaticMap<FirstEntry, RestEntries...> > : NAryExprRequirements<FirstEntry, RestEntries...>
{};

template < typename Entry >
struct requirements< StaticMap<Entry> > : UnaryExprRequirements<Entry>
{};

template < typename Map >
struct requirements< SumFormExpr< Map > > : UnaryExprRequirements<Map>
{};

template < FieldExpr Expr >
struct requirements< AverageExpr<Expr> > : UnaryExprRequirements<Expr>
{};

template < FieldExpr LHS, FieldExpr RHS >
struct requirements< DotExpr<LHS, RHS> > : BinaryExprRequirements<LHS, RHS>
{};

template < FieldExpr LHS, FieldExpr RHS >
struct requirements< InnerExpr<LHS, RHS> > : BinaryExprRequirements<LHS, RHS>
{};

template < FieldExpr LHS, FieldExpr RHS >
struct requirements< OuterExpr<LHS, RHS> > : BinaryExprRequirements<LHS, RHS>
{};

template < FieldExpr Expr >
struct requirements< JumpExpr<Expr> > : UnaryExprRequirements<Expr>
{};

template < FieldExpr AdvExpr, FieldExpr Expr >
struct requirements< UpwindExpr<AdvExpr, Expr> > : BinaryExprRequirements<AdvExpr, Expr>
{};

template < FieldExpr LHS, FieldExpr RHS >
struct requirements< MultFieldExpr<LHS, RHS> > : BinaryExprRequirements<LHS, RHS>
{};

template < FieldExpr LHS, FieldExpr RHS >
struct requirements< ProductExpr<LHS, RHS> > : BinaryExprRequirements<LHS, RHS>
{};

template<class T> inline constexpr bool need_trial_values_v = need_values(requirements<T>::trial_mask);
template<class T> inline constexpr bool need_trial_grads_v  = need_gradients (requirements<T>::trial_mask);
template<class T> inline constexpr bool need_test_values_v = need_values(requirements<T>::test_mask);
template<class T> inline constexpr bool need_test_grads_v  = need_gradients (requirements<T>::test_mask);

template <typename Expr>
struct facet_role
{
   static constexpr OperatorMask trial_mask = requirements<Expr>::trial_mask;
   static constexpr OperatorMask test_mask  = requirements<Expr>::test_mask;

   static constexpr bool is_trial =
      trial_mask != OperatorMask::None;

   static constexpr bool is_test =
      test_mask != OperatorMask::None;

   static_assert(
      is_trial != is_test,
      "Facet operators require an expression that is either purely trial or purely test."
   );
};

// Forward declarations for domain types
template<StaticString Name> struct Cells;
template<StaticString Name> struct InteriorFacets;
template<StaticString Name> struct BoundaryFacets;

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

template<typename Map>
struct has_cell_contributions<SumFormExpr<Map>> : has_cell_contributions<Map> {};

template<typename FirstEntry, typename... RestEntries>
struct has_cell_contributions<StaticMap<FirstEntry, RestEntries...>>
{
   static constexpr bool value =
      has_cell_contributions<FirstEntry>::value ||
      has_cell_contributions<StaticMap<RestEntries...>>::value;
};

template<typename Entry>
struct has_cell_contributions<StaticMap<Entry>> : has_cell_contributions<Entry> {};

template<typename Key, typename T>
struct has_cell_contributions<Entry<Key, T>> : has_cell_contributions<T> {};

template<typename Form>
inline constexpr bool has_cell_contributions_v = has_cell_contributions<Form>::value;

// Interior facets

template<typename Form>
struct has_interior_facet_contributions : std::false_type {};

template<typename Domain, FieldExpr Expr>
struct has_interior_facet_contributions<Integrand<Domain, Expr>> : is_interior_facets_domain<Domain> {};

template<typename Map>
struct has_interior_facet_contributions<SumFormExpr<Map>> : has_interior_facet_contributions<Map> {};

template<typename FirstEntry, typename... RestEntries>
struct has_interior_facet_contributions<StaticMap<FirstEntry, RestEntries...>>
{
   static constexpr bool value =
      has_interior_facet_contributions<FirstEntry>::value ||
      has_interior_facet_contributions<StaticMap<RestEntries...>>::value;
};

template<typename Entry>
struct has_interior_facet_contributions<StaticMap<Entry>> : has_interior_facet_contributions<Entry> {};

template<typename Key, typename T>
struct has_interior_facet_contributions<Entry<Key, T>> : has_interior_facet_contributions<T> {};

template<typename Form>
inline constexpr bool has_interior_facet_contributions_v = has_interior_facet_contributions<Form>::value;

// Boundary facets

template<typename Form>
struct has_boundary_facet_contributions : std::false_type {};

template<typename Domain, FieldExpr Expr>
struct has_boundary_facet_contributions<Integrand<Domain, Expr>> : is_boundary_facets_domain<Domain> {};

template<typename Map>
struct has_boundary_facet_contributions<SumFormExpr<Map>> : has_boundary_facet_contributions<Map> {};

template<typename FirstEntry, typename... RestEntries>
struct has_boundary_facet_contributions<StaticMap<FirstEntry, RestEntries...>>
{
   static constexpr bool value =
      has_boundary_facet_contributions<FirstEntry>::value ||
      has_boundary_facet_contributions<StaticMap<RestEntries...>>::value;
};

template<typename Entry>
struct has_boundary_facet_contributions<StaticMap<Entry>> : has_boundary_facet_contributions<Entry> {};

template<typename Key, typename T>
struct has_boundary_facet_contributions<Entry<Key, T>> : has_boundary_facet_contributions<T> {};

template<typename Form>
inline constexpr bool has_boundary_facet_contributions_v = has_boundary_facet_contributions<Form>::value;

}
