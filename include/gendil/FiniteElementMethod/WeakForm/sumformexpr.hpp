// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"

// This file assumes make_map / make_entry / map_insert_or_merge are visible
// from the surrounding includes as in your current setup.

namespace gendil
{

// -----------------------------------------------------------------------------
// Cross-domain weak-form sums
// -----------------------------------------------------------------------------

template < class Map >
struct SumFormExpr : FormBase
{
   Map map;

   constexpr SumFormExpr(Map&& map_)
      : map(std::forward<Map>(map_))
   {}
};

template < class T >
struct is_sum_form_expr : std::false_type {};

template < class Map >
struct is_sum_form_expr< SumFormExpr<Map> > : std::true_type {};

template < class T >
inline constexpr bool is_sum_form_expr_v =
   is_sum_form_expr<std::remove_cvref_t<T>>::value;

template < FormExpr Form >
   requires (!is_sum_form_expr_v<Form>)
constexpr auto MakeSumFormExpr(Form&& form)
{
   using F   = std::remove_cvref_t<Form>;
   using Key = typename F::domain_type;

   auto map = make_map(
      make_entry<Key>(std::forward<Form>(form))
   );

   return SumFormExpr<decltype(map)>{ std::move(map) };
}

template < class Map >
std::ostream& operator<<(std::ostream& os, const SumFormExpr<Map>& sum)
{
   bool first = true;
   std::apply(
      [&](auto const&... entries)
      {
         (((os << (first ? "" : " + ") << entries.value), first = false), ...);
      },
      sum.map.entries
   );
   return os;
}

// -----------------------------------------------------------------------------
// Same-domain integrand fusion
// -----------------------------------------------------------------------------

// Same-domain integrands are fused inside the Integrand expression:
//   Integrand<Domain, ExprA> + Integrand<Domain, ExprB>
//      -> Integrand<Domain, SumExpr<ExprA, ExprB>>
template < DomainExpr Domain, FieldExpr LHSExpr, FieldExpr RHSExpr >
constexpr auto Add(
   const Integrand<Domain, LHSExpr>& lhs,
   const Integrand<Domain, RHSExpr>& rhs)
{
   auto expr = lhs.expr + rhs.expr;
   using Expr = std::remove_cvref_t<decltype(expr)>;

   return Integrand<Domain, Expr>{ lhs.domain, std::move(expr) };
}

// -----------------------------------------------------------------------------
// Cross-domain form normalization
// -----------------------------------------------------------------------------

template < class Map, FormExpr Form >
   requires (!is_sum_form_expr_v<Form>)
constexpr auto AddFormTerm(SumFormExpr<Map> sum, Form&& form)
{
   using F   = std::remove_cvref_t<Form>;
   using Key = std::pair< typename F::domain_type, std::integral_constant<OperatorMask, requirements<F>::test_mask> >;

   auto new_map = map_insert_or_merge<Key>(
      std::move(sum.map),
      std::forward<Form>(form),
      [] (auto&& old_term, auto&& new_term)
      {
         // If the key already exists, this will use the Integrand-specific
         // operator+ above and therefore fuse same-domain terms into
         // Integrand<Domain, SumExpr<...>>.
         // return std::forward<decltype(old_term)>(old_term)
         //      + std::forward<decltype(new_term)>(new_term);
         return Add( old_term, new_term );
      }
   );

   return SumFormExpr<decltype(new_map)>{ std::move(new_map) };
}

template < class MapL, class MapR >
constexpr auto MergeSumFormExpr(SumFormExpr<MapL> lhs, const SumFormExpr<MapR>& rhs)
{
   return std::apply(
      [&](auto const&... entries)
      {
         auto result = std::move(lhs);
         ((result = AddFormTerm(std::move(result), entries.value)), ...);
         return result;
      },
      rhs.map.entries
   );
}

// -----------------------------------------------------------------------------
// Form operator+
// -----------------------------------------------------------------------------

// Form + Form  (leaf + leaf)
// If they have the same domain and are Integrands, the more specialized
// Integrand overload above is selected.
// Otherwise this creates a normalized cross-domain SumFormExpr.
template < FormExpr LHS, FormExpr RHS >
   requires (!is_sum_form_expr_v<LHS> && !is_sum_form_expr_v<RHS>)
constexpr auto operator+(LHS&& lhs, RHS&& rhs)
{
   auto sum = MakeSumFormExpr(std::forward<LHS>(lhs));
   return AddFormTerm(std::move(sum), std::forward<RHS>(rhs));
}

// SumFormExpr + Form
template < class Map, FormExpr RHS >
   requires (!is_sum_form_expr_v<RHS>)
constexpr auto operator+(SumFormExpr<Map> lhs, RHS&& rhs)
{
   return AddFormTerm(std::move(lhs), std::forward<RHS>(rhs));
}

// Form + SumFormExpr
template < FormExpr LHS, class Map >
   requires (!is_sum_form_expr_v<LHS>)
constexpr auto operator+(LHS&& lhs, SumFormExpr<Map> rhs)
{
   return AddFormTerm(std::move(rhs), std::forward<LHS>(lhs));
}

// SumFormExpr + SumFormExpr
template < class MapL, class MapR >
constexpr auto operator+(SumFormExpr<MapL> lhs, const SumFormExpr<MapR>& rhs)
{
   return MergeSumFormExpr(std::move(lhs), rhs);
}

} // namespace gendil
