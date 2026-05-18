// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/neg.hpp"

// This file assumes make_map / make_entry / map_insert_or_merge are visible
// from the surrounding includes as in your current setup.

namespace gendil
{

// -----------------------------------------------------------------------------
// Field-expression sums
// -----------------------------------------------------------------------------

template < FieldExpr Head, FieldExpr... Tail >
struct SumExpr : FieldBase
{
   std::tuple<Head, Tail...> terms;

   constexpr SumExpr(const Head& head, const Tail&... tail)
      : terms(head, tail...)
   {}

   constexpr SumExpr(Head&& head, Tail&&... tail)
      : terms(std::move(head), std::move(tail)...)
   {}

   template < typename... Args >
   GENDIL_HOST_DEVICE
   constexpr auto operator()(Args&&... args) const
   {
      return std::apply(
         [&](auto const&... ts)
         {
            return (ts(std::forward<Args>(args)...) + ...);
         },
         terms
      );
   }
};

template < class T >
struct is_sum_expr : std::false_type {};

template < FieldExpr Head, FieldExpr... Tail >
struct is_sum_expr< SumExpr<Head, Tail...> > : std::true_type {};

template < class T >
inline constexpr bool is_sum_expr_v = is_sum_expr<std::remove_cvref_t<T>>::value;

template < FieldExpr Expr >
constexpr auto AsSumTuple(Expr&& expr)
{
   using E = std::remove_cvref_t<Expr>;

   if constexpr (is_sum_expr_v<E>)
   {
      return std::forward<Expr>(expr).terms;
   }
   else
   {
      return std::tuple<E>{ std::forward<Expr>(expr) };
   }
}

template < class Tuple >
constexpr auto MakeSumExprFromTuple(Tuple&& tuple)
{
   return std::apply(
      [](auto&&... xs)
      {
         using Sum = SumExpr<std::remove_cvref_t<decltype(xs)>...>;
         return Sum(std::forward<decltype(xs)>(xs)...);
      },
      std::forward<Tuple>(tuple)
   );
}

template < FieldExpr... Terms >
std::ostream& operator<<(std::ostream& os, const SumExpr<Terms...>& sum)
{
   os << "(";
   bool first = true;
   std::apply(
      [&](auto const&... ts)
      {
         (((os << (first ? "" : " + ") << ts), first = false), ...);
      },
      sum.terms
   );
   os << ")";
   return os;
}

// Flattening field-expression addition:
//   Expr + Expr              -> SumExpr<Expr, Expr>
//   SumExpr + Expr           -> SumExpr<...>
//   Expr + SumExpr           -> SumExpr<...>
//   SumExpr + SumExpr        -> SumExpr<...>
template < FieldExpr LHS, FieldExpr RHS >
constexpr auto operator+(LHS&& lhs, RHS&& rhs)
{
   auto lhs_terms = AsSumTuple(std::forward<LHS>(lhs));
   auto rhs_terms = AsSumTuple(std::forward<RHS>(rhs));
   auto all_terms = std::tuple_cat(std::move(lhs_terms), std::move(rhs_terms));
   return MakeSumExprFromTuple(std::move(all_terms));
}

template < FieldExpr LHS, FieldExpr RHS >
constexpr auto operator-(LHS&& lhs, RHS&& rhs)
{
   auto lhs_terms = AsSumTuple(std::forward<LHS>(lhs));
   auto rhs_terms = AsSumTuple(NegExpr{std::forward<RHS>(rhs)});
   auto all_terms = std::tuple_cat(std::move(lhs_terms), std::move(rhs_terms));
   return MakeSumExprFromTuple(std::move(all_terms));
}

// Optional compatibility alias if some code still refers to AddFieldExpr.
template < FieldExpr LHS, FieldExpr RHS >
using AddFieldExpr = SumExpr<LHS, RHS>;

} // namespace gendil
