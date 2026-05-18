// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"

namespace gendil
{

// Enumeration of the types of trial operators that can be applied to a trial space.
enum class TrialOperatorType
{
   None,
   InterpolateValues,
   InterpolateGradients
};

std::ostream& operator<<(std::ostream& os, const TrialOperatorType& op_type )
{
   switch ( op_type )
   {
      case TrialOperatorType::None:
         return os << "None";
      case TrialOperatorType::InterpolateValues:
         return os << "InterpolateValues";
      case TrialOperatorType::InterpolateGradients:
         return os << "InterpolateGradients";
      default:
         return os << "UnknownTrialOperatorType";
   }
}

template < StaticString Name, TrialOperatorType... OpTypes >
struct TrialOperator
{
   static constexpr auto name = Name;
   using op_types = std::integer_sequence<TrialOperatorType, OpTypes...>;
};

template < StaticString Name, TrialOperatorType FirstOp, TrialOperatorType... OpTypes >
std::ostream& operator<<(std::ostream& os, const TrialOperator<Name, FirstOp, OpTypes...>& grad)
{
   os << "TrialOperators< " << Name.view() << " >( " << FirstOp;
   ((os << ", " << OpTypes), ...);
   os << " )";
   return os;
}

template < TrialOperatorType op_type, typename TrialOp >
struct HasOperatorType
{
   static constexpr bool value = false;
};

template < TrialOperatorType op_type, StaticString Name, TrialOperatorType FirstOpType, TrialOperatorType... RestOpTypes >
struct HasOperatorType< op_type, TrialOperator< Name, FirstOpType, RestOpTypes... > >
{
   static constexpr bool value = ( FirstOpType == op_type ) || HasOperatorType< op_type, TrialOperator< Name, RestOpTypes... > >::value;
};

template < TrialOperatorType op_type, typename TrialOp >
static constexpr bool HasOperatorType_v = HasOperatorType< op_type, TrialOp >::value;

template < StaticString Name, TrialOperatorType... LHSOpTypes >
auto CatOperatorTypes(
   const TrialOperator<Name, LHSOpTypes...> & lhs,
   const TrialOperator<Name> & rhs )
{
   return TrialOperator<Name, LHSOpTypes...>{};
}

template < StaticString Name, TrialOperatorType... RHSOpTypes >
auto CatOperatorTypes(
   const TrialOperator<Name> & lhs,
   const TrialOperator<Name, RHSOpTypes...> & rhs )
{
   return TrialOperator<Name, RHSOpTypes...>{};
}

template < StaticString Name, TrialOperatorType FirstRHSOpType, TrialOperatorType... LHSOpTypes, TrialOperatorType... RestRHSOpTypes >
auto CatOperatorTypes(
   const TrialOperator<Name, LHSOpTypes...> & lhs,
   const TrialOperator<Name, FirstRHSOpType, RestRHSOpTypes...> & rhs )
{
   if constexpr ( HasOperatorType_v< FirstRHSOpType, decltype(lhs) > )
   {
      return CatOperatorTypes( TrialOperator<Name, LHSOpTypes...>{}, TrialOperator<Name, RestRHSOpTypes...>{} );
   }
   else
   {
      return CatOperatorTypes( TrialOperator<Name, LHSOpTypes..., FirstRHSOpType>{}, TrialOperator<Name, RestRHSOpTypes...>{} );
   }
}

template < StaticString LHSName, StaticString RHSName, TrialOperatorType... LHSOpTypes, TrialOperatorType... RHSOpTypes >
auto operator+(
   const TrialOperator<LHSName, LHSOpTypes...> & lhs,
   const TrialOperator<RHSName, RHSOpTypes...> & rhs )
{
   // static_assert( false, "There can only be one trial space." );
   return TrialOperator<"Error", TrialOperatorType::None>{};
}

template < StaticString Name, TrialOperatorType... LHSOpTypes, TrialOperatorType... RHSOpTypes >
auto operator+(
   const TrialOperator<Name, LHSOpTypes...> & lhs,
   const TrialOperator<Name, RHSOpTypes...> & rhs )
{
   return CatOperatorTypes( lhs, rhs );
}

template < StaticString Name, TrialOperatorType... OpTypes >
auto operator+(
   const TrialOperator<Name, OpTypes...> & lhs,
   const Empty & rhs )
{
   return TrialOperator<Name, OpTypes...>{};
}

template < StaticString Name, TrialOperatorType... OpTypes >
auto operator+(
   const Empty & lhs,
   const TrialOperator<Name, OpTypes...> & rhs )
{
   return TrialOperator<Name, OpTypes...>{};
}

auto operator+(
   const Empty & lhs,
   const Empty & rhs )
{
   return TrialOperator<"Error", TrialOperatorType::None>{};
}

template < StaticString Name, FieldShape Shape >
auto GetTrialOperators( const TrialSpace<Name, Shape> & expr )
   // -> TrialOperator<Name, TrialOperatorType::InterpolateValues>
{
   return TrialOperator<Name, TrialOperatorType::InterpolateValues>{};
}

template < StaticString Name, FieldShape Shape >
auto GetTrialOperators( const GradientExpr<TrialSpace<Name, Shape>> & expr )
   // -> TrialOperator<Name, TrialOperatorType::InterpolateGradients>
{
   return TrialOperator<Name, TrialOperatorType::InterpolateGradients>{};
}

template < StaticString Name, FieldShape Shape >
auto GetTrialOperators( const GradientExpr<TestSpace<Name, Shape>> & expr )
   // -> Empty
{
   return Empty{};
}

// template < FieldExpr LHSExpr, FieldExpr RHSExpr >
// auto GetTrialOperators( const AddFieldExpr<LHSExpr, RHSExpr> & expr )
// {
//    auto lhs_ops = GetTrialOperators( expr.lhs );
//    auto rhs_ops = GetTrialOperators( expr.rhs );

//    return lhs_ops + rhs_ops;
// }

template < FieldExpr Head, FieldExpr... Tail >
auto GetTrialOperators( const SumExpr<Head, Tail...> & expr )
   // -> decltype(std::apply(
   //    [](auto const&... ts) { return (GetTrialOperators(ts) + ...); },
   //    expr.terms))
{
   return std::apply(
      [&](auto const&... ts)
      {
         return (GetTrialOperators(ts) + ...);
      },
      expr.terms
   );
}

template < FieldExpr Expr >
auto GetTrialOperators( const AverageExpr<Expr> & expr )
   // -> decltype(GetTrialOperators( std::declval<Expr>() ))
{
   return GetTrialOperators( expr.expr );
}

template <StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
auto GetTrialOperators( const Coefficient<Name, Shape, Fn, Inputs...> & expr )
   // -> Empty
{
   return Empty{};
}

template < FieldExpr LHS, FieldExpr RHS >
auto GetTrialOperators( const DotExpr<LHS, RHS> & expr )
   // -> decltype(GetTrialOperators( std::declval<LHS>() ) + GetTrialOperators( std::declval<RHS>() ))
{
   auto lhs_ops = GetTrialOperators( expr.lhs );
   auto rhs_ops = GetTrialOperators( expr.rhs );

   return lhs_ops + rhs_ops;
}

template < FieldExpr LHS, FieldExpr RHS >
auto GetTrialOperators( const InnerExpr<LHS, RHS> & expr )
   // -> decltype(GetTrialOperators( std::declval<LHS>() ) + GetTrialOperators( std::declval<RHS>() ))
{
   auto lhs_ops = GetTrialOperators( expr.lhs );
   auto rhs_ops = GetTrialOperators( expr.rhs );

   return lhs_ops + rhs_ops;
}

template < StaticString Name >
auto GetTrialOperators( const FiniteElementField<Name> & expr )
   // -> Empty
{
   return Empty{};
}

template < FieldExpr Expr >
auto GetTrialOperators( const JumpExpr<Expr> & expr )
   // -> decltype(GetTrialOperators( std::declval<Expr>() ))
{
   return GetTrialOperators( expr.expr );
}

template < FieldExpr LHS, FieldExpr RHS >
auto GetTrialOperators( const MultFieldExpr<LHS, RHS> & expr )
   // -> decltype(GetTrialOperators( std::declval<LHS>() ) + GetTrialOperators( std::declval<RHS>() ))
{
   auto lhs_ops = GetTrialOperators( expr.lhs );
   auto rhs_ops = GetTrialOperators( expr.rhs );

   return lhs_ops + rhs_ops;
}

template < FieldExpr LHS, FieldExpr RHS >
auto GetTrialOperators( const ProductExpr<LHS, RHS> & expr )
   // -> decltype(GetTrialOperators( std::declval<LHS>() ) + GetTrialOperators( std::declval<RHS>() ))
{
   auto lhs_ops = GetTrialOperators( expr.lhs );
   auto rhs_ops = GetTrialOperators( expr.rhs );

   return lhs_ops + rhs_ops;
}

template < FieldExpr AdvExpr, FieldExpr Expr >
auto GetTrialOperators( const UpwindExpr<AdvExpr, Expr> & expr )
   // -> decltype(GetTrialOperators( std::declval<AdvExpr>() ) + GetTrialOperators( std::declval<Expr>() ))
{
   auto adv_ops = GetTrialOperators( expr.adv_expr );
   auto expr_ops = GetTrialOperators( expr.expr );

   return adv_ops + expr_ops;
}

auto GetTrialOperators( const Normal & expr )
   // -> Empty
{
   return Empty{};
}

template < StaticString Name, FieldShape Shape >
auto GetTrialOperators( const TestSpace<Name, Shape> & expr )
   // -> Empty
{
   return Empty{};
}

template < DomainExpr Domain, FieldExpr Expr >
auto GetTrialOperators( const Integrand<Domain, Expr> & expr )
   // -> decltype(GetTrialOperators( std::declval<Expr>() ))
{
   return GetTrialOperators( expr.expr );
}

} // namespace gendil
