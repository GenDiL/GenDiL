#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
// #include "gendil/FiniteElementMethod/WeakForm/gettrialoperators.hpp" // your file
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"

namespace gendil
{

//-------------------------------------------
// Tiny typelist utilities
//-------------------------------------------
template<class... Ts> struct type_list {};

template<class A, class B> struct concat;
template<class... As, class... Bs>
struct concat<type_list<As...>, type_list<Bs...>> { using type = type_list<As..., Bs...>; };
template<class A, class B>
using concat_t = typename concat<A,B>::type;

template<class T, class List> struct contains;
template<class T> struct contains<T, type_list<>> : std::false_type {};
template<class T, class U, class... Rest>
struct contains<T, type_list<U, Rest...>>
  : std::bool_constant<std::is_same_v<T,U> || contains<T, type_list<Rest...>>::value> {};

template<class List> struct unique;
template<> struct unique<type_list<>> { using type = type_list<>; };
template<class T, class... Rest>
struct unique<type_list<T, Rest...>> {
  using tail_u = typename unique<type_list<Rest...>>::type;
  using type = std::conditional_t<
    contains<T, tail_u>::value,
    tail_u,
    concat_t<type_list<T>, tail_u>
  >;
};
template<class List>
using unique_t = typename unique<List>::type;

//-------------------------------------------
// Collect FE fields used in an expression
//-------------------------------------------
template<class Expr> struct fe_field_deps { using type = type_list<>; };

template<StaticString Name>
struct fe_field_deps<FiniteElementField<Name>> { using type = type_list<NameTag<Name>>; };

template<StaticString Name, FieldShape Shape>
struct fe_field_deps<TrialSpace<Name, Shape>> { using type = type_list<>; };

template<StaticString Name, FieldShape Shape>
struct fe_field_deps<TestSpace<Name, Shape>> { using type = type_list<>; };

template<> struct fe_field_deps<Normal> { using type = type_list<>; };

template<FieldExpr Expr>
struct fe_field_deps<AverageExpr<Expr>> { using type = typename fe_field_deps<Expr>::type; };

template<FieldExpr Expr>
struct fe_field_deps<JumpExpr<Expr>> { using type = typename fe_field_deps<Expr>::type; };

template<FieldExpr Expr>
struct fe_field_deps<GradientExpr<Expr>> { using type = typename fe_field_deps<Expr>::type; };

template<FieldExpr L, FieldExpr R>
struct fe_field_deps<AddFieldExpr<L,R>> { using type = concat_t<typename fe_field_deps<L>::type, typename fe_field_deps<R>::type>; };

template<FieldExpr L, FieldExpr R>
struct fe_field_deps<MultFieldExpr<L,R>> { using type = concat_t<typename fe_field_deps<L>::type, typename fe_field_deps<R>::type>; };

template<FieldExpr L, FieldExpr R>
struct fe_field_deps<DotExpr<L,R>> { using type = concat_t<typename fe_field_deps<L>::type, typename fe_field_deps<R>::type>; };

template<FieldExpr L, FieldExpr R>
struct fe_field_deps<ProductExpr<L,R>> { using type = concat_t<typename fe_field_deps<L>::type, typename fe_field_deps<R>::type>; };

// template<StaticString Name, CoefficientInput... Inputs>
// struct fe_field_deps<Coefficient<Name, Inputs...>> { using type = concat_t<Inputs...>; };

template<DomainExpr Domain, FieldExpr Expr>
struct fe_field_deps<Integrand<Domain, Expr>> { using type = typename fe_field_deps<Expr>::type; };

template<class Expr>
using fe_field_deps_t = unique_t<typename fe_field_deps<Expr>::type>;

} // namespace gendil