// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/zero.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/namedrequirements.hpp"
#include <tuple>
#include <type_traits>

namespace gendil
{

/** A declaration carried by syntax, independent of any registered space. */
template<StaticString Name, FieldShape Shape>
struct SymbolicFieldDeclaration
{
   static constexpr auto name = Name;
   static constexpr FieldShape shape = Shape;
};

namespace weak_form_traits
{
template<class List> struct list_size;
template<class... Ts>
struct list_size<type_list<Ts...>> : std::integral_constant<size_t, sizeof...(Ts)> {};

template<class List, StaticString Name> struct declares_name;
template<class... Declarations, StaticString Name>
struct declares_name<type_list<Declarations...>, Name>
   : std::bool_constant<((Declarations::name == Name) || ...)> {};

template<class List> struct consistent_declarations;
template<> struct consistent_declarations<type_list<>> : std::true_type {};
template<class Head, class... Tail>
struct consistent_declarations<type_list<Head, Tail...>>
   : std::bool_constant<
        (((Head::name != Tail::name) || (Head::shape == Tail::shape)) && ...) &&
        consistent_declarations<type_list<Tail...>>::value> {};

template<class Active, class Bound> struct disjoint_declarations;
template<class... Active, class Bound>
struct disjoint_declarations<type_list<Active...>, Bound>
   : std::bool_constant<(!contains<NameTag<Active::name>, Bound>::value && ...)> {};

template<bool Supported, class Active = type_list<>, class Tests = type_list<>,
         class Bound = type_list<>, class CallableRelations = type_list<>>
struct symbol_summary
{
   using type = symbol_summary;
   static constexpr bool supported = Supported;
   using active = Active;
   using tests = Tests;
   using bound = Bound;
   using retained_callable_input_relations = CallableRelations;
};

// An unrecognized extension is never mistaken for a dependency-free terminal.
template<class T> struct symbols_of : symbol_summary<false> {};

template<class... Summaries>
struct merge_symbols : symbol_summary<(Summaries::supported && ...),
   unique_t<concat_many_t<typename Summaries::active...>>,
   unique_t<concat_many_t<typename Summaries::tests...>>,
   unique_t<concat_many_t<typename Summaries::bound...>>,
   unique_t<concat_many_t<typename Summaries::retained_callable_input_relations...>>> {};

template<class Declarations, class Sides>
struct symbols_of<ZeroExpr<Declarations, Sides>> : Declarations {};

template<StaticString Name, FieldShape Shape>
struct symbols_of<TrialSpace<Name, Shape>>
   : symbol_summary<true, type_list<SymbolicFieldDeclaration<Name, Shape>>> {};
template<StaticString Name, FieldShape Shape>
struct symbols_of<TestSpace<Name, Shape>>
   : symbol_summary<true, type_list<>, type_list<SymbolicFieldDeclaration<Name, Shape>>> {};
template<StaticString Name>
struct symbols_of<FiniteElementField<Name>>
   : symbol_summary<true, type_list<>, type_list<>, type_list<NameTag<Name>>> {};
template<StaticString Name, FieldShape Shape, class Fn, CoefficientInput... Inputs>
struct symbols_of<Coefficient<Name, Shape, Fn, Inputs...>> : symbol_summary<true> {};
template<> struct symbols_of<ScaleExpr> : symbol_summary<true> {};
template<> struct symbols_of<Normal> : symbol_summary<true> {};

#define GENDIL_SYMBOL_UNARY(Node) \
template<FieldExpr E> struct symbols_of<Node<E>> : symbols_of<E> {};
GENDIL_SYMBOL_UNARY(GradientExpr)
GENDIL_SYMBOL_UNARY(DivergenceExpr)
GENDIL_SYMBOL_UNARY(NegExpr)
GENDIL_SYMBOL_UNARY(MinusTraceExpr)
GENDIL_SYMBOL_UNARY(PlusTraceExpr)
GENDIL_SYMBOL_UNARY(AverageExpr)
GENDIL_SYMBOL_UNARY(JumpExpr)
GENDIL_SYMBOL_UNARY(TransposeExpr)
#undef GENDIL_SYMBOL_UNARY

#define GENDIL_SYMBOL_BINARY(Node) \
template<FieldExpr L, FieldExpr R> \
struct symbols_of<Node<L, R>> : merge_symbols<symbols_of<L>, symbols_of<R>> {};
GENDIL_SYMBOL_BINARY(ProductExpr)
GENDIL_SYMBOL_BINARY(MultFieldExpr)
GENDIL_SYMBOL_BINARY(DotExpr)
GENDIL_SYMBOL_BINARY(InnerExpr)
GENDIL_SYMBOL_BINARY(OuterExpr)
GENDIL_SYMBOL_BINARY(UpwindExpr)
#undef GENDIL_SYMBOL_BINARY

template<FieldExpr Adv, typename Interior, typename Exterior>
struct symbols_of<BoundaryUpwindExpr<Adv, Interior, Exterior>>
   : merge_symbols<symbols_of<Adv>,
      std::conditional_t<is_boundary_upwind_zero_v<Interior>, symbol_summary<true>, symbols_of<Interior>>,
      std::conditional_t<is_boundary_upwind_zero_v<Exterior>, symbol_summary<true>, symbols_of<Exterior>>> {};

template<FieldExpr Head, FieldExpr... Tail>
struct symbols_of<SumExpr<Head, Tail...>> : merge_symbols<symbols_of<Head>, symbols_of<Tail>...> {};
template<DomainExpr Domain, FieldExpr E>
struct symbols_of<Integrand<Domain, E>> : symbols_of<E> {};
template<class... Terms>
struct symbols_of<std::tuple<Terms...>>
   : merge_symbols<symbols_of<std::remove_cvref_t<Terms>>...> {};
template<class Terms, class Declarations>
struct symbols_of<SumFormExpr<Terms, Declarations>>
   : merge_symbols<symbols_of<Terms>, Declarations> {};
template<class Terms>
struct symbols_of<SumFormExpr<Terms, void>> : symbols_of<Terms> {};
} // namespace weak_form_traits

/** Collect declarations before classifying any coefficient's named reads. */
template<class Form>
struct form_symbols
{
   using source = weak_form_traits::symbols_of<std::remove_cvref_t<Form>>;
   using active_declarations = unique_t<typename source::active>;
   using test_declarations = unique_t<typename source::tests>;
   using bound_declarations = unique_t<typename source::bound>;
   // Retain declarations across transformations without retaining old AST or
   // callable types. Distinct conflicting declarations and unsupported origins
   // remain visible even if a later transformation removes their syntax.
   using plain_declarations = weak_form_traits::symbol_summary<source::supported,
      active_declarations, test_declarations, bound_declarations>;
   using declarations = weak_form_traits::symbol_summary<source::supported,
      active_declarations, test_declarations, bound_declarations,
      typename source::retained_callable_input_relations>;
   static constexpr bool supported = source::supported;
   static constexpr bool compatible_shapes = weak_form_traits::consistent_declarations<
      concat_t<active_declarations, test_declarations>>::value;
   static constexpr bool compatible_roles = weak_form_traits::disjoint_declarations<
      active_declarations, bound_declarations>::value;
   static constexpr bool valid = supported && compatible_shapes && compatible_roles;

   template<StaticString Name>
   static constexpr bool is_active = weak_form_traits::declares_name<active_declarations, Name>::value;
   template<StaticString Name>
   static constexpr bool is_test = weak_form_traits::declares_name<test_declarations, Name>::value;
   template<StaticString Name>
   static constexpr bool is_explicitly_bound = contains<NameTag<Name>, bound_declarations>::value;
};

template<class Form>
using form_symbols_t = form_symbols<std::remove_cvref_t<Form>>;

} // namespace gendil
