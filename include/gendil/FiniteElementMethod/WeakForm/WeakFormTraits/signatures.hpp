// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/expressionsemantics.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/domaintraits.hpp"

namespace gendil
{

namespace weak_form_traits
{
// Extract a field only after its list is known to contain exactly one entry.
// Empty and coupled lists deliberately provide no field or sentinel name.
template<class Fields> struct only_field;
template<class Field> struct only_field<type_list<Field>> { using type = Field; };
template<class Fields> using only_field_t = typename only_field<Fields>::type;

// Names of the sole declared trial/test fields, including declarations retained
// by transformations. These are not queries for callable inputs or actual reads.
template<class Form>
   requires (list_size<typename form_symbols_t<Form>::active_declarations>::value == 1)
inline constexpr auto only_trial_field_name_v =
   only_field_t<typename form_symbols_t<Form>::active_declarations>::name;

template<class Form>
   requires (list_size<typename form_symbols_t<Form>::test_declarations>::value == 1)
inline constexpr auto only_test_field_name_v =
   only_field_t<typename form_symbols_t<Form>::test_declarations>::name;

template<class List, StaticString Name> struct all_named;
template<class... Fields, StaticString Name>
struct all_named<type_list<Fields...>, Name>
   : std::bool_constant<((Fields::name == Name) && ...)> {};

// A residual's reads and writes must belong to its sole declared pair.
// Assembly admission and current-row reuse impose their own additional rules.
template<class Signature, class Inputs = typename Signature::declared_inputs,
         class Outputs = typename Signature::declared_outputs>
struct has_single_field_pair : std::false_type {};
template<class Signature, class Input, class Output>
struct has_single_field_pair<Signature, type_list<Input>, type_list<Output>>
   : std::bool_constant<
        Signature::valid &&
        Signature::result_kind == FormResultKind::Residual &&
        all_named<typename Signature::active_inputs, Input::name>::value &&
        all_named<typename Signature::test_outputs, Output::name>::value> {};

// Formal input placement is retained by a split even when its current terms
// have no reads. Domain and side relationships remain type-only metadata.
template<class Relations, class Domain> struct callable_relations_for;
template<class Domain>
struct callable_relations_for<type_list<>, Domain> { using type = side_requirements<>; };
template<class Key, class Sides, class... Rest, class Domain>
struct callable_relations_for<type_list<Entry<Key, Sides>, Rest...>, Domain>
{
   using tail = typename callable_relations_for<type_list<Rest...>, Domain>::type;
   using type = std::conditional_t<std::is_same_v<Key, Domain>, merge_sides<Sides, tail>, tail>;
};
template<class Relations> struct callable_relation_inputs;
template<class... Relations>
struct callable_relation_inputs<type_list<Relations...>>
{
   using type = merge_named_field_requirement_list_t<
      concat_many_t<typename Relations::value_type::all...>>;
};
template<class Actual, class Retained> struct form_callable_inputs
{
   using type = merge_named_field_requirement_list_t<concat_t<Actual,
      typename callable_relation_inputs<Retained>::type>>;
};
template<class Actual>
struct form_callable_inputs<Actual, type_list<>> { using type = Actual; };
template<class Analysis, class = void>
struct logical_sides { using type = typename Analysis::output_sides; };
template<class Analysis>
struct logical_sides<Analysis, std::void_t<typename Analysis::logical_output_sides>>
{ using type = typename Analysis::logical_output_sides; };

template<class Domain>
inline constexpr auto domain_semantic_entity_v =
   is_cells_domain<Domain>::value ? SemanticEntity::Cell :
   is_boundary_facets_domain<Domain>::value ? SemanticEntity::BoundaryFacet :
   is_interior_facets_domain<Domain>::value ? SemanticEntity::InteriorFacet :
   SemanticEntity::Unspecified;
}

/** A semantic summary for one original integration term, before grouping. */
template<class IntegrandType, class Symbols = form_symbols_t<IntegrandType>>
struct integrand_signature : weak_form_traits::expose_semantics<weak_form_traits::empty_semantics>
{
   using integrand_type = std::remove_cvref_t<IntegrandType>;
   using domain_type = void;
   using declared_inputs = type_list<>;
   using declared_outputs = type_list<>;
   using callable_inputs = type_list<>;
   using callable_input_relations = type_list<>;
   using logical_output_sides = weak_form_traits::side_requirements<>;
   using logical_outputs = type_list<>;
   template<SemanticSide Side> using logical_outputs_on = type_list<>;
   static constexpr bool is_zero = false;
   static constexpr bool has_zero_terms = false;
   static constexpr bool supported = false;
   static constexpr auto variance = SemanticVariance::Unsupported;
   static constexpr auto symbolic_shape = SemanticShape::Unsupported;
   static constexpr unsigned possible_shapes = 0u;
   static constexpr auto test_linearity = SemanticTestLinearity::Unsupported;
   static constexpr auto active_affinity = ActiveAffinity::Unknown;
   static constexpr auto result_kind = FormResultKind::Invalid;
   static constexpr bool valid = false;
};

template<DomainExpr Domain, FieldExpr E, class Symbols>
struct integrand_signature<Integrand<Domain, E>, Symbols>
 : expression_semantics_t<E, Symbols, weak_form_traits::domain_semantic_entity_v<Domain>>
{
   using integrand_type = Integrand<Domain, E>;
   using analysis = expression_semantics_t<E, Symbols, weak_form_traits::domain_semantic_entity_v<Domain>>;
   static constexpr auto entity = weak_form_traits::domain_semantic_entity_v<Domain>;
   using domain_type = Domain;
   static constexpr bool is_zero = weak_form_traits::is_zero_expr_v<E>;
   static constexpr bool has_zero_terms = is_zero;
   using callable_inputs = typename analysis::active_inputs;
   using callable_input_relations = std::conditional_t<is_empty_type_list_v<callable_inputs>,
      type_list<>, type_list<Entry<Domain, typename analysis::active_sides>>>;
   using logical_output_sides = typename weak_form_traits::logical_sides<analysis>::type;
   using logical_outputs = typename logical_output_sides::all;
   template<SemanticSide Side> using logical_outputs_on = typename logical_output_sides::template on<Side>;
   using geometry_sides = weak_form_traits::merge_geometry<typename analysis::geometry_sides,
      weak_form_traits::one_side_geometry<SemanticSide::Unqualified,
         is_zero ? GeometryRequirement::None : GeometryRequirement::IntegrationMeasure>>;
   static constexpr auto geometry = geometry_sides::all;
   template<SemanticSide Side> static constexpr auto geometry_on = geometry_sides::template on<Side>;
   using declared_inputs = typename form_symbols_t<E>::active_declarations;
   using declared_outputs = typename form_symbols_t<E>::test_declarations;
   static constexpr bool scalar_shape =
      (analysis::possible_shapes & weak_form_traits::shape_options(SemanticShape::Scalar)) != 0u;
   static constexpr bool resolved_shape = analysis::symbolic_shape != SemanticShape::Unresolved;
   using shape_obligations = std::conditional_t<resolved_shape,
      typename analysis::shape_obligations,
      concat_t<typename analysis::shape_obligations, type_list<ShapeObligation<ShapeOperation::ScalarIntegrand, E>>>>;
   static constexpr bool unqualified_named_reads =
      !is_empty_type_list_v<typename analysis::template active_inputs_on<SemanticSide::Unqualified>> ||
      !is_empty_type_list_v<typename analysis::template bound_inputs_on<SemanticSide::Unqualified>> ||
      !is_empty_type_list_v<typename analysis::template test_outputs_on<SemanticSide::Unqualified>>;
   static constexpr bool valid_entity =
      (is_cells_domain<Domain>::value && !analysis::requires_facet) ||
      (is_boundary_facets_domain<Domain>::value && !analysis::requires_two_sides) ||
      (is_interior_facets_domain<Domain>::value && !unqualified_named_reads);
   static constexpr bool valid = Symbols::valid && analysis::valid && valid_entity && scalar_shape &&
      (analysis::test_linearity == SemanticTestLinearity::TestFree ||
       weak_form_traits::list_size<logical_outputs>::value == 1);
   static constexpr auto result_kind = !valid ? FormResultKind::Invalid :
      analysis::test_linearity == SemanticTestLinearity::TestFree ? FormResultKind::Scalar : FormResultKind::Residual;
};

template<class IntegrandType, class Symbols = form_symbols_t<IntegrandType>>
using integrand_signature_t = integrand_signature<std::remove_cvref_t<IntegrandType>, Symbols>;

namespace weak_form_traits
{
template<class T> struct integrand_sequence { using type = type_list<T>; };
template<class... Terms>
struct integrand_sequence<std::tuple<Terms...>>
{
   using type = concat_many_t<typename integrand_sequence<Terms>::type...>;
};
template<class Terms, class Symbols>
struct integrand_sequence<SumFormExpr<Terms, Symbols>> : integrand_sequence<Terms> {};
template<class Terms, class Symbols> struct summarize_form;
template<class... Terms, class Symbols>
struct summarize_form<type_list<Terms...>, Symbols>
   : expose_semantics<merge_information<integrand_signature_t<Terms, Symbols>...>>
{
   using integrands = type_list<integrand_signature_t<Terms, Symbols>...>;
   static constexpr bool is_zero = sizeof...(Terms) != 0 && (integrand_signature_t<Terms, Symbols>::is_zero && ...);
   static constexpr bool has_zero_terms = (integrand_signature_t<Terms, Symbols>::is_zero || ...);
   using callable_input_relations = concat_many_t<
      typename integrand_signature_t<Terms, Symbols>::callable_input_relations...,
      typename Symbols::declarations::retained_callable_input_relations>;
   using callable_inputs = typename form_callable_inputs<
      typename merge_information<integrand_signature_t<Terms, Symbols>...>::active_sides::all,
      typename Symbols::declarations::retained_callable_input_relations>::type;
   using logical_output_sides = merge_sides<typename integrand_signature_t<Terms, Symbols>::logical_output_sides...>;
   using logical_outputs = typename logical_output_sides::all;
   template<SemanticSide Side> using logical_outputs_on = typename logical_output_sides::template on<Side>;
   static constexpr auto active_affinity = [] {
      auto result = ActiveAffinity::Independent;
      bool has_term = false;
      ([&] {
         if constexpr (!integrand_signature_t<Terms, Symbols>::is_zero)
         {
            constexpr auto affinity = integrand_signature_t<Terms, Symbols>::active_affinity;
            result = has_term ? sum_affinity(result, affinity) : affinity;
            has_term = true;
         }
      }(), ...);
      return result;
   }();
   using shape_obligations = unique_t<concat_many_t<typename integrand_signature_t<Terms, Symbols>::shape_obligations...>>;
   static constexpr bool all_scalar = ((integrand_signature_t<Terms, Symbols>::result_kind == FormResultKind::Scalar) && ...);
   static constexpr bool all_residual = ((integrand_signature_t<Terms, Symbols>::result_kind == FormResultKind::Residual) && ...);
   static constexpr unsigned possible_shapes = (0u | ... | integrand_signature_t<Terms, Symbols>::possible_shapes);
   static constexpr auto symbolic_shape = summarize_shape(possible_shapes);
   // Different integrands can have different output names while remaining
   // jointly linear in the complete tuple of tests.
   static constexpr auto test_linearity = all_scalar ? SemanticTestLinearity::TestFree :
      all_residual ? SemanticTestLinearity::Linear : SemanticTestLinearity::Unsupported;
   static constexpr bool valid = sizeof...(Terms) != 0 && Symbols::valid &&
      (integrand_signature_t<Terms, Symbols>::valid && ...) && (all_scalar || all_residual);
   static constexpr auto result_kind = !valid ? FormResultKind::Invalid :
      all_scalar ? FormResultKind::Scalar : FormResultKind::Residual;
};
// Select already-resolved signatures without changing complete-form field roles.
// Validation includes typed zeros to check their retained dimensions and domains.
template<class Signatures, StaticString Name, SemanticEntity Entity = SemanticEntity::Unspecified,
         bool IncludeZero = false>
struct domain_signatures;
template<class... Signatures, StaticString Name, SemanticEntity Entity, bool IncludeZero>
struct domain_signatures<type_list<Signatures...>, Name, Entity, IncludeZero>
{
   using type = concat_many_t<std::conditional_t<
      Signatures::domain_type::name == Name &&
      (IncludeZero || !Signatures::is_zero) &&
      (Entity == SemanticEntity::Unspecified || Signatures::entity == Entity),
      type_list<Signatures>, type_list<>>...>;
};
} // namespace weak_form_traits

/** Formal declarations survive transforms; actual reads describe current terms. */
template<class Form>
struct form_signature : weak_form_traits::summarize_form<
   typename weak_form_traits::integrand_sequence<std::remove_cvref_t<Form>>::type, form_symbols_t<Form>>
{
   using symbols = form_symbols_t<Form>;
   using formal_inputs = typename symbols::active_declarations;
   using formal_outputs = typename symbols::test_declarations;
};
template<class Form>
using form_signature_t = form_signature<std::remove_cvref_t<Form>>;

// True for a nonempty integrand/form (or selected tuple) containing only typed
// zero terms. This does not infer zeros from coefficient values or replace
// semantic validation; retained input/output dimensions still apply.
template<class Form>
inline constexpr bool is_zero = form_signature_t<Form>::is_zero;

// Does this domain/entity selection contain any integrand to evaluate?
// Nonzero excludes typed-zero terms, not coefficients that evaluate to zero.
// Unspecified selects all entity kinds; complete-form validation remains separate.
template<class WeakForm, StaticString DomainName,
         SemanticEntity Entity = SemanticEntity::Unspecified>
inline constexpr bool has_nonzero_integrands =
   !is_empty_type_list_v<typename weak_form_traits::domain_signatures<
      typename form_signature_t<WeakForm>::integrands, DomainName, Entity>::type>;

} // namespace gendil
