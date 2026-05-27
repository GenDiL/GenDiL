#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakform.hpp"

#include <type_traits>
#include <utility>

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

template<class... Lists>
struct concat_many;

template<>
struct concat_many<> { using type = type_list<>; };

template<class List>
struct concat_many<List> { using type = List; };

template<class First, class Second, class... Rest>
struct concat_many<First, Second, Rest...>
{
   using type = typename concat_many<concat_t<First, Second>, Rest...>::type;
};

template<class... Lists>
using concat_many_t = typename concat_many<Lists...>::type;

// Forward declaration for the optional matrix-vector expression extension.
template<typename, typename> struct MatVecExpr;

// ---------------------------------------------------------------------------
// Named-field requirement model
// ---------------------------------------------------------------------------

enum class NamedFieldProvenance : unsigned
{
   None                    = 0u,
   CoefficientInput        = 1u,
   FiniteElementExpression = 2u,
   ActiveTrial             = 4u
};

constexpr NamedFieldProvenance operator+(NamedFieldProvenance a, NamedFieldProvenance b)
{
   return NamedFieldProvenance(unsigned(a) | unsigned(b));
}

constexpr bool has_provenance(NamedFieldProvenance provenance, NamedFieldProvenance query)
{
   return (unsigned(provenance) & unsigned(query)) != 0u;
}

template<StaticString Name, OperatorMask Mask, NamedFieldProvenance Provenance>
struct NamedFieldRequirement
{
   static constexpr auto name = Name;
   static constexpr OperatorMask mask = Mask;
   static constexpr NamedFieldProvenance provenance = Provenance;
};

template<class Req>
using named_field_requirement_name_t = NameTag<Req::name>;

template<class LHS, class RHS>
struct merge_named_field_requirement
{
   static_assert(LHS::name == RHS::name,
      "Named field requirements can only be merged when they have the same name.");

   using type = NamedFieldRequirement<
      LHS::name,
      LHS::mask + RHS::mask,
      LHS::provenance + RHS::provenance>;
};

template<class LHS, class RHS>
using merge_named_field_requirement_t =
   typename merge_named_field_requirement<LHS, RHS>::type;

template<class Req, class List>
struct merge_requirement_into_list;

template<class Req>
struct merge_requirement_into_list<Req, type_list<>>
{
   using type = type_list<Req>;
};

template<bool SameName, class Req, class Head, class TailList>
struct merge_requirement_into_nonempty_list;

template<class Req, class Head, class... Tail>
struct merge_requirement_into_nonempty_list<true, Req, Head, type_list<Tail...>>
{
   using type = type_list<merge_named_field_requirement_t<Req, Head>, Tail...>;
};

template<class Req, class Head, class... Tail>
struct merge_requirement_into_nonempty_list<false, Req, Head, type_list<Tail...>>
{
   using merged_tail = typename merge_requirement_into_list<Req, type_list<Tail...>>::type;
   using type = concat_t<type_list<Head>, merged_tail>;
};

template<class Req, class Head, class... Tail>
struct merge_requirement_into_list<Req, type_list<Head, Tail...>>
   : merge_requirement_into_nonempty_list<(Req::name == Head::name), Req, Head, type_list<Tail...>>
{};

template<class List>
struct merge_named_field_requirement_list;

template<>
struct merge_named_field_requirement_list<type_list<>>
{
   using type = type_list<>;
};

template<class Req, class... Rest>
struct merge_named_field_requirement_list<type_list<Req, Rest...>>
{
   using merged_rest = typename merge_named_field_requirement_list<type_list<Rest...>>::type;
   using type = typename merge_requirement_into_list<Req, merged_rest>::type;
};

template<class List>
using merge_named_field_requirement_list_t =
   typename merge_named_field_requirement_list<List>::type;

template<class List, NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance;

template<NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance<type_list<>, Provenance>
{
   using type = type_list<>;
};

template<class Req, class... Rest, NamedFieldProvenance Provenance>
struct filter_requirements_by_provenance<type_list<Req, Rest...>, Provenance>
{
   using filtered_rest =
      typename filter_requirements_by_provenance<type_list<Rest...>, Provenance>::type;

   using type = std::conditional_t<
      has_provenance(Req::provenance, Provenance),
      concat_t<type_list<Req>, filtered_rest>,
      filtered_rest>;
};

template<class List, NamedFieldProvenance Provenance>
using filter_requirements_by_provenance_t =
   typename filter_requirements_by_provenance<List, Provenance>::type;

template<class List>
struct named_field_requirement_names;

template<>
struct named_field_requirement_names<type_list<>>
{
   using type = type_list<>;
};

template<class Req, class... Rest>
struct named_field_requirement_names<type_list<Req, Rest...>>
{
   using rest_names = typename named_field_requirement_names<type_list<Rest...>>::type;
   using type = unique_t<concat_t<type_list<NameTag<Req::name>>, rest_names>>;
};

template<class List>
using named_field_requirement_names_t =
   typename named_field_requirement_names<List>::type;

template<class List, StaticString Name>
struct contains_named_field_requirement;

template<StaticString Name>
struct contains_named_field_requirement<type_list<>, Name> : std::false_type {};

template<class Req, class... Rest, StaticString Name>
struct contains_named_field_requirement<type_list<Req, Rest...>, Name>
   : std::bool_constant<
        (Req::name == Name) ||
        contains_named_field_requirement<type_list<Rest...>, Name>::value>
{};

template<class List, StaticString Name>
inline constexpr bool contains_named_field_requirement_v =
   contains_named_field_requirement<List, Name>::value;

template<class List, StaticString Name>
struct find_named_field_requirement;

template<StaticString Name>
struct find_named_field_requirement<type_list<>, Name>
{
   using type = void;
};

template<bool SameName, class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty;

template<class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty<true, Req, RestList, Name>
{
   using type = Req;
};

template<class Req, class RestList, StaticString Name>
struct find_named_field_requirement_nonempty<false, Req, RestList, Name>
{
   using type = typename find_named_field_requirement<RestList, Name>::type;
};

template<class Req, class... Rest, StaticString Name>
struct find_named_field_requirement<type_list<Req, Rest...>, Name>
   : find_named_field_requirement_nonempty<(Req::name == Name), Req, type_list<Rest...>, Name>
{};

template<class List, StaticString Name>
using find_named_field_requirement_t =
   typename find_named_field_requirement<List, Name>::type;

template<class List>
struct is_empty_type_list : std::false_type {};

template<>
struct is_empty_type_list<type_list<>> : std::true_type {};

template<class List>
inline constexpr bool is_empty_type_list_v = is_empty_type_list<List>::value;

// ---------------------------------------------------------------------------
// Coefficient input requirements
// ---------------------------------------------------------------------------

template<class Input> struct coefficient_input_named_field_requirements
{
   using type = type_list<>;
};

template<StaticString Name>
struct coefficient_input_named_field_requirements<FieldValue<Name>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Values,
         NamedFieldProvenance::CoefficientInput>>;
};

template<StaticString Name>
struct coefficient_input_named_field_requirements<FieldGradient<Name>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Gradients,
         NamedFieldProvenance::CoefficientInput>>;
};

template<class... Inputs>
struct coefficient_inputs_named_field_requirements
{
   using type = concat_many_t<typename coefficient_input_named_field_requirements<Inputs>::type...>;
};

template<class... Inputs>
using coefficient_inputs_named_field_requirements_t =
   typename coefficient_inputs_named_field_requirements<Inputs...>::type;

// ---------------------------------------------------------------------------
// Raw named-field requirements found directly in an expression tree.
// This intentionally excludes active-trial requirements, which are derived from
// the existing weak-form requirements traits below.
// ---------------------------------------------------------------------------

template<class Expr> struct raw_named_field_requirements { using type = type_list<>; };

template<StaticString Name>
struct raw_named_field_requirements<FiniteElementField<Name>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Values,
         NamedFieldProvenance::FiniteElementExpression>>;
};

template<StaticString Name>
struct raw_named_field_requirements<GradientExpr<FiniteElementField<Name>>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Gradients,
         NamedFieldProvenance::FiniteElementExpression>>;
};

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct raw_named_field_requirements<Coefficient<Name, Shape, Fn, Inputs...>>
{
   using type = coefficient_inputs_named_field_requirements_t<Inputs...>;
};

template<FieldExpr Expr>
struct raw_named_field_requirements<NegExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<FieldExpr Expr>
struct raw_named_field_requirements<AverageExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<FieldExpr Expr>
struct raw_named_field_requirements<JumpExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<FieldExpr Expr>
struct raw_named_field_requirements<GradientExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<FieldExpr AdvExpr, FieldExpr Expr>
struct raw_named_field_requirements<UpwindExpr<AdvExpr, Expr>>
{
   using type = concat_t<
      typename raw_named_field_requirements<AdvExpr>::type,
      typename raw_named_field_requirements<Expr>::type>;
};

template<FieldExpr Head, FieldExpr... Tail>
struct raw_named_field_requirements<SumExpr<Head, Tail...>>
{
   using type = concat_many_t<typename raw_named_field_requirements<Head>::type,
                             typename raw_named_field_requirements<Tail>::type...>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct raw_named_field_requirements<MultFieldExpr<LHS,RHS>>
{
   using type = concat_t<typename raw_named_field_requirements<LHS>::type,
                         typename raw_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct raw_named_field_requirements<DotExpr<LHS,RHS>>
{
   using type = concat_t<typename raw_named_field_requirements<LHS>::type,
                         typename raw_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct raw_named_field_requirements<InnerExpr<LHS,RHS>>
{
   using type = concat_t<typename raw_named_field_requirements<LHS>::type,
                         typename raw_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct raw_named_field_requirements<OuterExpr<LHS,RHS>>
{
   using type = concat_t<typename raw_named_field_requirements<LHS>::type,
                         typename raw_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct raw_named_field_requirements<ProductExpr<LHS,RHS>>
{
   using type = concat_t<typename raw_named_field_requirements<LHS>::type,
                         typename raw_named_field_requirements<RHS>::type>;
};

template<class MatrixExpr, class VectorExpr>
struct raw_named_field_requirements<MatVecExpr<MatrixExpr, VectorExpr>>
{
   using type = concat_t<typename raw_named_field_requirements<MatrixExpr>::type,
                         typename raw_named_field_requirements<VectorExpr>::type>;
};

template<DomainExpr Domain, FieldExpr Expr>
struct raw_named_field_requirements<Integrand<Domain, Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<class Key, class T>
struct raw_named_field_requirements<Entry<Key, T>>
{
   using type = typename raw_named_field_requirements<T>::type;
};

template<class... Entries>
struct raw_named_field_requirements<StaticMap<Entries...>>
{
   using type = concat_many_t<typename raw_named_field_requirements<Entries>::type...>;
};

template<class Map>
struct raw_named_field_requirements<SumFormExpr<Map>>
{
   using type = typename raw_named_field_requirements<Map>::type;
};

template<class Expr>
using raw_named_field_requirements_t =
   typename raw_named_field_requirements<std::remove_cvref_t<Expr>>::type;

template<class Expr>
using finite_element_expr_named_field_requirements_t =
   merge_named_field_requirement_list_t<
      filter_requirements_by_provenance_t<
         raw_named_field_requirements_t<Expr>,
         NamedFieldProvenance::FiniteElementExpression>>;

template<class Expr>
using coefficient_named_field_requirements_t =
   merge_named_field_requirement_list_t<
      filter_requirements_by_provenance_t<
         raw_named_field_requirements_t<Expr>,
         NamedFieldProvenance::CoefficientInput>>;

// ---------------------------------------------------------------------------
// Active trial requirements are derived from the existing weak-form requirement
// machinery to avoid a second trial-space collector.
// ---------------------------------------------------------------------------

template<class Expr, bool HasActiveTrial>
struct active_trial_named_field_requirements_impl
{
   using type = type_list<>;
};

template<class Expr>
struct active_trial_named_field_requirements_impl<Expr, true>
{
   using E = std::remove_cvref_t<Expr>;
   using type = type_list<
      NamedFieldRequirement<
         requirements<E>::trial_name,
         requirements<E>::trial_mask,
         NamedFieldProvenance::ActiveTrial>>;
};

template<class Expr>
struct active_trial_named_field_requirements
{
   using E = std::remove_cvref_t<Expr>;
   static constexpr bool has_active_trial =
      requirements<E>::trial_name != StaticString{"Error"} &&
      requirements<E>::trial_mask != OperatorMask::None;

   using type = typename active_trial_named_field_requirements_impl<E, has_active_trial>::type;
};

template<class Expr>
using active_trial_named_field_requirements_t =
   typename active_trial_named_field_requirements<Expr>::type;

template<class Expr>
using input_named_field_requirements_t =
   merge_named_field_requirement_list_t<
      concat_t<raw_named_field_requirements_t<Expr>,
               active_trial_named_field_requirements_t<Expr>>>;

template<class Expr>
using interpolation_named_field_requirements_t =
   input_named_field_requirements_t<Expr>;

template<class Expr>
using named_field_requirements_t =
   input_named_field_requirements_t<Expr>;

template<class Expr>
inline constexpr bool has_side_dependent_named_field_inputs_v =
   !is_empty_type_list_v<named_field_requirements_t<Expr>>;

// ---------------------------------------------------------------------------
// Unqualified side-dependent named-field requirements.
//
// This collector is intentionally separate from the interpolation requirement
// collector above. Interpolation needs every named field/channel used anywhere
// in the expression. Interior-facet validation needs only the named-field uses
// that are not protected by an explicit side-selection wrapper such as
// average(...) or jump(...).
//
// average(E) and jump(E) clear the unqualified dependency set for the two
// interior-facet cases with explicit side semantics:
//   - test-free/side-evaluable expressions, where the operand is evaluated on
//     the minus and plus sides and then combined;
//   - test-linear pullback expressions, where GenDiL's current-side facet
//     convention evaluates coefficient inputs on the current side and the
//     opposite contribution is produced when the neighbor is current.
// ---------------------------------------------------------------------------

template<class Expr>
struct unqualified_side_dependent_named_field_requirements
{
   using type = type_list<>;
};

template<StaticString Name, FieldShape Shape>
struct unqualified_side_dependent_named_field_requirements<TrialSpace<Name, Shape>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Values,
         NamedFieldProvenance::ActiveTrial>>;
};

template<StaticString Name, FieldShape Shape>
struct unqualified_side_dependent_named_field_requirements<
   GradientExpr<TrialSpace<Name, Shape>>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Gradients,
         NamedFieldProvenance::ActiveTrial>>;
};

template<StaticString Name>
struct unqualified_side_dependent_named_field_requirements<FiniteElementField<Name>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Values,
         NamedFieldProvenance::FiniteElementExpression>>;
};

template<StaticString Name>
struct unqualified_side_dependent_named_field_requirements<
   GradientExpr<FiniteElementField<Name>>>
{
   using type = type_list<
      NamedFieldRequirement<
         Name,
         OperatorMask::Gradients,
         NamedFieldProvenance::FiniteElementExpression>>;
};

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct unqualified_side_dependent_named_field_requirements<
   Coefficient<Name, Shape, Fn, Inputs...>>
{
   using type = coefficient_inputs_named_field_requirements_t<Inputs...>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<NegExpr<Expr>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Expr>::type;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<AverageExpr<Expr>>
{
   using type = std::conditional_t<
      is_side_evaluable_v<Expr> || is_test_linear_v<Expr>,
      type_list<>,
      typename unqualified_side_dependent_named_field_requirements<Expr>::type>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<JumpExpr<Expr>>
{
   using type = std::conditional_t<
      is_side_evaluable_v<Expr> || is_test_linear_v<Expr>,
      type_list<>,
      typename unqualified_side_dependent_named_field_requirements<Expr>::type>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<GradientExpr<Expr>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Expr>::type;
};

template<FieldExpr AdvExpr, FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<
   UpwindExpr<AdvExpr, Expr>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<AdvExpr>::type,
      std::conditional_t<
         is_side_evaluable_v<Expr>,
         type_list<>,
         typename unqualified_side_dependent_named_field_requirements<Expr>::type>>;
};

template<FieldExpr Head, FieldExpr... Tail>
struct unqualified_side_dependent_named_field_requirements<
   SumExpr<Head, Tail...>>
{
   using type = concat_many_t<
      typename unqualified_side_dependent_named_field_requirements<Head>::type,
      typename unqualified_side_dependent_named_field_requirements<Tail>::type...>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct unqualified_side_dependent_named_field_requirements<
   MultFieldExpr<LHS,RHS>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<LHS>::type,
      typename unqualified_side_dependent_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct unqualified_side_dependent_named_field_requirements<DotExpr<LHS,RHS>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<LHS>::type,
      typename unqualified_side_dependent_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct unqualified_side_dependent_named_field_requirements<InnerExpr<LHS,RHS>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<LHS>::type,
      typename unqualified_side_dependent_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct unqualified_side_dependent_named_field_requirements<OuterExpr<LHS,RHS>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<LHS>::type,
      typename unqualified_side_dependent_named_field_requirements<RHS>::type>;
};

template<FieldExpr LHS, FieldExpr RHS>
struct unqualified_side_dependent_named_field_requirements<
   ProductExpr<LHS,RHS>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<LHS>::type,
      typename unqualified_side_dependent_named_field_requirements<RHS>::type>;
};

template<class MatrixExpr, class VectorExpr>
struct unqualified_side_dependent_named_field_requirements<
   MatVecExpr<MatrixExpr, VectorExpr>>
{
   using type = concat_t<
      typename unqualified_side_dependent_named_field_requirements<MatrixExpr>::type,
      typename unqualified_side_dependent_named_field_requirements<VectorExpr>::type>;
};

template<DomainExpr Domain, FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<
   Integrand<Domain, Expr>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Expr>::type;
};

template<class Key, class T>
struct unqualified_side_dependent_named_field_requirements<Entry<Key, T>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<T>::type;
};

template<class... Entries>
struct unqualified_side_dependent_named_field_requirements<
   StaticMap<Entries...>>
{
   using type = concat_many_t<
      typename unqualified_side_dependent_named_field_requirements<Entries>::type...>;
};

template<class Map>
struct unqualified_side_dependent_named_field_requirements<SumFormExpr<Map>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Map>::type;
};

template<class Expr>
using unqualified_side_dependent_named_field_requirements_t =
   merge_named_field_requirement_list_t<
      typename unqualified_side_dependent_named_field_requirements<
         std::remove_cvref_t<Expr>>::type>;

template<class Expr>
inline constexpr bool has_unqualified_side_dependent_inputs_v =
   !is_empty_type_list_v<
      unqualified_side_dependent_named_field_requirements_t<Expr>>;

template<class Req>
inline constexpr bool is_value_only_requirement_v =
   need_values(Req::mask) && !need_gradients(Req::mask);

template<class WeakFormContext, StaticString Name, bool HasField>
struct trace_continuous_named_field_impl : std::false_type {};

template<class WeakFormContext, StaticString Name>
struct trace_continuous_named_field_impl<WeakFormContext, Name, true>
{
   using WFC = std::remove_cvref_t<WeakFormContext>;
   using FieldView = std::remove_cvref_t<
      decltype(std::declval<const WFC&>().fe_fields.template get<
         FiniteElementFieldKey<Name>>())>;
   using Space = std::remove_cvref_t<
      decltype(std::declval<FieldView>().space)>;

   static constexpr bool value =
      std::is_same_v<typename Space::restriction_type, H1Restriction>;
};

template<class WeakFormContext, StaticString Name>
inline constexpr bool is_trace_continuous_named_field_v =
   trace_continuous_named_field_impl<
      std::remove_cvref_t<WeakFormContext>,
      Name,
      std::remove_cvref_t<WeakFormContext>::template has_fe_field<Name>()>::value;

template<class FaceContext, class = void>
struct face_context_is_conforming : std::false_type {};

template<class FaceContext>
struct face_context_is_conforming<
   FaceContext,
   std::void_t<
      typename std::remove_cvref_t<FaceContext>::minus_side_type,
      typename std::remove_cvref_t<FaceContext>::plus_side_type>>
   : std::bool_constant<
        std::remove_cvref_t<FaceContext>::minus_side_type::is_conforming &&
        std::remove_cvref_t<FaceContext>::plus_side_type::is_conforming>
{};

template<class FaceContext>
inline constexpr bool face_context_is_conforming_v =
   face_context_is_conforming<FaceContext>::value;

template<class Req, class WeakFormContext, class FaceContext, StaticString TrialName>
struct is_allowed_unqualified_interior_side_dependency
   : std::bool_constant<
        is_value_only_requirement_v<Req> &&
        Req::name != TrialName &&
        !has_provenance(Req::provenance, NamedFieldProvenance::ActiveTrial) &&
        std::remove_cvref_t<WeakFormContext>::template has_fe_field<Req::name>() &&
        is_trace_continuous_named_field_v<WeakFormContext, Req::name> &&
        face_context_is_conforming_v<FaceContext>>
{};

template<class List, class WeakFormContext, class FaceContext, StaticString TrialName>
struct all_unqualified_interior_side_dependencies_allowed;

template<class WeakFormContext, class FaceContext, StaticString TrialName>
struct all_unqualified_interior_side_dependencies_allowed<
   type_list<>, WeakFormContext, FaceContext, TrialName> : std::true_type {};

template<class Req, class... Rest, class WeakFormContext, class FaceContext, StaticString TrialName>
struct all_unqualified_interior_side_dependencies_allowed<
   type_list<Req, Rest...>, WeakFormContext, FaceContext, TrialName>
   : std::bool_constant<
        is_allowed_unqualified_interior_side_dependency<
           Req, WeakFormContext, FaceContext, TrialName>::value &&
        all_unqualified_interior_side_dependencies_allowed<
           type_list<Rest...>, WeakFormContext, FaceContext, TrialName>::value>
{};

template<class Expr, class WeakFormContext, class FaceContext>
inline constexpr bool has_invalid_unqualified_interior_side_dependencies_v =
   !all_unqualified_interior_side_dependencies_allowed<
      unqualified_side_dependent_named_field_requirements_t<Expr>,
      WeakFormContext,
      FaceContext,
      requirements<std::remove_cvref_t<Expr>>::trial_name>::value;

template<class Expr>
inline constexpr bool has_active_trial_coefficient_dependency_v =
   requirements<std::remove_cvref_t<Expr>>::trial_name != StaticString{"Error"} &&
   contains_named_field_requirement_v<
      coefficient_named_field_requirements_t<Expr>,
      requirements<std::remove_cvref_t<Expr>>::trial_name>;

// ---------------------------------------------------------------------------
// Plus-side physical geometry requirements for interior facets.
//
// Two-sided expression evaluation does not necessarily require two-sided
// physical geometry. Value traces need plus-side field data and reference-side
// evaluation, but they do not need the plus cell Jacobian. For this milestone,
// plus-side Jacobians are required only when an expression is actually evaluated
// in the plus-side context and asks for a physical gradient there.
// ---------------------------------------------------------------------------

template<class Input>
struct coefficient_input_requires_plus_side_jacobian : std::false_type {};

template<StaticString Name>
struct coefficient_input_requires_plus_side_jacobian<FieldGradient<Name>>
   : std::true_type {};

template<class Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement
{
   static constexpr bool value = false;
};

template<StaticString Name, FieldShape Shape>
struct plus_side_jacobian_requirement<
   GradientExpr<TrialSpace<Name, Shape>>, true>
{
   static constexpr bool value = true;
};

template<StaticString Name>
struct plus_side_jacobian_requirement<
   GradientExpr<FiniteElementField<Name>>, true>
{
   static constexpr bool value = true;
};

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct plus_side_jacobian_requirement<
   Coefficient<Name, Shape, Fn, Inputs...>, true>
{
   static constexpr bool value =
      (coefficient_input_requires_plus_side_jacobian<Inputs>::value || ...);
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<NegExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<GradientExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<AverageExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value = [] {
      if constexpr (is_side_evaluable_v<Expr>)
      {
         return plus_side_jacobian_requirement<Expr, true>::value;
      }
      else
      {
         return plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
      }
   }();
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<JumpExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value = [] {
      if constexpr (is_side_evaluable_v<Expr>)
      {
         return plus_side_jacobian_requirement<Expr, true>::value;
      }
      else
      {
         return plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
      }
   }();
};

template<FieldExpr AdvExpr, FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   UpwindExpr<AdvExpr, Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<AdvExpr, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<bool InPlusSideContext, FieldExpr Head, FieldExpr... Tail>
struct plus_side_jacobian_requirement<
   SumExpr<Head, Tail...>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Head, InPlusSideContext>::value ||
      (plus_side_jacobian_requirement<Tail, InPlusSideContext>::value || ...);
};

template<FieldExpr LHS, FieldExpr RHS, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   MultFieldExpr<LHS, RHS>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<LHS, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<RHS, InPlusSideContext>::value;
};

template<FieldExpr LHS, FieldExpr RHS, bool InPlusSideContext>
struct plus_side_jacobian_requirement<DotExpr<LHS, RHS>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<LHS, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<RHS, InPlusSideContext>::value;
};

template<FieldExpr LHS, FieldExpr RHS, bool InPlusSideContext>
struct plus_side_jacobian_requirement<InnerExpr<LHS, RHS>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<LHS, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<RHS, InPlusSideContext>::value;
};

template<FieldExpr LHS, FieldExpr RHS, bool InPlusSideContext>
struct plus_side_jacobian_requirement<OuterExpr<LHS, RHS>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<LHS, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<RHS, InPlusSideContext>::value;
};

template<FieldExpr LHS, FieldExpr RHS, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   ProductExpr<LHS, RHS>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<LHS, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<RHS, InPlusSideContext>::value;
};

template<class MatrixExpr, class VectorExpr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   MatVecExpr<MatrixExpr, VectorExpr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<MatrixExpr, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<VectorExpr, InPlusSideContext>::value;
};

template<DomainExpr Domain, FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   Integrand<Domain, Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<class Key, class T, bool InPlusSideContext>
struct plus_side_jacobian_requirement<Entry<Key, T>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<T, InPlusSideContext>::value;
};

template<bool InPlusSideContext, class... Entries>
struct plus_side_jacobian_requirement<
   StaticMap<Entries...>, InPlusSideContext>
{
   static constexpr bool value =
      (plus_side_jacobian_requirement<Entries, InPlusSideContext>::value || ...);
};

template<class Map, bool InPlusSideContext>
struct plus_side_jacobian_requirement<SumFormExpr<Map>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Map, InPlusSideContext>::value;
};

template<class Expr>
inline constexpr bool requires_plus_side_jacobian_v =
   plus_side_jacobian_requirement<std::remove_cvref_t<Expr>, false>::value;

//-------------------------------------------
// Compatibility adapter: explicit FE fields used in an expression
//-------------------------------------------
template<class Expr> struct fe_field_deps
{
   using type = named_field_requirement_names_t<
      finite_element_expr_named_field_requirements_t<Expr>>;
};

template<class Expr>
using fe_field_deps_t = typename fe_field_deps<Expr>::type;

} // namespace gendil
