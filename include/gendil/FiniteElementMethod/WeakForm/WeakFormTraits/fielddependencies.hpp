#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticstring.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/coefficientinputs.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/namedrequirements.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

// Syntax queries preserve the origin of each field occurrence for direct trace
// checks and provenance inspection. Numerical execution uses resolved signatures
// instead: merging reads by name can lose which occurrence requested a gradient.

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
   // Match GetCoefficientInput's normalization of descriptor types.
   using type = concat_many_t<typename coefficient_input_named_field_requirements<
      std::remove_cvref_t<Inputs>>::type...>;
};

template<class... Inputs>
using coefficient_inputs_named_field_requirements_t =
   typename coefficient_inputs_named_field_requirements<Inputs...>::type;

// ---------------------------------------------------------------------------
// Raw named-field requirements found directly in an expression tree.
// This intentionally excludes explicit TrialSpace/TestSpace declarations; it
// retains the occurrence provenance of coefficient and bound-field syntax.
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
struct raw_named_field_requirements<MinusTraceExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<FieldExpr Expr>
struct raw_named_field_requirements<PlusTraceExpr<Expr>>
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

template<FieldExpr Adv, typename Interior, typename Exterior>
struct raw_named_field_requirements<BoundaryUpwindExpr<Adv, Interior, Exterior>>
{
   using type = concat_many_t<
      typename raw_named_field_requirements<Adv>::type,
      typename raw_named_field_requirements<Interior>::type,
      typename raw_named_field_requirements<Exterior>::type>;
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

template<FieldExpr Expr>
struct raw_named_field_requirements<TransposeExpr<Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
};

template<DomainExpr Domain, FieldExpr Expr>
struct raw_named_field_requirements<Integrand<Domain, Expr>>
{
   using type = typename raw_named_field_requirements<Expr>::type;
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
// Unqualified side-dependent named-field requirements.
//
// This collector preserves individual occurrences for direct interior-facet
// syntax checks. Interpolation consumes resolved signatures; these checks need
// only the named-field uses
// that are not protected by an explicit side-selection wrapper such as
// average(...) or jump(...).
//
// minus(E), plus(E), average(E), and jump(E) clear the unqualified dependency
// set because they provide explicit canonical side semantics on interior
// facets. Bare FE trial/coefficient quantities are otherwise ambiguous on a
// two-trace face and must not silently fall back to the minus side. Analytical
// coefficients with no FE-field inputs are side-independent because they have
// no named-field requirements here; FE-backed quantities remain side-dependent
// until explicitly traced.
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
struct unqualified_side_dependent_named_field_requirements<MinusTraceExpr<Expr>>
{
   using type = type_list<>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<PlusTraceExpr<Expr>>
{
   using type = type_list<>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<AverageExpr<Expr>>
{
   using type = type_list<>;
};

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<JumpExpr<Expr>>
{
   using type = type_list<>;
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

template<FieldExpr Adv, typename Interior, typename Exterior>
struct unqualified_side_dependent_named_field_requirements<
   BoundaryUpwindExpr<Adv, Interior, Exterior>>
{
   using type = concat_many_t<
      typename unqualified_side_dependent_named_field_requirements<Adv>::type,
      typename unqualified_side_dependent_named_field_requirements<Interior>::type,
      typename unqualified_side_dependent_named_field_requirements<Exterior>::type>;
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

template<FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<
   TransposeExpr<Expr>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Expr>::type;
};

template<DomainExpr Domain, FieldExpr Expr>
struct unqualified_side_dependent_named_field_requirements<
   Integrand<Domain, Expr>>
{
   using type =
      typename unqualified_side_dependent_named_field_requirements<Expr>::type;
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

// ---------------------------------------------------------------------------
// Interior-facet test traces must be explicit.
//
// Unlike coefficient/trial side dependencies above, this validation is purely
// about test functions. An unqualified interior-facet test value or gradient is
// not mathematically defined because the test field has two traces. Older
// compatibility lowering assigned row-relative meaning to these expressions;
// current interior-facet lowering rejects them before pullback and requires
// minus(...), plus(...), jump(...), or average(...) trace syntax.
// ---------------------------------------------------------------------------

template<class Expr>
struct has_unqualified_interior_test_trace : std::false_type {};

template<StaticString Name, FieldShape Shape>
struct has_unqualified_interior_test_trace<TestSpace<Name, Shape>>
   : std::true_type {};

template<StaticString Name, FieldShape Shape>
struct has_unqualified_interior_test_trace<
   GradientExpr<TestSpace<Name, Shape>>>
   : std::true_type {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<NegExpr<Expr>>
   : has_unqualified_interior_test_trace<Expr> {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<GradientExpr<Expr>>
   : has_unqualified_interior_test_trace<Expr> {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<MinusTraceExpr<Expr>>
   : std::false_type {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<PlusTraceExpr<Expr>>
   : std::false_type {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<AverageExpr<Expr>>
   : std::false_type {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<JumpExpr<Expr>>
   : std::false_type {};

template<FieldExpr AdvExpr, FieldExpr Expr>
struct has_unqualified_interior_test_trace<UpwindExpr<AdvExpr, Expr>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<AdvExpr>::value ||
        has_unqualified_interior_test_trace<Expr>::value> {};

template<FieldExpr Adv, typename Interior, typename Exterior>
struct has_unqualified_interior_test_trace<BoundaryUpwindExpr<Adv, Interior, Exterior>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<Adv>::value ||
        has_unqualified_interior_test_trace<Interior>::value ||
        has_unqualified_interior_test_trace<Exterior>::value> {};

template<FieldExpr Head, FieldExpr... Tail>
struct has_unqualified_interior_test_trace<SumExpr<Head, Tail...>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<Head>::value ||
        (has_unqualified_interior_test_trace<Tail>::value || ...)> {};

template<FieldExpr LHS, FieldExpr RHS>
struct has_unqualified_interior_test_trace<MultFieldExpr<LHS, RHS>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<LHS>::value ||
        has_unqualified_interior_test_trace<RHS>::value> {};

template<FieldExpr LHS, FieldExpr RHS>
struct has_unqualified_interior_test_trace<ProductExpr<LHS, RHS>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<LHS>::value ||
        has_unqualified_interior_test_trace<RHS>::value> {};

template<FieldExpr LHS, FieldExpr RHS>
struct has_unqualified_interior_test_trace<DotExpr<LHS, RHS>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<LHS>::value ||
        has_unqualified_interior_test_trace<RHS>::value> {};

template<FieldExpr LHS, FieldExpr RHS>
struct has_unqualified_interior_test_trace<InnerExpr<LHS, RHS>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<LHS>::value ||
        has_unqualified_interior_test_trace<RHS>::value> {};

template<FieldExpr LHS, FieldExpr RHS>
struct has_unqualified_interior_test_trace<OuterExpr<LHS, RHS>>
   : std::bool_constant<
        has_unqualified_interior_test_trace<LHS>::value ||
        has_unqualified_interior_test_trace<RHS>::value> {};

template<FieldExpr Expr>
struct has_unqualified_interior_test_trace<TransposeExpr<Expr>>
   : has_unqualified_interior_test_trace<Expr> {};

template<DomainExpr Domain, FieldExpr Expr>
struct has_unqualified_interior_test_trace<Integrand<Domain, Expr>>
   : has_unqualified_interior_test_trace<Expr> {};

template<class Expr>
inline constexpr bool has_unqualified_interior_test_trace_v =
   has_unqualified_interior_test_trace<std::remove_cvref_t<Expr>>::value;

// ---------------------------------------------------------------------------
// Plus-side physical geometry requirements for interior facets.
//
// Two-sided expression evaluation does not necessarily require two-sided
// physical geometry. Value traces need plus-side field data and reference-side
// evaluation, but they do not need the plus cell Jacobian. A plus-side Jacobian
// is required only when an expression is actually evaluated
// in the plus-side context and asks for a physical gradient or inverse Jacobian.
// ---------------------------------------------------------------------------

template<class Input>
struct coefficient_input_requires_plus_side_jacobian : std::false_type {};

template<StaticString Name>
struct coefficient_input_requires_plus_side_jacobian<FieldGradient<Name>>
   : std::true_type {};

template<>
struct coefficient_input_requires_plus_side_jacobian<JacobianInverse>
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
      (coefficient_input_requires_plus_side_jacobian<std::remove_cvref_t<Inputs>>::value || ...);
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<NegExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<MinusTraceExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, false>::value;
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<PlusTraceExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, true>::value;
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
   static constexpr bool value =
      plus_side_jacobian_requirement<MinusTraceExpr<Expr>, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<PlusTraceExpr<Expr>, InPlusSideContext>::value;
};

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<JumpExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<MinusTraceExpr<Expr>, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<PlusTraceExpr<Expr>, InPlusSideContext>::value;
};

template<FieldExpr AdvExpr, FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   UpwindExpr<AdvExpr, Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<AdvExpr, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<Expr, false>::value ||
      plus_side_jacobian_requirement<Expr, true>::value;
};

template<FieldExpr Adv, typename Interior, typename Exterior, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   BoundaryUpwindExpr<Adv, Interior, Exterior>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Adv, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<Interior, InPlusSideContext>::value ||
      plus_side_jacobian_requirement<Exterior, InPlusSideContext>::value;
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

template<FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   TransposeExpr<Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<DomainExpr Domain, FieldExpr Expr, bool InPlusSideContext>
struct plus_side_jacobian_requirement<
   Integrand<Domain, Expr>, InPlusSideContext>
{
   static constexpr bool value =
      plus_side_jacobian_requirement<Expr, InPlusSideContext>::value;
};

template<class Expr>
inline constexpr bool requires_plus_side_jacobian_v =
   plus_side_jacobian_requirement<std::remove_cvref_t<Expr>, false>::value;

template<class... Terms>
struct raw_named_field_requirements<std::tuple<Terms...>>
{
   using type = concat_many_t<typename raw_named_field_requirements<std::remove_cvref_t<Terms>>::type...>;
};
template<class Terms, class Symbols>
struct raw_named_field_requirements<SumFormExpr<Terms, Symbols>> : raw_named_field_requirements<Terms> {};


template<class... Terms>
struct unqualified_side_dependent_named_field_requirements<std::tuple<Terms...>>
{
   using type = concat_many_t<typename unqualified_side_dependent_named_field_requirements<std::remove_cvref_t<Terms>>::type...>;
};
template<class Terms, class Symbols>
struct unqualified_side_dependent_named_field_requirements<SumFormExpr<Terms, Symbols>> : unqualified_side_dependent_named_field_requirements<Terms> {};


template<class... Terms>
struct has_unqualified_interior_test_trace<std::tuple<Terms...>>
   : std::bool_constant<(has_unqualified_interior_test_trace<std::remove_cvref_t<Terms>>::value || ...)> {};
template<class Terms, class Symbols>
struct has_unqualified_interior_test_trace<SumFormExpr<Terms, Symbols>> : has_unqualified_interior_test_trace<Terms> {};


template<class... Terms, bool Plus>
struct plus_side_jacobian_requirement<std::tuple<Terms...>, Plus>
   : std::bool_constant<(plus_side_jacobian_requirement<std::remove_cvref_t<Terms>, Plus>::value || ...)> {};
template<class Terms, class Symbols, bool Plus>
struct plus_side_jacobian_requirement<SumFormExpr<Terms, Symbols>, Plus>
   : plus_side_jacobian_requirement<Terms, Plus> {};

} // namespace gendil
