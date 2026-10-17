// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/**
 * @file
 * @brief Separate an affine weak residual into its linear operator and load.
 *
 * For a residual R(x) = A(x) + c, the equation R(x) = 0 becomes A(x) = -c.
 * lhs(residual) returns the form for A; rhs(residual) returns the form for -c.
 * Here x includes all active trial fields declared in the complete form.
 * "Constant" means independent of those fields: c may still vary in space and
 * depend on prescribed finite-element fields or geometry.
 *
 * @code
 * #include <gendil/gendil.hpp>
 * using namespace gendil;
 *
 * const TrialSpace<"u"> u;
 * const TestSpace<"v"> v;
 * const FiniteElementField<"f"> f;
 * const Cells<"mesh"> cells;
 *
 * auto residual = integrate(cells, (u - f) * v);
 * auto lhs_wf = lhs(residual); // Represents integrate(cells, u * v).
 * auto rhs_wf = rhs(residual); // Represents integrate(cells, f * v).
 * @endcode
 * Both results are ordinary weak forms. With a suitable context and integration
 * rule, the residual-operator factory builds an operator L from lhs_wf and a
 * load b from rhs_wf, applied as L(x, y) and b(y), respectively.
 *
 * Splitting uses the expression structure and complete-form declarations; it
 * needs no execution context and never evaluates a coefficient callback. The
 * original form must be a valid residual with provably affine dependence on
 * its active fields. This is a structural extraction, not differentiation or
 * linearization of a nonlinear residual.
 *
 * The result preserves integrand order, domains, and logical output rows.
 * Vanished integrands retain that identity as typed zeros without retaining
 * their discarded expressions. Retained expression objects and coefficient
 * captures are copied into the result; references held by captures still refer
 * to the original objects and keep their original lifetime requirements.
 */

#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include "gendil/FiniteElementMethod/WeakForm/zero.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/signatures.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/neg.hpp"
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/scale.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumformexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/upwind.hpp"

namespace gendil
{
namespace weak_form_detail
{
enum class AffinePart { Linear, Constant };

// Used only during construction. It is eliminated before rebuilding algebra;
// only a whole vanished integrand becomes a metadata-bearing ZeroExpr.
struct SplitZero {};
template<class T>
inline constexpr bool is_split_zero_v = std::is_same_v<std::remove_cvref_t<T>, SplitZero>;

template<class E> struct affine_node;

// Construct only the requested part. Whole independent or linear subtrees can
// be retained or discarded directly; mixed affine subtrees use the visitors below.
template<AffinePart Part, class Symbols, SemanticEntity Entity, class E>
auto AffineSplit(const E& expr)
{
   using Semantics = expression_semantics_t<E, Symbols, Entity>;
   if constexpr (weak_form_traits::is_zero_expr_v<E> ||
      (Part == AffinePart::Linear && Semantics::active_affinity == ActiveAffinity::Independent) ||
      (Part == AffinePart::Constant && Semantics::active_affinity == ActiveAffinity::Linear))
      return SplitZero{};
   else if constexpr (Semantics::active_affinity == ActiveAffinity::Independent ||
                      Semantics::active_affinity == ActiveAffinity::Linear)
      return expr;
   else
      return affine_node<E>::template split<Part, Symbols, Entity>(expr);
}

// Omit a vanished summand before reconstructing the sum's expression type.
template<class T>
auto NonzeroSplitTuple(T&& result)
{
   if constexpr (is_split_zero_v<T>) return std::tuple<>{};
   else return std::tuple<std::remove_cvref_t<T>>{std::forward<T>(result)};
}

template<FieldExpr Head, FieldExpr... Tail>
struct affine_node<SumExpr<Head, Tail...>>
{
   template<AffinePart Part, class Symbols, SemanticEntity Entity>
   static auto split(const SumExpr<Head, Tail...>& expr)
   {
      return std::apply([](const auto&... term) {
         // Build only the requested branch, once per term in source order.
         auto parts = std::tuple{AffineSplit<Part, Symbols, Entity>(term)...};
         return std::apply([](auto&&... part) {
            auto nonzero = std::tuple_cat(NonzeroSplitTuple(std::forward<decltype(part)>(part))...);
            if constexpr (std::tuple_size_v<decltype(nonzero)> == 0) return SplitZero{};
            else if constexpr (std::tuple_size_v<decltype(nonzero)> == 1) return std::get<0>(std::move(nonzero));
            else return MakeSumExprFromTuple(std::move(nonzero));
         }, std::move(parts));
      }, expr.terms);
   }
};

// Negation, transpose, and supported traces preserve the split of their operand.
template<template<class> class Node, class E>
struct affine_unary
{
   template<AffinePart Part, class Symbols, SemanticEntity Entity>
   static auto split(const Node<E>& expr)
   {
      auto part = AffineSplit<Part, Symbols, Entity>(expr.expr);
      if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
      else return Node<decltype(part)>{part};
   }
};
#define GENDIL_AFFINE_UNARY(Node) \
template<FieldExpr E> struct affine_node<Node<E>> : affine_unary<Node, E> {};
GENDIL_AFFINE_UNARY(NegExpr)
GENDIL_AFFINE_UNARY(TransposeExpr)
GENDIL_AFFINE_UNARY(MinusTraceExpr)
GENDIL_AFFINE_UNARY(PlusTraceExpr)
GENDIL_AFFINE_UNARY(AverageExpr)
GENDIL_AFFINE_UNARY(JumpExpr)
#undef GENDIL_AFFINE_UNARY

// Complete-form admission has established that at most one operand depends on
// active fields. Split that operand while retaining the other in its original order.
template<template<class, class> class Node, class L, class R>
struct affine_binary
{
   template<AffinePart Part, class Symbols, SemanticEntity Entity>
   static auto split(const Node<L, R>& expr)
   {
      if constexpr (expression_semantics_t<L, Symbols, Entity>::active_affinity == ActiveAffinity::Independent)
      {
         auto part = AffineSplit<Part, Symbols, Entity>(expr.rhs);
         if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
         else return Node<L, decltype(part)>{expr.lhs, part};
      }
      else
      {
         auto part = AffineSplit<Part, Symbols, Entity>(expr.lhs);
         if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
         else return Node<decltype(part), R>{part, expr.rhs};
      }
   }
};
#define GENDIL_AFFINE_BINARY(Node) \
template<FieldExpr L, FieldExpr R> struct affine_node<Node<L, R>> : affine_binary<Node, L, R> {};
GENDIL_AFFINE_BINARY(ProductExpr)
GENDIL_AFFINE_BINARY(MultFieldExpr)
GENDIL_AFFINE_BINARY(DotExpr)
GENDIL_AFFINE_BINARY(InnerExpr)
GENDIL_AFFINE_BINARY(OuterExpr)
#undef GENDIL_AFFINE_BINARY

// Admission requires an independent selector, so both parts use the same upwind
// choice. Only the transported expression needs splitting.
template<FieldExpr Selector, FieldExpr E>
struct affine_node<UpwindExpr<Selector, E>>
{
   template<AffinePart Part, class Symbols, SemanticEntity Entity>
   static auto split(const UpwindExpr<Selector, E>& expr)
   {
      auto part = AffineSplit<Part, Symbols, Entity>(expr.expr);
      if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
      else return UpwindExpr<Selector, decltype(part)>{expr.adv_expr, part};
   }
};

template<FieldExpr Selector, class Interior, class Exterior>
struct affine_node<BoundaryUpwindExpr<Selector, Interior, Exterior>>
{
   template<AffinePart Part, class Symbols, SemanticEntity Entity, class E>
   static auto split_state(const E& expr)
   {
      if constexpr (weak_form_traits::is_boundary_upwind_zero_v<E>) return expr;
      else
      {
         auto part = AffineSplit<Part, Symbols, Entity>(expr);
         if constexpr (is_split_zero_v<decltype(part)>)
         {
            if constexpr (has_field_shape_v<E>) return BoundaryUpwindZero<field_shape_v<E>>{};
            else return Empty{};
         }
         else return part;
      }
   }

   template<AffinePart Part, class Symbols, SemanticEntity Entity>
   static auto split(const BoundaryUpwindExpr<Selector, Interior, Exterior>& expr)
   {
      auto interior = split_state<Part, Symbols, Entity>(expr.interior_expr);
      auto exterior = split_state<Part, Symbols, Entity>(expr.exterior_expr);
      if constexpr (weak_form_traits::is_boundary_upwind_zero_v<decltype(interior)> &&
                    weak_form_traits::is_boundary_upwind_zero_v<decltype(exterior)>)
         return SplitZero{};
      else
         return BoundaryUpwindExpr<Selector, decltype(interior), decltype(exterior)>{
            expr.adv_expr, interior, exterior};
   }
};

template<class E> inline constexpr bool is_negated_expr_v = false;
template<FieldExpr E> inline constexpr bool is_negated_expr_v<NegExpr<E>> = true;

// Apply the equation's RHS sign to the complete scalar residual expression.
template<class E>
auto NegateResidualConstant(const E& expr)
{ return MakeNegExpr(expr); }

// Cancel a sign already present in a scalar residual product. Keeping the
// original operand order avoids introducing vector negation or distributing
// signs through sums, traces, or opaque coefficients.
template<template<class, class> class Node, FieldExpr L, FieldExpr R>
   requires (std::is_same_v<Node<L, R>, ProductExpr<L, R>> ||
             std::is_same_v<Node<L, R>, MultFieldExpr<L, R>>)
auto NegateResidualConstant(const Node<L, R>& expr)
{
   if constexpr (is_negated_expr_v<L>)
      return Node<std::remove_cvref_t<decltype(expr.lhs.expr)>, R>{expr.lhs.expr, expr.rhs};
   else if constexpr (is_negated_expr_v<R>)
      return Node<L, std::remove_cvref_t<decltype(expr.rhs.expr)>>{expr.lhs, expr.rhs.expr};
   else
      return MakeNegExpr(expr);
}

// Validate before pruning: discarded terms must not hide invalid semantics or
// nonlinear dependence. Each original integrand retains its domain and output.
template<AffinePart Part, FormExpr Form>
auto SplitResidual(const Form& form)
{
   using Signature = form_signature_t<Form>;
   static_assert(Signature::valid && Signature::result_kind == FormResultKind::Residual,
      "lhs/rhs requires a valid residual weak form.");
   static_assert(!Signature::valid || Signature::active_affinity != ActiveAffinity::Nonlinear,
      "lhs/rhs rejects nonlinear active dependence.");
   static_assert(!Signature::valid || Signature::active_affinity != ActiveAffinity::Unknown,
      "lhs/rhs cannot split unknown active dependence (opaque callbacks or active upwind selectors).");
   if constexpr (Signature::valid && Signature::result_kind == FormResultKind::Residual &&
      Signature::active_affinity != ActiveAffinity::Nonlinear &&
      Signature::active_affinity != ActiveAffinity::Unknown)
   {
      using Symbols = form_symbols_t<Form>;
      auto transformed = TransformIntegrands(form, [](const auto& term) {
         using Term = std::remove_cvref_t<decltype(term)>;
         using TermSignature = integrand_signature_t<Term, Symbols>;
         auto part = AffineSplit<Part, Symbols, TermSignature::entity>(term.expr);
         if constexpr (is_split_zero_v<decltype(part)>)
         {
            // Keep output identity and declarations as types, without dead captures.
            using Zero = ZeroExpr<typename form_symbols_t<Term>::plain_declarations,
               typename TermSignature::logical_output_sides>;
            return integrate(term.domain, Zero{});
         }
         else if constexpr (Part == AffinePart::Constant)
            return integrate(term.domain, NegateResidualConstant(part));
         else
            return integrate(term.domain, part);
      });
      // RHS is a load. Original declarations continue to classify symbols, but
      // its input-free callable contract must survive later concatenation and
      // repeated transformations. LHS retains its source's invocation domain.
      using Relations = std::conditional_t<Part == AffinePart::Linear,
         unique_t<typename Signature::callable_input_relations>, type_list<>>;
      using Original = typename Symbols::plain_declarations;
      using Declarations = weak_form_traits::symbol_summary<Original::supported, typename Original::active,
         typename Original::tests, typename Original::bound, Relations>;
      return SumFormExpr<typename decltype(transformed)::terms_type, Declarations>{std::move(transformed.terms)};
   }
}
} // namespace weak_form_detail

/**
 * @brief Return the input-linear form A from an affine residual R(x) = A(x) + c.
 *
 * Linearity is joint in all active fields declared in the complete form. Products
 * such as u*p are therefore nonlinear when both names are active. Coefficient
 * callbacks reading active fields have unknown dependence and cannot be split,
 * even if their C++ bodies happen to be linear. An upwind selector must also be
 * independent of all active fields. Runtime zero factors or apparent cancellation
 * do not establish affinity. Invalid or unprovably affine residuals fail at
 * compile time before any terms are discarded.
 *
 * The returned form preserves the source's callable input dimensions, including
 * dimensions retained by earlier transformations, and all logical outputs. Actual
 * reads contain only the surviving expressions. For an input-free load, lhs
 * returns an input-free zero form with the same output identity.
 *
 * @see rhs
 */
template<FormExpr Form>
auto lhs(const Form& residual)
{ return weak_form_detail::SplitResidual<weak_form_detail::AffinePart::Linear>(residual); }

/**
 * @brief Return the load -c from an affine residual R(x) = A(x) + c.
 *
 * The minus sign puts the independent contribution on the right-hand side of
 * R(x) = 0. For example, rhs of integrate(cells, (u + f) * v) is the form for
 * -f*v when f is independent of the active trials. The same validity and affinity
 * requirements as lhs apply to the complete original residual.
 *
 * The result is input-free even though the original trial declarations remain
 * available for inspection. It retains all logical outputs and domains; a purely
 * linear source yields a zero load with those outputs. Persistent fields are
 * required only when they occur in the surviving load expressions.
 *
 * Applying rhs to an input-free load negates it. Consequently rhs is not
 * idempotent: rhs(rhs(R)) represents c, rather than -c, for R(x) = A(x) + c.
 *
 * @see lhs
 */
template<FormExpr Form>
auto rhs(const Form& residual)
{ return weak_form_detail::SplitResidual<weak_form_detail::AffinePart::Constant>(residual); }
} // namespace gendil
