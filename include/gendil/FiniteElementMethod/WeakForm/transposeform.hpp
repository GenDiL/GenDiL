// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

/**
 * @file
 * @brief Build a weak form for the real algebraic transpose of a bilinear operator.
 *
 * transpose(a) exchanges the trial and test occurrences in a bilinear form.
 * Its result is an ordinary form: it can be added to compatible forms and passed
 * to MakeGenericOperator with the same named spaces, geometry and quadrature.
 * If the original operator maps U coefficients to V loads, the result maps V
 * coefficients to U loads and satisfies y^T A x = x^T A^T y.
 *
 * @code
 * #include <gendil/gendil.hpp>
 * using namespace gendil;
 *
 * TrialSpace<"u"> u;
 * TestSpace<"v"> v;
 * Cells<"mesh"> cells;
 * auto a = integrate(cells, u * v);
 * auto at = transpose(a); // Reads trial "v" and contributes to test "u".
 * auto twice_at = at + at;
 * @endcode
 * Bind "u" and "v" to their original spaces when constructing either operator.
 * Their names, shapes and DOF placement stay with the fields; their roles change.
 * See the [weak-form guide](README.md#transposing-a-bilinear-form) for a complete
 * rectangular example.
 *
 * @par Construction flow
 * The public transpose function at the end of this file coordinates four steps:
 * 1. Validate the complete original form, including all active-field declarations.
 * 2. Exchange callable input/output metadata, retaining even empty block slots.
 * 3. Isolate each original trial field's contribution and exchange role occurrences.
 * 4. Return the live integrands and metadata-only zeros in a SumFormExpr.
 *
 * For example, (u_trial + p_trial) * v_test becomes two contributions:
 * u_test * v_trial and p_test * v_trial. Each still has one test name, as required
 * by the DSL. Products retain operand order, and traces retain their original
 * side: minus(u_trial) becomes minus(u_test). Existing pullbacks then turn test
 * values and derivatives into coefficients for the normal matrix-free execution.
 *
 * @par Reading the implementation
 * Start with transpose(), then TransposeLiveTerms() and RewriteTranspose(). The
 * file defines their building blocks first, in these groups:
 * - Internal expression nodes: traversal provenance and lazy upwind branches.
 * - Traits: expose those nodes to shape, field and geometry analysis.
 * - Expression rewriting: project one trial field, then exchange roles.
 * - Integrand construction and metadata: preserve callable spaces and zero rows.
 * - Affine extraction: let lhs/rhs also handle the internal nodes.
 * - Public entry point: validate and assemble the resulting form.
 *
 * Type analysis and structural choices happen at compile time. Calling
 * transpose(a) constructs expression objects and copies their stored captures;
 * applying the resulting operator evaluates coefficient values at quadrature
 * points. An input-independent coefficient may therefore still change between
 * applications through its referenced data.
 */

#include "gendil/FiniteElementMethod/WeakForm/affinesplit.hpp"
#include "gendil/FiniteElementMethod/WeakForm/curl.hpp"
#include "gendil/FiniteElementMethod/WeakForm/divergence.hpp"

namespace gendil
{
namespace weak_form_detail
{
// -----------------------------------------------------------------------------
// 1. Internal expression nodes
// -----------------------------------------------------------------------------

/**
 * @brief Carry the source contribution's geometry requirements with its transpose.
 *
 * E is the live rewritten expression. ContextMask is the geometry requirement
 * mask of the projected contribution BEFORE roles were exchanged. Preparation
 * inspects it to reject local interior coefficients whose traversal indices or
 * reference coordinates could change when the output moves to the other cell.
 * The mask records requirements only; it stores no numerical evaluation context.
 *
 * Pullbacks in pullback.hpp simply forward to expr. A second transpose removes
 * this wrapper while rewriting its child again. Dead terms retain neither the
 * wrapper nor the original expression and its captures.
 */
template<FieldExpr E, GeometryRequirement ContextMask>
struct TransposedBilinearExpr : FieldBase
{
   E expr;
   GENDIL_HOST_DEVICE
   explicit TransposedBilinearExpr(const E& e) : expr(e) {}
};

enum class FluxBranch { InteriorMinus, InteriorPlus, BoundaryOutflow, BoundaryInflow };

/**
 * @brief Evaluate one signed normal-flux branch, skipping an unselected state.
 *
 * A is the frozen velocity expression: independent of active fields, but still
 * evaluated on each application. With s = beta.n, s^+ = max(s,0) and
 * s^- = min(s,0), the interior branches supply s^+ times the minus state and s^-
 * times the plus state. E retains the state's trace; the selector stays in its
 * original evaluation scope.
 *
 * This node has two uses. In the transposed form E can contain a test field,
 * which the public upwind evaluator does not accept. Its specialized pullback
 * first lowers E to channels (coefficients multiplying test values/gradients).
 * GateFluxChannels in pullback.hpp then wraps each COMPLETE channel coefficient
 * in another UpwindFluxBranchExpr with a test-free E. The evaluator below handles
 * those coefficients, or a test-free state after a second transpose. Gating the
 * complete coefficient keeps inactive transported-state callbacks unevaluated.
 *
 * Selection matches the public fluxes at zero speed: interior minus selects
 * s >= 0, whereas boundary outflow selects s > 0. Both negative branches select
 * s < 0, so a zero-speed boundary evaluates neither state.
 */
template<FieldExpr A, FieldExpr E, FluxBranch Branch>
struct UpwindFluxBranchExpr : FieldBase
{
   A adv_expr;
   E expr;
   GENDIL_HOST_DEVICE
   UpwindFluxBranchExpr(const A& a, const E& e) : adv_expr(a), expr(e) {}

   template<class K, class W, class O, class F, class Q, class Fields>
   GENDIL_HOST_DEVICE
   auto operator()(const K& k, const W& w, const O& o, const F& f,
                   const Q& q, const Fields& fields) const
   {
      const auto velocity = adv_expr(k, w, o, f, q, fields);
      const auto normal = GetFacetNormal(f, q);
      constexpr auto dimension = static_extent_v<std::remove_cvref_t<decltype(normal)>>;
      static_assert(dimension != std::dynamic_extent &&
                    static_extent_v<decltype(velocity)> == dimension,
         "Frozen flux velocity must match the physical domain dimension.");
      // Preserve each public flux's accumulation type and readable-vector
      // contract; boundary upwind also accepts mixed scalar array types.
      const auto speed = [&] {
         if constexpr (Branch == FluxBranch::InteriorMinus || Branch == FluxBranch::InteriorPlus)
            return Dot(velocity, normal);
         else
         {
            Real result = 0;
            for (Integer d = 0; d < dimension; ++d)
               result += GetValue(velocity, d) * GetValue(normal, d);
            return result;
         }
      }();
      using Value = decltype(expr(k, w, o, f, q, fields));
      using Result = typename decltype(BoundaryUpwindResultType<decltype(speed), Value, Empty>())::type;
      const bool selected = Branch == FluxBranch::InteriorMinus ? speed >= 0 :
         Branch == FluxBranch::BoundaryOutflow ? speed > 0 : speed < 0;
      if (selected)
         return ScaleBoundaryUpwindState<Result>(speed, expr(k, w, o, f, q, fields));
      // Value initialization of RecursiveArray does not initialize its entries.
      Result zero;
      if constexpr (static_num_rows_v<Result> != std::dynamic_extent &&
                    static_num_cols_v<Result> != std::dynamic_extent)
         for (Integer i = 0; i < static_num_rows_v<Result>; ++i)
            for (Integer j = 0; j < static_num_cols_v<Result>; ++j) zero(i,j) = 0;
      else if constexpr (static_extent_v<Result> != std::dynamic_extent)
         for (Integer i = 0; i < static_extent_v<Result>; ++i) zero(i) = 0;
      else zero = 0;
      return zero;
   }
};

template<FieldExpr E, GeometryRequirement M>
std::ostream& operator<<(std::ostream& os, const TransposedBilinearExpr<E, M>& e)
{ return os << e.expr; }

template<FieldExpr A, FieldExpr E, FluxBranch B>
std::ostream& operator<<(std::ostream& os, const UpwindFluxBranchExpr<A, E, B>& e)
{ return os << "flux_branch(" << e.adv_expr << ", " << e.expr << ")"; }

} // namespace weak_form_detail

// -----------------------------------------------------------------------------
// 2. Expose the internal nodes to normal weak-form analysis
// -----------------------------------------------------------------------------
// The provenance wrapper is transparent. A flux branch has its state's shape
// and test dependence, but needs inputs from both its selector and its state.
template<FieldExpr E, GeometryRequirement M>
struct field_shape_impl<weak_form_detail::TransposedBilinearExpr<E, M>> : field_shape_impl<E> {};

template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B>
struct field_shape_impl<weak_form_detail::UpwindFluxBranchExpr<A, E, B>> : field_shape_impl<E> {};

template<FieldExpr E, GeometryRequirement M>
struct test_linearity<weak_form_detail::TransposedBilinearExpr<E, M>> : test_linearity<E> {};

template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B>
struct test_linearity<weak_form_detail::UpwindFluxBranchExpr<A, E, B>> : test_linearity<E> {};

// Direct syntax queries are still used by trace and selected-side geometry IO.
#define GENDIL_TRANSPOSE_FORWARD_REQUIREMENTS(Trait) \
template<FieldExpr E, GeometryRequirement M> \
struct Trait<weak_form_detail::TransposedBilinearExpr<E, M>> : Trait<E> {}; \
template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B> \
struct Trait<weak_form_detail::UpwindFluxBranchExpr<A, E, B>> { \
   using type = concat_t<typename Trait<A>::type, typename Trait<E>::type>; };
GENDIL_TRANSPOSE_FORWARD_REQUIREMENTS(raw_named_field_requirements)
GENDIL_TRANSPOSE_FORWARD_REQUIREMENTS(unqualified_side_dependent_named_field_requirements)
#undef GENDIL_TRANSPOSE_FORWARD_REQUIREMENTS
template<FieldExpr E, GeometryRequirement M>
struct has_unqualified_interior_test_trace<weak_form_detail::TransposedBilinearExpr<E,M>>
   : has_unqualified_interior_test_trace<E> {};
template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B>
struct has_unqualified_interior_test_trace<weak_form_detail::UpwindFluxBranchExpr<A,E,B>>
   : std::bool_constant<has_unqualified_interior_test_trace<A>::value || has_unqualified_interior_test_trace<E>::value> {};
template<FieldExpr E, GeometryRequirement M, bool Side, bool Requested>
struct side_jacobian_requirement<weak_form_detail::TransposedBilinearExpr<E,M>,Side,Requested>
   : side_jacobian_requirement<E,Side,Requested> {};
template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B, bool Side, bool Requested>
struct side_jacobian_requirement<weak_form_detail::UpwindFluxBranchExpr<A,E,B>,Side,Requested>
   : std::bool_constant<side_jacobian_requirement<A,Side,Requested>::value || side_jacobian_requirement<E,Side,Requested>::value> {};

namespace weak_form_traits
{
template<FieldExpr E, GeometryRequirement M>
struct symbols_of<weak_form_detail::TransposedBilinearExpr<E,M>> : symbols_of<E> {};
template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B>
struct symbols_of<weak_form_detail::UpwindFluxBranchExpr<A,E,B>> : merge_symbols<symbols_of<A>,symbols_of<E>> {};
template<FieldExpr E, GeometryRequirement M, class Symbols, SemanticSide Side>
struct analyze<weak_form_detail::TransposedBilinearExpr<E,M>, Symbols, Side> : analyze<E,Symbols,Side>
{ static constexpr auto transpose_context_requirements = M; };
// An independent selector lets the branch inherit its state's dependence on
// active inputs. The selector's normal and field dependencies still reach execution.
template<FieldExpr A, FieldExpr E, weak_form_detail::FluxBranch B, class Symbols, SemanticSide Side>
struct analyze<weak_form_detail::UpwindFluxBranchExpr<A,E,B>, Symbols, Side>
   : merge_information<analyze<A,Symbols,Side>,analyze<E,Symbols,Side>>
{
   using selector = analyze<A,Symbols,Side>;
   using state = analyze<E,Symbols,Side>;
   using base = merge_information<selector,state>;
   using geometry_sides = merge_geometry<typename base::geometry_sides,
      one_side_geometry<Side,GeometryRequirement::Normal>>;
   static constexpr bool supported = base::supported &&
      (selector::possible_shapes & shape_options(SemanticShape::Vector)) != 0u;
   static constexpr bool requires_facet = true;
   static constexpr auto symbolic_shape = state::symbolic_shape;
   static constexpr unsigned possible_shapes = state::possible_shapes;
   static constexpr auto test_linearity = selector::test_linearity == SemanticTestLinearity::TestFree
      ? state::test_linearity : SemanticTestLinearity::Nonlinear;
   static constexpr auto active_affinity = selector::active_affinity == ActiveAffinity::Independent
      ? state::active_affinity : ActiveAffinity::Unknown;
   using shape_obligations = std::conditional_t<selector::symbolic_shape == SemanticShape::Unresolved,
      concat_t<typename base::shape_obligations,type_list<ShapeObligation<ShapeOperation::UpwindSelector,A>>>,
      typename base::shape_obligations>;
};
} // namespace weak_form_traits

namespace weak_form_detail
{
// -----------------------------------------------------------------------------
// 3. Rewrite expressions in two passes: select one trial, then exchange roles
// -----------------------------------------------------------------------------

/**
 * @brief Visit an expression for one original trial name.
 *
 * Swap=false projects onto Name: other trial leaves become SplitZero, and
 * surviving leaves keep their roles. Swap=true exchanges the surviving trial
 * and test occurrences. Overloads distinguish node types, so TrialSpace<"u"> and
 * TestSpace<"u"> remain distinguishable even when their names coincide.
 *
 * The complete source form must already be admitted as bilinear. Under that
 * precondition, removing other trials exactly isolates Name's contribution.
 * Projection may also normalize syntax: boundary upwind becomes internal flux
 * branches in either pass. Runtime coefficient values are never used to prune.
 */
template<StaticString Name, bool Swap, class E>
auto RewriteTranspose(const E& expr);

// Declare the complete overload set before any recursive definitions. Most AST
// nodes live in gendil, so argument-dependent lookup cannot find later overloads
// in weak_form_detail when a nested expression is instantiated.
template<StaticString Name, bool Swap> auto RewriteTranspose(const ScaleExpr& e);
template<StaticString Name, bool Swap> auto RewriteTranspose(const Normal& e);
template<StaticString Name, bool Swap, StaticString Field>
auto RewriteTranspose(const FiniteElementField<Field>& e);
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape,
   class Fn, CoefficientInput... Inputs>
auto RewriteTranspose(const Coefficient<Field,Shape,Fn,Inputs...>& e);
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape>
auto RewriteTranspose(const TrialSpace<Field,Shape>& e);
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape>
auto RewriteTranspose(const TestSpace<Field,Shape>& e);

#define GENDIL_DECLARE_TRANSPOSE_UNARY(Node) \
template<StaticString Name, bool Swap, FieldExpr E> \
auto RewriteTranspose(const Node<E>& e);
GENDIL_DECLARE_TRANSPOSE_UNARY(GradientExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(DivergenceExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(CurlExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(NegExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(TransposeExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(SymExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(SkewExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(NormalProjectionExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(TangentialProjectionExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(MinusTraceExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(PlusTraceExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(AverageExpr)
GENDIL_DECLARE_TRANSPOSE_UNARY(JumpExpr)
#undef GENDIL_DECLARE_TRANSPOSE_UNARY

#define GENDIL_DECLARE_TRANSPOSE_BINARY(Node) \
template<StaticString Name, bool Swap, FieldExpr L, FieldExpr R> \
auto RewriteTranspose(const Node<L,R>& e);
GENDIL_DECLARE_TRANSPOSE_BINARY(ProductExpr)
GENDIL_DECLARE_TRANSPOSE_BINARY(MultFieldExpr)
GENDIL_DECLARE_TRANSPOSE_BINARY(DotExpr)
GENDIL_DECLARE_TRANSPOSE_BINARY(InnerExpr)
GENDIL_DECLARE_TRANSPOSE_BINARY(OuterExpr)
#undef GENDIL_DECLARE_TRANSPOSE_BINARY

template<StaticString Name, bool Swap, FieldExpr H, FieldExpr... T>
auto RewriteTranspose(const SumExpr<H,T...>& e);
template<StaticString Name, bool Swap, FieldExpr A, FieldExpr E>
auto RewriteTranspose(const UpwindExpr<A,E>& e);
template<StaticString Name, bool Swap, FieldExpr A, class I, class O>
auto RewriteTranspose(const BoundaryUpwindExpr<A,I,O>& e);
template<StaticString Name, bool Swap, FieldExpr A, FieldExpr E, FluxBranch B>
auto RewriteTranspose(const UpwindFluxBranchExpr<A,E,B>& e);
template<StaticString Name, bool Swap, FieldExpr E, GeometryRequirement M>
auto RewriteTranspose(const TransposedBilinearExpr<E,M>& e);

// A semantic extension needs an explicit role-swap rule; silently retaining an
// unfamiliar active subtree could turn a formally admitted transpose into A.
template<StaticString Name, bool Swap, class E>
auto RewriteTranspose(const E&)
{
   static_assert(dependent_false_v<E>, "transpose has no role-swap rule for this expression.");
}
// Preserve stored objects, including non-default-constructible callbacks.
// Admission has already ruled out callbacks that read any original active field.
template<StaticString Name, bool Swap>
auto RewriteTranspose(const ScaleExpr& e) { return e; }
template<StaticString Name, bool Swap>
auto RewriteTranspose(const Normal& e) { return e; }
template<StaticString Name, bool Swap, StaticString Field>
auto RewriteTranspose(const FiniteElementField<Field>& e) { return e; }
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape,
   class Fn, CoefficientInput... Inputs>
auto RewriteTranspose(const Coefficient<Field,Shape,Fn,Inputs...>& e) { return e; }
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape>
auto RewriteTranspose(const TrialSpace<Field,Shape>& e)
{
   if constexpr (Name != Field) return SplitZero{};
   else if constexpr (Swap) return TestSpace<Field,Shape>{};
   else return e;
}
template<StaticString Name, bool Swap, StaticString Field, FieldShape Shape>
auto RewriteTranspose(const TestSpace<Field,Shape>& e)
{
   if constexpr (Swap) return TrialSpace<Field,Shape>{};
   else return e;
}
// Keep derivatives and trace selectors around the rewritten operand. A removed
// operand propagates SplitZero; it does not leave a callable dead subtree.
template<StaticString Name, bool Swap, template<class> class Node, class E>
auto RewriteTransposeUnary(const Node<E>& e)
{
   auto part = RewriteTranspose<Name,Swap>(e.expr);
   if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
   else return Node<decltype(part)>{part};
}
#define GENDIL_TRANSPOSE_UNARY(Node) \
template<StaticString Name, bool Swap, FieldExpr E> \
auto RewriteTranspose(const Node<E>& e) \
{ return RewriteTransposeUnary<Name,Swap>(e); }
GENDIL_TRANSPOSE_UNARY(GradientExpr)
GENDIL_TRANSPOSE_UNARY(DivergenceExpr)
GENDIL_TRANSPOSE_UNARY(CurlExpr)
GENDIL_TRANSPOSE_UNARY(NegExpr)
GENDIL_TRANSPOSE_UNARY(TransposeExpr)
GENDIL_TRANSPOSE_UNARY(SymExpr)
GENDIL_TRANSPOSE_UNARY(SkewExpr)
GENDIL_TRANSPOSE_UNARY(NormalProjectionExpr)
GENDIL_TRANSPOSE_UNARY(TangentialProjectionExpr)
GENDIL_TRANSPOSE_UNARY(MinusTraceExpr)
GENDIL_TRANSPOSE_UNARY(PlusTraceExpr)
GENDIL_TRANSPOSE_UNARY(AverageExpr)
GENDIL_TRANSPOSE_UNARY(JumpExpr)
#undef GENDIL_TRANSPOSE_UNARY
// Rebuild products in their original order. In particular, swapping field roles
// must not commute a nonsymmetric matrix coefficient past either operand.
template<StaticString Name, bool Swap, template<class,class> class Node, class L, class R>
auto RewriteTransposeBinary(const Node<L,R>& e)
{
   auto left = RewriteTranspose<Name,Swap>(e.lhs);
   auto right = RewriteTranspose<Name,Swap>(e.rhs);
   if constexpr (is_split_zero_v<decltype(left)> || is_split_zero_v<decltype(right)>) return SplitZero{};
   else return Node<decltype(left),decltype(right)>{left,right};
}
#define GENDIL_TRANSPOSE_BINARY(Node) \
template<StaticString Name, bool Swap, FieldExpr L, FieldExpr R> \
auto RewriteTranspose(const Node<L,R>& e) \
{ return RewriteTransposeBinary<Name,Swap>(e); }
GENDIL_TRANSPOSE_BINARY(ProductExpr)
GENDIL_TRANSPOSE_BINARY(MultFieldExpr)
GENDIL_TRANSPOSE_BINARY(DotExpr)
GENDIL_TRANSPOSE_BINARY(InnerExpr)
GENDIL_TRANSPOSE_BINARY(OuterExpr)
#undef GENDIL_TRANSPOSE_BINARY

// SplitZero is a temporary construction marker from affinesplit.hpp. Remove it
// before choosing the resulting sum type; runtime-valued zeros stay expressions.
template<class... Parts> auto TransposeSum(Parts&&... parts)
{
   auto live = std::tuple_cat(NonzeroSplitTuple(std::forward<Parts>(parts))...);
   if constexpr (std::tuple_size_v<decltype(live)> == 0) return SplitZero{};
   else if constexpr (std::tuple_size_v<decltype(live)> == 1) return std::get<0>(std::move(live));
   else return MakeSumExprFromTuple(std::move(live));
}
template<StaticString Name, bool Swap, FieldExpr H, FieldExpr... T>
auto RewriteTranspose(const SumExpr<H,T...>& e)
{
   return std::apply([](const auto&... term) {
      return TransposeSum(RewriteTranspose<Name,Swap>(term)...);
   },e.terms);
}
// A swapped transported state may contain a test. Give each original state side
// its own internal branch, keeping the velocity outside those trace selectors.
template<StaticString Name, bool Swap, FieldExpr A, FieldExpr E>
auto RewriteTranspose(const UpwindExpr<A,E>& e)
{
   auto part = RewriteTranspose<Name,Swap>(e.expr);
   if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
   else if constexpr (!Swap) return UpwindExpr<A,decltype(part)>{e.adv_expr,part};
   else return UpwindFluxBranchExpr<A,decltype(minus(part)),FluxBranch::InteriorMinus>{e.adv_expr,minus(part)} +
      UpwindFluxBranchExpr<A,decltype(plus(part)),FluxBranch::InteriorPlus>{e.adv_expr,plus(part)};
}
// Normalize boundary states in both passes. Omitted lhs/rhs states and projected
// zeros produce no branch, so prescribed inflow discarded by lhs stays discarded.
template<StaticString Name, bool Swap, FluxBranch B, FieldExpr A, class E>
auto RewriteBoundaryUpwindBranch(const A& a, const E& e)
{
   if constexpr (weak_form_traits::is_boundary_upwind_zero_v<E>) return SplitZero{};
   else
   {
      auto part = RewriteTranspose<Name,Swap>(e);
      if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
      else return UpwindFluxBranchExpr<A,decltype(part),B>{a,part};
   }
}
template<StaticString Name, bool Swap, FieldExpr A, class I, class O>
auto RewriteTranspose(const BoundaryUpwindExpr<A,I,O>& e)
{
   return TransposeSum(
      RewriteBoundaryUpwindBranch<Name,Swap,FluxBranch::BoundaryOutflow>(e.adv_expr,e.interior_expr),
      RewriteBoundaryUpwindBranch<Name,Swap,FluxBranch::BoundaryInflow>(e.adv_expr,e.exterior_expr));
}
// Existing branches survive later transformations, including double transpose.
template<StaticString Name, bool Swap, FieldExpr A, FieldExpr E, FluxBranch B>
auto RewriteTranspose(const UpwindFluxBranchExpr<A,E,B>& e)
{
   auto part = RewriteTranspose<Name,Swap>(e.expr);
   if constexpr (is_split_zero_v<decltype(part)>) return SplitZero{};
   else return UpwindFluxBranchExpr<A,decltype(part),B>{e.adv_expr,part};
}
// Projection keeps provenance; exchanging roles again removes it. Splitting and
// flux normalization can remain, so the restored action need not have the original
// C++ expression type.
template<StaticString Name, bool Swap, FieldExpr E, GeometryRequirement M>
auto RewriteTranspose(const TransposedBilinearExpr<E,M>& e)
{
   auto part = RewriteTranspose<Name,Swap>(e.expr);
   if constexpr (is_split_zero_v<decltype(part)> || Swap) return part;
   else return TransposedBilinearExpr<decltype(part),M>{part};
}
template<class E> inline constexpr bool is_transposed_bilinear_v = false;
template<FieldExpr E, GeometryRequirement M>
inline constexpr bool is_transposed_bilinear_v<TransposedBilinearExpr<E,M>> = true;

// -----------------------------------------------------------------------------
// 4. Construct integrands and exchange their callable input/output contracts
// -----------------------------------------------------------------------------

// A requirement identifies a named read or output and its value/gradient mask.
// A declaration supplies the field's shape. Recover shapes from the ORIGINAL
// declarations when converting old outputs to new trials and old inputs to tests.
template<class Fields, class Declarations> struct TransposeDeclarations;
template<class... Fields, class Declarations>
struct TransposeDeclarations<type_list<Fields...>,Declarations>
{ using type = type_list<find_named_field_requirement_t<Declarations,Fields::name>...>; };

// Change role provenance while keeping the field name, derivative mask and side.
// These are type-only requirements; numerical data remain bound by the same name.
template<class List, NamedFieldProvenance Role> struct TransposeRequirements;
template<class... R, NamedFieldProvenance Role>
struct TransposeRequirements<type_list<R...>,Role>
{ using type = type_list<NamedFieldRequirement<R::name,R::mask,Role>...>; };
template<class Sides, NamedFieldProvenance Role>
using TransposeSides = weak_form_traits::side_requirements<
   typename TransposeRequirements<typename Sides::template on<SemanticSide::Unqualified>,Role>::type,
   typename TransposeRequirements<typename Sides::template on<SemanticSide::Minus>,Role>::type,
   typename TransposeRequirements<typename Sides::template on<SemanticSide::Plus>,Role>::type>;

// The read map supports one data source per name. A previously prescribed read
// cannot share its name with a new active input; include Name in the instantiation
// so the compiler identifies the collision without rewriting callback input tags.
template<class Name, class Outputs> consteval void CheckTransposeBoundName()
{
   static_assert(!contains_named_field_requirement_v<Outputs,Name::name>,
      "transpose would rebind a prescribed field as an active trial.");
}

/**
 * @brief Produce the live transposed contributions of one source integrand.
 *
 * Fields lists the complete form's callable inputs. The original signature says
 * which of them this term actually reads; absent fields and typed zeros produce
 * no live integrand here. Each surviving field is projected before role exchange,
 * yielding one test name per output integrand even for a coupled source term.
 *
 * Analyze the projected expression against the ORIGINAL Symbols to retain its
 * coefficient context requirements after pruning. The returned tuple owns its
 * expressions and is flattened with the other terms by transpose().
 */
template<class Symbols, class Term, class... Fields>
auto TransposeLiveTerms(const Term& term, type_list<Fields...>)
{
   using S = integrand_signature_t<Term,Symbols>;
   return std::tuple_cat([&] {
      if constexpr (S::is_zero || !contains_named_field_requirement_v<typename S::active_inputs,Fields::name>)
         return std::tuple<>{};
      else
      {
         auto projected = RewriteTranspose<Fields::name,false>(term.expr);
         using P = decltype(projected);
         auto swapped = RewriteTranspose<Fields::name,true>(projected);
         if constexpr (is_transposed_bilinear_v<P>)
            return std::tuple{integrate(term.domain,swapped)};
         else
         {
            constexpr auto mask = expression_semantics_t<P,Symbols,S::entity>::geometry;
            return std::tuple{integrate(term.domain,TransposedBilinearExpr<decltype(swapped),mask>{swapped})};
         }
      }
   }()...);
}

// Select one named field while retaining whichever trace sides requested it.
template<class List, StaticString Name> struct SelectTransposeName;
template<class... R, StaticString Name>
struct SelectTransposeName<type_list<R...>,Name>
{ using type = concat_many_t<std::conditional_t<R::name == Name,type_list<R>,type_list<>>...>; };

/**
 * @brief Preserve one source callable relation as named logical zero rows.
 *
 * Relation associates a domain with its input fields and trace sides. Its old
 * inputs must remain output slots after transposition, even if they have no live
 * expression. Emit one ZeroExpr integrand per field to preserve that contract.
 * transpose() does this for every distinct callable relation, alongside live
 * blocks as well as wholly vanished blocks. These zeros retain only metadata:
 * they never schedule a kernel or acquire coefficient/input data.
 *
 * SplitZero above discards a subtree during construction; ZeroExpr here survives
 * in the returned form to preserve its callable dimensions. Declarations alone
 * cannot supply those dimensions: rhs retains declarations but is input-free.
 */
template<class Declarations, class Relation, class... Fields>
auto TransposeZeroRows(type_list<Fields...>)
{
   using Sides = typename Relation::value_type;
   return std::tuple{[] {
      using Selected = weak_form_traits::side_requirements<
         typename SelectTransposeName<typename Sides::template on<SemanticSide::Unqualified>,Fields::name>::type,
         typename SelectTransposeName<typename Sides::template on<SemanticSide::Minus>,Fields::name>::type,
         typename SelectTransposeName<typename Sides::template on<SemanticSide::Plus>,Fields::name>::type>;
      using Zero = ZeroExpr<Declarations,TransposeSides<Selected,NamedFieldProvenance::ActiveTest>>;
      return integrate(typename Relation::key_type{},Zero{});
   }()...};
}

// -----------------------------------------------------------------------------
// 5. Keep the internal nodes usable by affine extraction (lhs/rhs)
// -----------------------------------------------------------------------------
// Delegate to the child/state while retaining the independent selector. If the
// requested part vanishes, propagate SplitZero and drop its captures/provenance.
template<AffinePart Part, class Symbols, SemanticEntity Entity, FieldExpr E, GeometryRequirement M>
auto SplitAffineNode(const TransposedBilinearExpr<E,M>& expr)
{
   auto part = weak_form_detail::AffineSplit<Part,Symbols,Entity>(expr.expr);
   if constexpr (is_split_zero_v<decltype(part)>) return part;
   else return TransposedBilinearExpr<decltype(part),M>{part};
}
template<AffinePart Part, class Symbols, SemanticEntity Entity, FieldExpr A, FieldExpr E, FluxBranch B>
auto SplitAffineNode(const UpwindFluxBranchExpr<A,E,B>& expr)
{
   auto part = weak_form_detail::AffineSplit<Part,Symbols,Entity>(expr.expr);
   if constexpr (is_split_zero_v<decltype(part)>) return part;
   else return UpwindFluxBranchExpr<A,decltype(part),B>{expr.adv_expr,part};
}
} // namespace weak_form_detail

// -----------------------------------------------------------------------------
// 6. Public construction: complete-form admission, metadata, then live terms
// -----------------------------------------------------------------------------

/**
 * @brief Return the real algebraic transpose of a bilinear weak form.
 *
 * Constructing A from a and AT from transpose(a), with the same context and rules,
 * exchanges their input/output spaces and satisfies y^T A x = x^T AT y. The pairing
 * is between coefficient vectors and dual loads; it uses no mass solve or
 * integration by parts. Restrictions, offsets and geometry keep their usual roles.
 * The result can be added to compatible forms. Use transpose(lhs(R)) for affine R.
 *
 * @par Admission and bindings
 * Require homogeneous joint linearity in all original active trials and linearity
 * in the test. Validate before projection: u*p is jointly nonlinear when both
 * names are active, and opaque callbacks reading active fields have unknown
 * dependence. Nonzero loads and unsplit affine residuals are rejected. A typed
 * zero with known callable spaces retains their exchanged contract; an input-free
 * rhs cannot acquire an input from its retained trial declarations.
 *
 * Callbacks are copied without evaluation and keep their borrowed-data lifetimes.
 * Prescribed reads keep their data source; a name that would also become an
 * active input is diagnosed. A read-only callback may see updated referenced data
 * on later applications, so compare both directions with the same coefficient data.
 *
 * @par Facet preparation
 * Local interior traversal rejects live transposed terms using ElementIndex,
 * QuadraturePointIndex or ReferenceCoordinates, including traced uses. Moving
 * the output row can change those inputs. Physical coordinates and explicitly
 * traced prescribed fields remain supported. See the weak-form guide for the
 * complete facet contract and pullback.hpp for the numerical channel lowering.
 */
template<FormExpr Form>
auto transpose(const Form& form)
{
   // 1. Establish eligibility from the whole source before discarding any term.
   using S = form_signature_t<Form>;
   static_assert(S::valid && S::result_kind == FormResultKind::Residual,
      "transpose requires a valid bilinear weak form.");
   static_assert(!is_empty_type_list_v<typename S::callable_inputs>,
      "transpose requires a callable input space.");
   static_assert(S::active_affinity != ActiveAffinity::Nonlinear,
      "transpose rejects nonlinear active dependence.");
   static_assert(S::active_affinity != ActiveAffinity::Unknown,
      "transpose cannot prove linearity of active-dependent callbacks or selectors.");
   static_assert(S::is_zero || (S::active_affinity != ActiveAffinity::Independent &&
                               S::active_affinity != ActiveAffinity::Affine),
      "transpose rejects an affine residual or nonzero load; use lhs first.");
   if constexpr (S::valid && S::result_kind == FormResultKind::Residual &&
      !is_empty_type_list_v<typename S::callable_inputs> &&
      (S::is_zero || S::active_affinity == ActiveAffinity::Linear))
   {
      using namespace weak_form_detail;
      using Symbols = form_symbols_t<Form>;
      // 2. Check prescribed-name collisions, then exchange callable roles.
      [&]<class... B>(type_list<B...>) {
         (CheckTransposeBoundName<NameTag<B::name>,typename S::logical_outputs>(),...);
      }(typename S::bound_inputs{});
      using Active = typename TransposeDeclarations<typename S::logical_outputs,typename Symbols::test_declarations>::type;
      using Tests = typename TransposeDeclarations<typename S::callable_inputs,typename Symbols::active_declarations>::type;
      using Bound = named_field_requirement_names_t<typename S::bound_inputs>;
      // Old logical outputs become input slots, including dimensions with no reads.
      using Relations = decltype([]<class... T>(type_list<T...>) {
         return unique_t<type_list<Entry<typename T::domain_type,
            TransposeSides<typename T::logical_output_sides,NamedFieldProvenance::ActiveTrial>>...>>{};
      }(typename S::integrands{}));
      using Declarations = weak_form_traits::symbol_summary<true,Active,Tests,Bound,Relations>;
      // 3. Build executable contributions one original trial field at a time.
      auto live = std::apply([&](const auto&... term) {
         return std::tuple_cat(TransposeLiveTerms<Symbols>(term,typename S::callable_inputs{})...);
      },AsFormTerms(form));
      // 4. Keep old callable inputs as output slots, then own the complete form.
      auto zeros = []<class... R>(type_list<R...>) {
         return std::tuple_cat(TransposeZeroRows<Declarations,R>(typename R::value_type::all{})...);
      }(unique_t<typename S::callable_input_relations>{});
      auto terms = std::tuple_cat(std::move(live),std::move(zeros));
      return SumFormExpr<decltype(terms),Declarations>{std::move(terms)};
   }
}
} // namespace gendil
