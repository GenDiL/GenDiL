// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dot.hpp"
#include "gendil/FiniteElementMethod/WeakForm/inner.hpp"
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/transpose.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/outer.hpp"
#include "gendil/FiniteElementMethod/WeakForm/neg.hpp"
#include "gendil/FiniteElementMethod/WeakForm/normal.hpp"
#include "gendil/FiniteElementMethod/WeakForm/scale.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/jump.hpp"
#include "gendil/FiniteElementMethod/WeakForm/average.hpp"

namespace gendil
{

// =============================================================================
// Design NOTE: Seed-Based Pullback with Compile-Time Channels
// =============================================================================
//
// This header implements the pullback mechanism for weak-form expressions.
//
// **Pullback API:**
//    pullback(expr, seed) → PullbackResult<Channels...>
//
// Where:
//  - expr is a test-linear weak-form expression
//  - seed is the upstream coefficient/cotangent multiplying expr
//  - Result is a compile-time set of channel contributions
//
// **For a complete scalar integrand:**
//    lower_integrand(expr) = pullback(expr, ScalarIdentity{})
//
// **Coordinate frame convention:**
//    GradientChannel stores physical-gradient coefficients produced by
//    weak-form pullback. Mapping transpose from physical to reference
//    coordinates is not applied in this skeleton. It will be handled later
//    by the channel-application stage that writes GradientChannel
//    contributions into the quadrature data consumed by
//    ApplyGradientTestFunctions.
//
// =============================================================================

// =============================================================================
// Channel Application Design
// =============================================================================

// =============================================================================
// Channel Tags
// =============================================================================

/**
 * @brief Value channel: contributions multiplying test function values.
 *
 * For scalar test space TestSpace<Name>: seed is Scalar
 * For vector test space VectorTestSpace<Name>: seed is Vector
 */
struct ValueChannel {};

/**
 * @brief Gradient channel: contributions multiplying test function gradients.
 *
 * For scalar test space gradient grad(TestSpace<Name>): seed is Vector
 * For vector test space gradient grad(VectorTestSpace<Name>): seed is Matrix
 *
 * COORDINATE FRAME: Stores physical-gradient coefficients.
 * Mapping transpose to reference frame is deferred to channel-application stage.
 */
struct GradientChannel {};

/**
 * @brief Canonical global-interior value channel for the minus test row.
 */
struct ValueMinusChannel {};

/**
 * @brief Canonical global-interior value channel for the plus test row.
 */
struct ValuePlusChannel {};

/**
 * @brief Canonical global-interior gradient channel for the minus test row.
 */
struct GradientMinusChannel {};

/**
 * @brief Canonical global-interior gradient channel for the plus test row.
 */
struct GradientPlusChannel {};

template<class Channels>
inline constexpr bool contains_ordinary_test_channel_v =
   std::remove_cvref_t<Channels>::template contains<ValueChannel>() ||
   std::remove_cvref_t<Channels>::template contains<GradientChannel>();

template<class Channels>
inline constexpr bool contains_side_qualified_interior_test_channel_v =
   std::remove_cvref_t<Channels>::template contains<ValueMinusChannel>() ||
   std::remove_cvref_t<Channels>::template contains<ValuePlusChannel>() ||
   std::remove_cvref_t<Channels>::template contains<GradientMinusChannel>() ||
   std::remove_cvref_t<Channels>::template contains<GradientPlusChannel>();

// =============================================================================
// PullbackResult: StaticMap-Based Channel Storage
// =============================================================================

/**
 * @brief Compile-time map from channel tags to seed expressions.
 *
 * PullbackResult is a thin wrapper around StaticMap that stores channel
 * contributions with compile-time presence/absence guarantees.
 *
 * Usage:
 *    auto result = pullback(v, c);
 *    static_assert(decltype(result)::contains<ValueChannel>());
 *    auto& seed = result.template get<ValueChannel>();
 *
 * @tparam Entries Pack of Entry<ChannelTag, SeedExpr> types
 */
template<class... Entries>
struct PullbackResult : StaticMap<Entries...>
{
   using StaticMap<Entries...>::StaticMap;

   // Constructor from StaticMap (for make_map compatibility)
   GENDIL_HOST_DEVICE
   constexpr PullbackResult(StaticMap<Entries...>&& map)
      : StaticMap<Entries...>(std::move(map))
   {}

   GENDIL_HOST_DEVICE
   constexpr PullbackResult(const StaticMap<Entries...>& map)
      : StaticMap<Entries...>(map)
   {}
};

// =============================================================================
// Seed Shape Validation Helper
// =============================================================================

/**
 * @brief SFINAE-safe validation that seed shape matches expression shape.
 *
 * pullback(expr, seed) requires:
 *    field_shape_v<Seed> == field_shape_v<Expr>
 *
 * This two-level pattern avoids hard errors on types without has_field_shape_v.
 */
template<class Expr, class Seed, bool BothHaveShape =
   has_field_shape_v<Expr> && has_field_shape_v<Seed>>
struct is_seed_shape_compatible_impl : std::false_type {};

template<class Expr, class Seed>
struct is_seed_shape_compatible_impl<Expr, Seed, true>
   : std::bool_constant<
        field_shape_v<std::remove_cvref_t<Expr>> ==
        field_shape_v<std::remove_cvref_t<Seed>>>
{};

template<class Expr, class Seed>
inline constexpr bool is_seed_shape_compatible_v =
   is_seed_shape_compatible_impl<Expr, Seed>::value;

// =============================================================================
// Primitive Pullback Rules
// =============================================================================

/**
 * @brief Pullback for test space value (scalar or vector).
 *
 * pullback(v, seed) → ValueChannel with seed
 *
 * For scalar test space (Shape=Scalar): seed must be Scalar
 * For vector test space (Shape=Vector): seed must be Vector
 *
 * @tparam Name Test space name
 * @tparam Shape Field shape (Scalar or Vector)
 * @tparam Seed Seed expression type (must match Shape)
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const TestSpace<Name, Shape>& /*v*/, const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   return PullbackResult{make_map(make_entry<ValueChannel>(seed))};
}

/**
 * @brief Pullback for test space gradient (scalar or vector).
 *
 * pullback(grad(v), seed) → GradientChannel with seed (physical frame)
 *
 * For scalar test space gradient (grad(TestSpace<Scalar>)): seed must be Vector
 * For vector test space gradient (grad(TestSpace<Vector>)): seed must be Matrix
 *
 * @tparam Name Test space name
 * @tparam Shape Field shape (Scalar or Vector)
 * @tparam Seed Seed expression type (must match grad(Shape))
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const GradientExpr<TestSpace<Name, Shape>>& /*grad_v*/, const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   return PullbackResult{make_map(make_entry<GradientChannel>(seed))};
}

template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const MinusTraceExpr<E>&, const Seed&)
{
   static_assert(
      dependent_false_v<E, Seed>,
      "minus(expr) is a canonical interior-facet trace operator. The local/"
      "current-row and boundary pullback paths do not assign row-relative "
      "semantics to public minus(...); use jump/average in the local path or "
      "the canonical global-interior lowering path.");
   return PullbackResult<>{};
}

template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const PlusTraceExpr<E>&, const Seed&)
{
   static_assert(
      dependent_false_v<E, Seed>,
      "plus(expr) is a canonical interior-facet trace operator. The local/"
      "current-row and boundary pullback paths do not assign row-relative "
      "semantics to public plus(...); use jump/average in the local path or "
      "the canonical global-interior lowering path.");
   return PullbackResult<>{};
}

// =============================================================================
// Placeholder Overloads for Deferred Expression Types
// =============================================================================
//
// These overloads produce clear compile-time errors when pullback is called
// on expression types not yet implemented. The static_assert is dependent
// and fires only when pullback is actually invoked.

// Forward declarations needed for placeholder overloads
// (Already declared in dslbase.hpp with FieldExpr constraints)

/**
 * @brief Pullback for ProductExpr (algebraic product).
 *
 * Currently implemented for ProductKind::ScalarTimes and the two linear
 * ProductKind::MatVec adjoints.
 *
 * ScalarTimes adjoint rule:
 *   pullback(scalar_free * test_expr, seed)
 *      = pullback(test_expr, seed * scalar_free)
 *
 * Valid combinations:
 *   - test-linear × test-free scalar
 *   - test-free scalar × test-linear
 *
 * Invalid combinations (will fail with clear static_assert):
 *   - Both test-free → no test contribution
 *   - Both test-linear → NonlinearInTest
 *   - ProductKind::MatVec → supported when exactly one operand is test-linear
 *   - ProductKind::MatMat → requires product-specific adjoint (not implemented)
 *
 * @tparam LHS Left operand type
 * @tparam RHS Right operand type
 * @tparam Seed Seed expression type (must match ProductExpr shape)
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const ProductExpr<LHS, RHS>& expr, const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = ProductExpr<LHS, RHS>;

   // Validate seed shape
   static_assert(is_seed_shape_compatible_v<Expr, S>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   // Branch on ProductKind
   if constexpr (Expr::product_kind == ProductKind::ScalarTimes)
   {
      // ScalarTimes: at least one operand must be Scalar
      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         // LHS is test-linear, RHS is test-free
         static_assert(field_shape_v<R> == FieldShape::Scalar,
            "ScalarTimes ProductExpr pullback requires the test-free multiplier to be Scalar.");

         return pullback(expr.lhs, seed * expr.rhs);
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         // LHS is test-free, RHS is test-linear
         static_assert(field_shape_v<L> == FieldShape::Scalar,
            "ScalarTimes ProductExpr pullback requires the test-free multiplier to be Scalar.");

         return pullback(expr.rhs, seed * expr.lhs);
      }
      else
      {
         static_assert(dependent_false_v<L, R, S>,
            "ScalarTimes ProductExpr pullback requires exactly one TestLinear operand and one TestFree scalar operand. "
            "Both test-free → no test contribution; both test-linear → NonlinearInTest.");
      }
   }
   else if constexpr (Expr::product_kind == ProductKind::MatVec)
   {
      static_assert(field_shape_v<L> == FieldShape::Matrix,
         "MatVec ProductExpr pullback requires the left operand to be Matrix-shaped.");
      static_assert(field_shape_v<R> == FieldShape::Vector,
         "MatVec ProductExpr pullback requires the right operand to be Vector-shaped.");
      static_assert(field_shape_v<S> == FieldShape::Vector,
         "MatVec ProductExpr pullback requires a Vector-shaped seed.");
      static_assert(is_test_free_v<S>,
         "MatVec ProductExpr pullback requires a test-free seed.");

      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         return pullback(
            expr.lhs,
            outer(seed, expr.rhs));
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         return pullback(
            expr.rhs,
            transpose(expr.lhs) * seed);
      }
      else
      {
         static_assert(dependent_false_v<L, R, S>,
            "MatVec ProductExpr pullback requires exactly one TestLinear operand. "
            "Supported directions are test-linear matrix * test-free vector "
            "and test-free matrix * test-linear vector.");
      }
   }
   else
   {
      static_assert(dependent_false_v<L, R, S>,
         "ProductExpr pullback is implemented for ProductKind::ScalarTimes and "
         "the linear ProductKind::MatVec adjoints. "
         "MatMat and broader product-specific adjoints are not implemented.");
   }
}

/**
 * @brief Pullback for DotExpr (vector dot product).
 *
 * pullback(dot(a, b), seed) where exactly one operand is test-linear.
 *
 * Bilinear adjoint rule:
 *   pullback(dot(test_expr, free_expr), seed)
 *      = pullback(test_expr, seed * free_expr)
 *
 * Valid combinations:
 *   - dot(test-linear vector, test-free vector)
 *   - dot(test-free vector, test-linear vector)
 *
 * Invalid combinations (will fail with clear static_assert):
 *   - dot(test-free, test-free) → no test contribution
 *   - dot(test-linear, test-linear) → NonlinearInTest
 *
 * @tparam LHS Left operand type
 * @tparam RHS Right operand type
 * @tparam Seed Seed expression type (must be Scalar)
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const DotExpr<LHS, RHS>& expr, const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = DotExpr<LHS, RHS>;

   // Validate seed shape (DotExpr is Scalar, so seed must be Scalar)
   static_assert(is_seed_shape_compatible_v<Expr, S>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   // Branch on test-linearity
   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      // LHS is test-linear, RHS is test-free
      return pullback(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      // LHS is test-free, RHS is test-linear
      return pullback(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(dependent_false_v<L, R, S>,
         "DotExpr pullback requires exactly one test-linear operand and one test-free operand. "
         "Both test-free → no test contribution; both test-linear → NonlinearInTest.");
   }
}

/**
 * @brief Pullback for NegExpr.
 */
template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const NegExpr<Expr>& expr, const Seed& seed)
{
   return pullback(expr.expr, -seed);
}

// =============================================================================
// Channel Merge Utilities
// =============================================================================

/**
 * @brief Helper to merge one entry from b into the accumulating map.
 *
 * Base case: no more entries to process.
 */
template<class...EntriesA>
GENDIL_HOST_DEVICE
constexpr auto merge_entries_impl(StaticMap<EntriesA...> acc_map)
{
   return acc_map;
}

/**
 * @brief Helper to merge one entry from b into the accumulating map.
 *
 * Recursive case: process first entry, then recurse on rest.
 */
template<class... EntriesA, class FirstEntry, class... RestEntries>
GENDIL_HOST_DEVICE
constexpr auto merge_entries_impl(
   StaticMap<EntriesA...> acc_map,
   const FirstEntry& first_entry,
   const RestEntries&... rest_entries)
{
   using Key = typename FirstEntry::key_type;
   auto new_map = map_insert_or_merge<Key>(
      std::move(acc_map),
      first_entry.value,
      [](const auto& old_val, const auto& new_val)
      {
         return old_val + new_val;
      });

   return merge_entries_impl(std::move(new_map), rest_entries...);
}

/**
 * @brief Merge two PullbackResult maps by combining duplicate channels.
 *
 * If a channel exists in only one result: keep it.
 * If the same channel exists in both results: combine values with operator+.
 *
 * This preserves the test-free invariant and FieldShape compatibility.
 *
 * @tparam EntriesA Entries from first PullbackResult
 * @tparam EntriesB Entries from second PullbackResult
 */
template<class... EntriesA, class... EntriesB>
GENDIL_HOST_DEVICE
constexpr auto merge_two_pullback_results(
   const PullbackResult<EntriesA...>& a,
   const PullbackResult<EntriesB...>& b)
{
   // Unpack entries from b and recursively insert/merge into a's map
   auto result_map = std::apply(
      [&](const auto&... b_entries)
      {
         return merge_entries_impl(StaticMap{a.entries}, b_entries...);
      },
      b.entries);

   return PullbackResult{std::move(result_map)};
}

/**
 * @brief Variadic merge of multiple PullbackResult maps.
 *
 * Base case: single result.
 */
template<class Result>
GENDIL_HOST_DEVICE
constexpr auto merge_pullback_results(Result&& result)
{
   return std::forward<Result>(result);
}

/**
 * @brief Variadic merge: recursively merge pairwise.
 */
template<class ResultA, class ResultB, class... Rest>
GENDIL_HOST_DEVICE
constexpr auto merge_pullback_results(ResultA&& a, ResultB&& b, Rest&&... rest)
{
   auto merged = merge_two_pullback_results(
      std::forward<ResultA>(a),
      std::forward<ResultB>(b));
   return merge_pullback_results(std::move(merged), std::forward<Rest>(rest)...);
}

// =============================================================================
// SumExpr Pullback
// =============================================================================

/**
 * @brief Pullback for SumExpr (sum of field expressions).
 *
 * pullback(t1 + t2 + ..., seed) = merge_channels(
 *    pullback(t1, seed),
 *    pullback(t2, seed),
 *    ...
 * )
 *
 * The same seed is propagated to all terms.
 * Channel maps from each term are merged, combining duplicate channels with operator+.
 *
 * Examples:
 *   pullback(u*v + dot(grad(u), grad(v)), c)
 *      → ValueChannel: c*u
 *      → GradientChannel: c*grad(u)
 *
 *   pullback(u*v + mu*v, c)
 *      → ValueChannel: (c*u) + (c*mu)
 *
 * @tparam Head First term type
 * @tparam Tail Additional term types
 * @tparam Seed Seed expression type (must match SumExpr shape)
 */
template<FieldExpr Head, FieldExpr... Tail, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const SumExpr<Head, Tail...>& expr, const Seed& seed)
{
   using Expr = SumExpr<Head, Tail...>;
   using S = std::remove_cvref_t<Seed>;

   // Validate seed shape (all SumExpr terms have the same FieldShape)
   static_assert(is_seed_shape_compatible_v<Expr, S>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   // Pullback each term with the same seed, then merge all channel maps
   return std::apply(
      [&](const auto&... terms)
      {
         return merge_pullback_results(pullback(terms, seed)...);
      },
      expr.terms);
}

// =============================================================================
// InnerExpr Pullback
// =============================================================================
//
// InnerExpr represents Frobenius inner product for pairing operations.
//
// Pullback rule (narrow scope - exactly one test-linear operand):
//    pullback(inner(coeff_expr, test_expr), seed)
//       = pullback(test_expr, seed * coeff_expr)
//
// Valid patterns:
//    inner(A, outer(jump(V), normal)) - coeff matrix, test outer product
//    inner(outer(jump(U), normal), average(grad(V))) - test trial, test gradient
//
// NARROW SCOPE:
//    - Only handles exactly one test-linear operand
//    - Rejects both-test-dependent cases (would be nonlinear/quadratic)
//    - Uses ProductExpr (seed * coeff_expr) for seed multiplication
//

/**
 * @brief InnerExpr pullback: coeff-test pattern.
 *
 * For inner(coeff_expr, test_expr) where RHS is test-linear:
 *    Variation: coeff_expr : delta_test_expr
 *    Pullback seed for test_expr: seed * coeff_expr
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_free_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Frobenius inner: A : B where B is test-linear
   // Pullback seed for B: seed * A
   auto rhs_seed = seed * expr.lhs;
   return pullback(expr.rhs, rhs_seed);
}

/**
 * @brief InnerExpr pullback: test-coeff pattern (symmetric case).
 *
 * For inner(test_expr, coeff_expr) where LHS is test-linear:
 *    Variation: delta_test_expr : coeff_expr
 *    Pullback seed for test_expr: seed * coeff_expr
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_free_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Symmetric case: inner(test_expr, coeff_expr)
   // Pullback seed for test_expr: seed * coeff_expr
   auto lhs_seed = seed * expr.rhs;
   return pullback(expr.lhs, lhs_seed);
}

/**
 * @brief InnerExpr pullback: reject both-test-dependent case.
 *
 * If both operands are test-dependent, the expression is quadratic in test
 * function (nonlinear). Such expressions are invalid for weak forms.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "InnerExpr pullback: both operands test-dependent not supported. "
      "inner(test_expr1, test_expr2) is quadratic in test function (nonlinear/invalid).");
   return PullbackResult<>{};
}

// =============================================================================
// OuterExpr Pullback
// =============================================================================
//
// OuterExpr represents the outer product of two vector expressions.
//
// Pullback rule (narrow scope - test-linear LHS only):
//    pullback(outer(test_expr, test_free_vector), tensor_seed)
//       = pullback(test_expr, contracted_seed)
//
// Where:
//    contracted_seed[c] = sum_j tensor_seed[c][j] * test_free_vector[j]
//
// This contraction is implemented via ProductExpr with MatVec kind:
//    contracted_seed = tensor_seed * test_free_vector
//
// Target use case (vector SIPDG):
//    pullback(outer(jump(V), normal), A)
//       → pullback(jump(V), A * normal)
//
// NARROW SCOPE:
//    - Only implements outer(test_expr, test_free_vector)
//    - Does NOT implement outer(test_free, test_expr) unless compiler proves needed
//    - Rejects both-test-dependent cases (quadratic, invalid)
//

/**
 * @brief OuterExpr pullback: test-linear LHS, test-free RHS.
 *
 * For outer(test_expr, test_free_vector):
 *    Variation: outer(delta_test_expr, test_free_vector)
 *    Pullback seed for test_expr: contracted_seed = tensor_seed * test_free_vector
 *
 * The contraction uses ProductExpr (Matrix×Vector → Vector).
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_free_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Contract tensor seed with RHS vector:
   // contracted_seed[c] = sum_j seed[c][j] * expr.rhs[j]
   //
   // ProductExpr with MatVec kind implements this contraction.
   // Verified: product_kind_v<Matrix, Vector> == ProductKind::MatVec.
   auto contracted_seed = seed * expr.rhs;  // Matrix×Vector → Vector
   return pullback(expr.lhs, contracted_seed);
}

/**
 * @brief OuterExpr pullback: test-free LHS, test-linear RHS (not implemented).
 *
 * For outer(test_free_vector, test_expr):
 *    Not needed for vector SIPDG (RHS is always Normal/geometry).
 *    Reserved for future extension if compiler proves necessary.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_free_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "OuterExpr pullback: test-linear RHS not yet implemented. "
      "Only outer(test_expr, test_free_vector) is supported for vector SIPDG.");
   return PullbackResult<>{};
}

/**
 * @brief OuterExpr pullback: reject both-test-dependent case.
 *
 * If both operands are test-dependent, the expression is quadratic in test
 * function (nonlinear). Such expressions are invalid for weak forms.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "OuterExpr pullback: both operands test-dependent (quadratic in test function, invalid). "
      "outer(test_expr1, test_expr2) represents a nonlinear weak form.");
   return PullbackResult<>{};
}

// =============================================================================
// AverageExpr Pullback for Test Functions
// =============================================================================
//
// For interior facets, test functions only use minus side with 1/2 factor:
//    average(v) = 0.5 * v_minus
//
// Pullback rule:
//    pullback(average(v), seed) = pullback(v, 0.5 * seed)
//
// Trial functions (not implemented here):
//    average(u) = 0.5 * (u_minus + u_plus)  (handled by AverageExpr::operator())
//


/**
 * @brief Pullback for average(E) on interior facets when E is test-linear.
 *
 * With the current/minus-side convention used by the pullback operator:
 *
 *     average(E) = 0.5 * E_minus
 *
 * Therefore:
 *
 *     pullback(average(E), seed)
 *       = pullback(E, 0.5 * seed)
 *
 * This handles average(v), average(grad(v)), and more complex
 * test-linear expressions such as average(coef * grad(v)), provided
 * the inner expression E has a valid pullback rule.
 */
template<FieldExpr E, class Seed>
   requires (is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const AverageExpr<E>& expr, const Seed& seed)
{
   return pullback(expr.expr, 0.5 * seed);
}

/**
 * @brief Reject pullback of average(E) when E is not test-linear.
 *
 * Trial-only/test-free average expressions should be evaluated as coefficients
 * and should not enter the pullback path directly.
 */
template<FieldExpr E, class Seed>
   requires (!is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const AverageExpr<E>&, const Seed&)
{
   static_assert(dependent_false_v<E, Seed>,
      "AverageExpr pullback is only implemented for test-linear expressions. "
      "Trial-only/test-free average expressions should be evaluated, not pulled back.");
   return PullbackResult<>{};
}

// =============================================================================
// JumpExpr Pullback for Test Functions
// =============================================================================
//
// For interior facets, test functions only use minus side:
//    jump(v) = v_minus
//
// Pullback rule:
//    pullback(jump(v), seed) = pullback(v, seed)
//
// Trial functions (not implemented here):
//    jump(u) = u_minus - u_plus  (handled by JumpExpr::operator())
//
// NARROW SCOPE: Only handles JumpExpr<TestSpace<Name, Shape>>
//

/**
 * @brief Pullback for jump of test space (interior facets).
 *
 * For test functions on interior facets:
 *    jump(v) = v_minus (only current/minus side)
 *
 * Therefore:
 *    pullback(jump(v), seed) = pullback(v, seed) = ValueChannel with seed
 *
 * @tparam Name Test space name
 * @tparam Shape Test space field shape (Scalar or Vector)
 * @tparam Seed Seed expression type
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const JumpExpr<TestSpace<Name, Shape>>& jump_expr, const Seed& seed)
{
   // For test functions, jump only uses minus side
   // Extract underlying test space and pullback directly
   const auto& test_space = jump_expr.expr;
   return pullback(test_space, seed);
}

/**
 * @brief Placeholder for general JumpExpr pullback (not yet implemented).
 *
 * This overload catches JumpExpr of non-TestSpace expressions.
 * For trial functions, jump evaluation handles minus/plus automatically.
 * The specific TestSpace overload above will be preferred for test functions.
 */
template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const JumpExpr<E>& /*expr*/, const Seed& /*seed*/)
{
   // If this is called, it means E is not TestSpace
   static_assert(dependent_false_v<E, Seed>,
      "JumpExpr pullback only implemented for TestSpace. Trial functions handled by JumpExpr::operator().");
   return PullbackResult<>{};
}

// =============================================================================
// Canonical Global Interior Pullback
// =============================================================================
//
// The ordinary pullback rules above are row-relative: jump(v) contributes to
// the current/minus row only. Canonical global-face lowering evaluates both
// rows from one canonical face context, so test-side dependence is lowered into
// side-qualified channel keys instead. Trace selection is pushed structurally
// through the expression tree before pullback; this keeps public minus/plus as
// semantic DSL nodes without carrying a side template parameter through every
// pullback overload.

namespace details
{

template<FieldExpr Expr>
auto TraceToMinusSide(const Expr& expr)
{
   return minus(expr);
}

template<FieldExpr Expr>
auto TraceToPlusSide(const Expr& expr)
{
   return plus(expr);
}

inline auto TraceToMinusSide(const ScaleExpr& expr)
{
   return expr;
}

inline auto TraceToPlusSide(const ScaleExpr& expr)
{
   return expr;
}

inline auto TraceToMinusSide(const Normal& expr)
{
   return expr;
}

inline auto TraceToPlusSide(const Normal& expr)
{
   return expr;
}

template<FieldExpr Expr>
auto TraceToMinusSide(const MinusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as minus(minus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
auto TraceToMinusSide(const PlusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as minus(plus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
auto TraceToPlusSide(const MinusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as plus(minus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
auto TraceToPlusSide(const PlusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as plus(plus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
auto TraceToMinusSide(const NegExpr<Expr>& expr)
{
   return -TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
auto TraceToPlusSide(const NegExpr<Expr>& expr)
{
   return -TraceToPlusSide(expr.expr);
}

template<FieldExpr Head, FieldExpr... Tail>
auto TraceToMinusSide(const SumExpr<Head, Tail...>& expr)
{
   return std::apply(
      [] (const auto&... terms)
      {
         return (TraceToMinusSide(terms) + ...);
      },
      expr.terms);
}

template<FieldExpr Head, FieldExpr... Tail>
auto TraceToPlusSide(const SumExpr<Head, Tail...>& expr)
{
   return std::apply(
      [] (const auto&... terms)
      {
         return (TraceToPlusSide(terms) + ...);
      },
      expr.terms);
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToMinusSide(const DotExpr<LHS, RHS>& expr)
{
   return dot(TraceToMinusSide(expr.lhs), TraceToMinusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToPlusSide(const DotExpr<LHS, RHS>& expr)
{
   return dot(TraceToPlusSide(expr.lhs), TraceToPlusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToMinusSide(const InnerExpr<LHS, RHS>& expr)
{
   return inner(TraceToMinusSide(expr.lhs), TraceToMinusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToPlusSide(const InnerExpr<LHS, RHS>& expr)
{
   return inner(TraceToPlusSide(expr.lhs), TraceToPlusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToMinusSide(const ProductExpr<LHS, RHS>& expr)
{
   return TraceToMinusSide(expr.lhs) * TraceToMinusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToPlusSide(const ProductExpr<LHS, RHS>& expr)
{
   return TraceToPlusSide(expr.lhs) * TraceToPlusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToMinusSide(const MultFieldExpr<LHS, RHS>& expr)
{
   return TraceToMinusSide(expr.lhs) * TraceToMinusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
auto TraceToPlusSide(const MultFieldExpr<LHS, RHS>& expr)
{
   return TraceToPlusSide(expr.lhs) * TraceToPlusSide(expr.rhs);
}

template<FieldExpr Expr>
auto TraceToMinusSide(const JumpExpr<Expr>& expr)
{
   return TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
auto TraceToPlusSide(const JumpExpr<Expr>& expr)
{
   return - TraceToPlusSide(expr.expr);
}

template<FieldExpr Expr>
auto TraceToMinusSide(const AverageExpr<Expr>& expr)
{
   return 0.5 * TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
auto TraceToPlusSide(const AverageExpr<Expr>& expr)
{
   return 0.5 * TraceToPlusSide(expr.expr);
}

} // namespace details

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<TestSpace<Name, Shape>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{make_map(make_entry<ValueMinusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<TestSpace<Name, Shape>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{make_map(make_entry<ValuePlusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<GradientExpr<TestSpace<Name, Shape>>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{make_map(make_entry<GradientMinusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<GradientExpr<TestSpace<Name, Shape>>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{make_map(make_entry<GradientPlusChannel>(seed))};
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<Expr>& expr,
   const Seed& seed)
{
   auto side_expr = details::TraceToMinusSide(expr.expr);
   using SideExpr = std::remove_cvref_t<decltype(side_expr)>;

   static_assert(
      !std::is_same_v<SideExpr, MinusTraceExpr<Expr>>,
      "Canonical global interior pullback cannot pull back a pure minus-side "
      "test-free leaf. The integrand must be test-linear.");

   return pullback_global_interior(side_expr, seed);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<Expr>& expr,
   const Seed& seed)
{
   auto side_expr = details::TraceToPlusSide(expr.expr);
   using SideExpr = std::remove_cvref_t<decltype(side_expr)>;

   static_assert(
      !std::is_same_v<SideExpr, PlusTraceExpr<Expr>>,
      "Canonical global interior pullback cannot pull back a pure plus-side "
      "test-free leaf. The integrand must be test-linear.");

   return pullback_global_interior(side_expr, seed);
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const TestSpace<Name, Shape>&,
   const Seed&)
{
   static_assert(
      dependent_false_v<Seed>,
      "Unqualified TestSpace on canonical global interior facets is "
      "ambiguous. Use minus(v), plus(v), jump(v), or average(v).");
   return PullbackResult<>{};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const GradientExpr<TestSpace<Name, Shape>>&,
   const Seed&)
{
   static_assert(
      dependent_false_v<Seed>,
      "Unqualified grad(v) on canonical global interior facets is ambiguous. "
      "Use minus(grad(v)), plus(grad(v)), jump(grad(v)), or "
      "average(grad(v)).");
   return PullbackResult<>{};
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const JumpExpr<Expr>& expr,
   const Seed& seed)
{
   auto minus_channels =
      pullback_global_interior(details::TraceToMinusSide(expr.expr), seed);
   auto plus_channels =
      pullback_global_interior(details::TraceToPlusSide(expr.expr), -seed);
   return merge_pullback_results(minus_channels, plus_channels);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const AverageExpr<Expr>& expr,
   const Seed& seed)
{
   auto half_seed = 0.5 * seed;
   auto minus_channels =
      pullback_global_interior(details::TraceToMinusSide(expr.expr), half_seed);
   auto plus_channels =
      pullback_global_interior(details::TraceToPlusSide(expr.expr), half_seed);
   return merge_pullback_results(minus_channels, plus_channels);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const NegExpr<Expr>& expr,
   const Seed& seed)
{
   return pullback_global_interior(expr.expr, -seed);
}

template<FieldExpr Head, FieldExpr... Tail, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const SumExpr<Head, Tail...>& expr,
   const Seed& seed)
{
   using Expr = SumExpr<Head, Tail...>;
   using S = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return std::apply(
      [&](const auto&... terms)
      {
         return merge_pullback_results(
            pullback_global_interior(terms, seed)...);
      },
      expr.terms);
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const DotExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = DotExpr<LHS, RHS>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      return pullback_global_interior(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      return pullback_global_interior(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior DotExpr pullback requires exactly one "
         "TestLinear operand and one TestFree operand.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const InnerExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;

   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      return pullback_global_interior(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      return pullback_global_interior(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior InnerExpr pullback requires exactly one "
         "TestLinear operand and one TestFree operand.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const ProductExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = ProductExpr<LHS, RHS>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   if constexpr (Expr::product_kind == ProductKind::ScalarTimes)
   {
      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         static_assert(
            field_shape_v<R> == FieldShape::Scalar,
            "Canonical global interior scalar-times pullback requires the "
            "test-free multiplier to be Scalar.");
         return pullback_global_interior(expr.lhs, seed * expr.rhs);
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         static_assert(
            field_shape_v<L> == FieldShape::Scalar,
            "Canonical global interior scalar-times pullback requires the "
            "test-free multiplier to be Scalar.");
         return pullback_global_interior(expr.rhs, seed * expr.lhs);
      }
      else
      {
         static_assert(
            dependent_false_v<L, R, S>,
            "Canonical global interior ProductExpr pullback requires exactly "
            "one TestLinear operand and one TestFree scalar operand.");
      }
   }
   else if constexpr (Expr::product_kind == ProductKind::MatVec)
   {
      static_assert(
         field_shape_v<L> == FieldShape::Matrix,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Matrix-shaped left operand.");
      static_assert(
         field_shape_v<R> == FieldShape::Vector,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Vector-shaped right operand.");
      static_assert(
         field_shape_v<S> == FieldShape::Vector,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Vector-shaped seed.");

      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         return pullback_global_interior(
            expr.lhs,
            outer(seed, expr.rhs));
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         return pullback_global_interior(
            expr.rhs,
            transpose(expr.lhs) * seed);
      }
      else
      {
         static_assert(
            dependent_false_v<L, R, S>,
            "Canonical global interior MatVec ProductExpr pullback requires "
            "exactly one TestLinear operand. Supported directions are "
            "test-linear matrix * test-free vector and test-free matrix * "
            "test-linear vector.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior ProductExpr pullback currently supports "
         "scalar-times products and linear MatVec products.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MultFieldExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;

   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      static_assert(
         field_shape_v<R> == FieldShape::Scalar,
         "Canonical global interior multiplication pullback requires the "
         "test-free multiplier to be Scalar.");
      return pullback_global_interior(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      static_assert(
         field_shape_v<L> == FieldShape::Scalar,
         "Canonical global interior multiplication pullback requires the "
         "test-free multiplier to be Scalar.");
      return pullback_global_interior(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior multiplication pullback requires exactly "
         "one TestLinear operand and one TestFree scalar operand.");
   }
}

template<class Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const Expr&, const Seed&)
{
   static_assert(
      dependent_false_v<Expr, Seed>,
      "Canonical global interior pullback cannot lower this traced expression "
      "yet. The expression is structurally valid, but the canonical pullback "
      "must provide a side-qualified channel rule for its test-linear part.");
   return PullbackResult<>{};
}

} // namespace gendil
