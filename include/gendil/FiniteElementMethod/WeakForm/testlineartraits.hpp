// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

// Forward declarations for expression types
namespace gendil {
   template<typename, typename> struct MatVecExpr;
}

namespace gendil
{

/**
 * @brief Classification of expressions with respect to test-space linearity.
 */
enum class TestLinearity {
   TestFree,           // No test-space dependence
   TestLinear,         // Linear in exactly one test field
   NonlinearInTest,    // Nonlinear in test function (invalid for weak forms)
   Unsupported         // Multiple test fields or ambiguous
};

/**
 * @brief Primary trait for classifying expression test-linearity.
 */
template<typename Expr>
struct test_linearity;

// Convenience traits
template<typename Expr>
inline constexpr bool is_test_free_v =
   (test_linearity<std::remove_cvref_t<Expr>>::value == TestLinearity::TestFree);

template<typename Expr>
inline constexpr bool is_test_linear_v =
   (test_linearity<std::remove_cvref_t<Expr>>::value == TestLinearity::TestLinear);

template<typename Expr>
inline constexpr bool is_nonlinear_in_test_v =
   (test_linearity<std::remove_cvref_t<Expr>>::value == TestLinearity::NonlinearInTest);

template<typename Expr>
inline constexpr bool is_unsupported_test_linear_v =
   (test_linearity<std::remove_cvref_t<Expr>>::value == TestLinearity::Unsupported);

template<typename Expr>
inline constexpr auto test_name_v = test_linearity<std::remove_cvref_t<Expr>>::test_name;

// ============================================================================
// Leaf Node Classifications
// ============================================================================

template<StaticString Name, FieldShape Shape>
struct test_linearity<TrialSpace<Name, Shape>> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<StaticString Name, FieldShape Shape>
struct test_linearity<TestSpace<Name, Shape>> {
   static constexpr TestLinearity value = TestLinearity::TestLinear;
   static constexpr auto test_name = Name;
};

template<StaticString Name>
struct test_linearity<FiniteElementField<Name>> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct test_linearity<Coefficient<Name, Shape, Fn, Inputs...>> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<>
struct test_linearity<ScaleExpr> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<>
struct test_linearity<Normal> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

// ============================================================================
// Unary Operators: Propagate child linearity
// ============================================================================

template<FieldExpr E>
struct test_linearity<GradientExpr<E>> {
   static constexpr TestLinearity value = test_linearity<E>::value;
   static constexpr auto test_name = test_linearity<E>::test_name;
};

template<FieldExpr E>
struct test_linearity<JumpExpr<E>> {
   static constexpr TestLinearity value = test_linearity<E>::value;
   static constexpr auto test_name = test_linearity<E>::test_name;
};

template<FieldExpr E>
struct test_linearity<AverageExpr<E>> {
   static constexpr TestLinearity value = test_linearity<E>::value;
   static constexpr auto test_name = test_linearity<E>::test_name;
};

template<FieldExpr E>
struct test_linearity<NegExpr<E>> {
   static constexpr TestLinearity value = test_linearity<E>::value;
   static constexpr auto test_name = test_linearity<E>::test_name;
};

template<FieldExpr AdvExpr, FieldExpr E>
struct test_linearity<UpwindExpr<AdvExpr, E>> {
   static constexpr TestLinearity value = test_linearity<E>::value;
   static constexpr auto test_name = test_linearity<E>::test_name;
};

// ============================================================================
// Binary Operators: Helpers using partial specialization
// ============================================================================

namespace detail {

// Helper for MultFieldExpr: specialized on LHS/RHS linearity
template<TestLinearity LValue, TestLinearity RValue, class LHS, class RHS>
struct combine_mult_linearity;

// TestFree * TestFree -> TestFree
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestFree, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

// TestLinear * TestLinear -> NonlinearInTest
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestLinear, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

// TestFree * TestLinear -> Unsupported (ambiguous: scalar mult, normal contraction, etc.)
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestFree, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

// TestLinear * TestFree -> Unsupported (ambiguous)
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestLinear, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

// If either is NonlinearInTest, result is NonlinearInTest
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::NonlinearInTest, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestFree, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::NonlinearInTest, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestLinear, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::NonlinearInTest, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

// Catch-all for Unsupported combinations
template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::Unsupported, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestFree, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::Unsupported, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::TestLinear, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::Unsupported, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::Unsupported, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_mult_linearity<TestLinearity::NonlinearInTest, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

// Helper for pairing (DotExpr/InnerExpr)
template<TestLinearity LValue, TestLinearity RValue, class LHS, class RHS>
struct combine_pairing_linearity;

// TestFree paired with TestLinear -> TestLinear (take name from TestLinear side)
template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestFree, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::TestLinear;
   static constexpr auto test_name = test_linearity<RHS>::test_name;
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestLinear, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::TestLinear;
   static constexpr auto test_name = test_linearity<LHS>::test_name;
};

// TestLinear paired with TestLinear -> NonlinearInTest
template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestLinear, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

// TestFree paired with TestFree -> TestFree
template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestFree, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

// NonlinearInTest cases
template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::NonlinearInTest, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestFree, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::NonlinearInTest, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestLinear, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::NonlinearInTest, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

// Unsupported cases
template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::Unsupported, TestLinearity::TestFree, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestFree, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::Unsupported, TestLinearity::TestLinear, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::TestLinear, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::Unsupported, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::Unsupported, TestLinearity::NonlinearInTest, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<class LHS, class RHS>
struct combine_pairing_linearity<TestLinearity::NonlinearInTest, TestLinearity::Unsupported, LHS, RHS> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

} // namespace detail

// ============================================================================
// Binary Operators: MultFieldExpr
// ============================================================================

template<FieldExpr LHS, FieldExpr RHS>
struct test_linearity<MultFieldExpr<LHS, RHS>>
   : detail::combine_mult_linearity<
      test_linearity<LHS>::value,
      test_linearity<RHS>::value,
      LHS, RHS>
{};

// ============================================================================
// Binary Operators: DotExpr
// ============================================================================

template<FieldExpr LHS, FieldExpr RHS>
struct test_linearity<DotExpr<LHS, RHS>>
   : detail::combine_pairing_linearity<
      test_linearity<LHS>::value,
      test_linearity<RHS>::value,
      LHS, RHS>
{};

// ============================================================================
// Binary Operators: InnerExpr
// ============================================================================

template<FieldExpr LHS, FieldExpr RHS>
struct test_linearity<InnerExpr<LHS, RHS>>
   : detail::combine_pairing_linearity<
      test_linearity<LHS>::value,
      test_linearity<RHS>::value,
      LHS, RHS>
{};

// ============================================================================
// Binary Operators: OuterExpr
// ============================================================================

/**
 * @brief Test-linearity for OuterExpr (outer product).
 *
 * OuterExpr is test-linear ONLY if exactly one operand is test-linear (XOR rule).
 *
 * Valid test-linear cases:
 *    outer(test_expr, test_free_expr) → TestLinear
 *    outer(test_free_expr, test_expr) → TestLinear
 *
 * Invalid quadratic case:
 *    outer(test_expr_1, test_expr_2) → NonlinearInTest
 *
 * Test-free case:
 *    outer(test_free_1, test_free_2) → TestFree
 *
 * Uses combine_pairing_linearity helper (same as DotExpr/InnerExpr).
 */
template<FieldExpr LHS, FieldExpr RHS>
struct test_linearity<OuterExpr<LHS, RHS>>
   : detail::combine_pairing_linearity<
      test_linearity<LHS>::value,
      test_linearity<RHS>::value,
      LHS, RHS>
{};

// ============================================================================
// N-ary Operators: SumExpr
// ============================================================================

namespace detail {

// Helper for summing linearities
template<typename Head, typename... Tail>
struct sum_linearity;

// Base case: single term
template<FieldExpr T>
struct sum_linearity<T> {
   static constexpr TestLinearity value = test_linearity<T>::value;
   static constexpr auto test_name = test_linearity<T>::test_name;
};

// Recursive case via partial specialization on linearity values
template<TestLinearity HeadLin, TestLinearity RestLin, typename Head, typename... Tail>
struct sum_linearity_combine;

// TestFree + TestFree -> TestFree
template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestFree, TestLinearity::TestFree, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

// TestLinear + TestLinear (check name match at runtime, but type-wise we need specialization)
// For now, conservatively: different implementation needed
// We'll handle this by making the recursive case check explicitly

// Actually, for SumExpr we need to compare test names at compile time, which is tricky
// Let's use a simpler approach: classify based on linearity only, and for TestLinear+TestLinear
// we'll just require they have the same name (checked elsewhere if needed)

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestLinear, TestLinearity::TestLinear, Head, Tail...> {
   // For now, assume TestLinear + TestLinear -> TestLinear (name match checked elsewhere)
   // Conservative: could make this Unsupported
   static constexpr TestLinearity value = TestLinearity::TestLinear;
   static constexpr auto test_name = test_linearity<Head>::test_name;
};

// TestFree + TestLinear -> Unsupported
template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestFree, TestLinearity::TestLinear, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestLinear, TestLinearity::TestFree, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

// NonlinearInTest + anything -> NonlinearInTest
template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::NonlinearInTest, TestLinearity::TestFree, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestFree, TestLinearity::NonlinearInTest, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::NonlinearInTest, TestLinearity::TestLinear, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestLinear, TestLinearity::NonlinearInTest, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::NonlinearInTest, TestLinearity::NonlinearInTest, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::NonlinearInTest;
   static constexpr auto test_name = StaticString{""};
};

// Unsupported cases
template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::Unsupported, TestLinearity::TestFree, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestFree, TestLinearity::Unsupported, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::Unsupported, TestLinearity::TestLinear, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::TestLinear, TestLinearity::Unsupported, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::Unsupported, TestLinearity::Unsupported, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::Unsupported, TestLinearity::NonlinearInTest, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

template<typename Head, typename... Tail>
struct sum_linearity_combine<TestLinearity::NonlinearInTest, TestLinearity::Unsupported, Head, Tail...> {
   static constexpr TestLinearity value = TestLinearity::Unsupported;
   static constexpr auto test_name = StaticString{""};
};

// Recursive case
template<FieldExpr Head, FieldExpr... Tail>
struct sum_linearity<Head, Tail...>
   : sum_linearity_combine<
      test_linearity<Head>::value,
      sum_linearity<Tail...>::value,
      Head, Tail...>
{};

} // namespace detail

// SumExpr classification
template<FieldExpr Head, FieldExpr... Tail>
struct test_linearity<SumExpr<Head, Tail...>>
   : detail::sum_linearity<Head, Tail...>
{};

// ============================================================================
// Form-level: Integrand propagates integrand expression linearity
// ============================================================================

template<DomainExpr Domain, FieldExpr Expr>
struct test_linearity<Integrand<Domain, Expr>> {
   static constexpr TestLinearity value = test_linearity<Expr>::value;
   static constexpr auto test_name = test_linearity<Expr>::test_name;
};

// ============================================================================
// MatVecExpr - Rank-2 × Rank-1 Matrix-Vector Product
// ============================================================================

template<typename MatrixExpr, typename VectorExpr>
struct test_linearity<MatVecExpr<MatrixExpr, VectorExpr>> {
   // Vector must be test-free (e.g., Normal, prescribed direction)
   static_assert(is_test_free_v<VectorExpr>,
      "VectorExpr in MatVecExpr must be test-free. "
      "Matrix-vector contraction with test-dependent vector is nonlinear or unsupported.");

   // Propagate linearity from matrix expression
   static constexpr TestLinearity value = test_linearity<MatrixExpr>::value;
   static constexpr auto test_name = test_linearity<MatrixExpr>::test_name;
};

// ============================================================================
// ProductExpr - Algebraic Product (ScalarTimes, MatVec, MatMat)
// ============================================================================

template<FieldExpr LHS, FieldExpr RHS>
struct test_linearity<ProductExpr<LHS, RHS>>
   : detail::combine_pairing_linearity<
      test_linearity<LHS>::value,
      test_linearity<RHS>::value,
      LHS, RHS>
{};

} // namespace gendil
