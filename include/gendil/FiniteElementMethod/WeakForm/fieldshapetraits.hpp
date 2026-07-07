// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include <type_traits>

namespace gendil
{

// =============================================================================
// Field shape traits
// =============================================================================

// Boolean trait: is this a field reference (TrialSpace or TestSpace)?
template<typename T>
struct is_field_reference : std::false_type {};

template<StaticString Name, FieldShape Shape>
struct is_field_reference<TrialSpace<Name, Shape>> : std::true_type {};

template<StaticString Name, FieldShape Shape>
struct is_field_reference<TestSpace<Name, Shape>> : std::true_type {};

template<typename T>
inline constexpr bool is_field_reference_v = is_field_reference<std::remove_cvref_t<T>>::value;

// Boolean trait: is this a scalar field reference?
template<typename T>
struct is_scalar_field_reference : std::false_type {};

template<StaticString Name>
struct is_scalar_field_reference<TrialSpace<Name, FieldShape::Scalar>> : std::true_type {};

template<StaticString Name>
struct is_scalar_field_reference<TestSpace<Name, FieldShape::Scalar>> : std::true_type {};

template<typename T>
inline constexpr bool is_scalar_field_reference_v = is_scalar_field_reference<std::remove_cvref_t<T>>::value;

// Boolean trait: is this a vector field reference?
template<typename T>
struct is_vector_field_reference : std::false_type {};

template<StaticString Name>
struct is_vector_field_reference<TrialSpace<Name, FieldShape::Vector>> : std::true_type {};

template<StaticString Name>
struct is_vector_field_reference<TestSpace<Name, FieldShape::Vector>> : std::true_type {};

template<typename T>
inline constexpr bool is_vector_field_reference_v = is_vector_field_reference<std::remove_cvref_t<T>>::value;

// Extract field shape (only use when T is known to be a field reference)
// Do NOT use field_shape_v<T> in generic boolean expressions
template<typename T>
struct field_shape_impl;

template<StaticString Name, FieldShape Shape>
struct field_shape_impl<TrialSpace<Name, Shape>> {
   static constexpr FieldShape value = Shape;
};

template<StaticString Name, FieldShape Shape>
struct field_shape_impl<TestSpace<Name, Shape>> {
   static constexpr FieldShape value = Shape;
};

template<typename T>
inline constexpr FieldShape field_shape_v = field_shape_impl<std::remove_cvref_t<T>>::value;

// =============================================================================
// Coefficient shape specializations
// =============================================================================

template<StaticString Name, FieldShape Shape, typename Fn, CoefficientInput... Inputs>
struct field_shape_impl<Coefficient<Name, Shape, Fn, Inputs...>> {
   static constexpr FieldShape value = Shape;
};

// =============================================================================
// Fixed-shape expression nodes
// =============================================================================

// Normal is a geometry vector (always Vector)
template<>
struct field_shape_impl<Normal> {
   static constexpr FieldShape value = FieldShape::Vector;
};

// ScaleExpr is a scalar literal (always Scalar)
template<>
struct field_shape_impl<ScaleExpr> {
   static constexpr FieldShape value = FieldShape::Scalar;
};

// =============================================================================
// GradientExpr shape specializations
// =============================================================================

// grad(scalar field) → vector (gradient of scalar is vector)
template<StaticString Name>
struct field_shape_impl<GradientExpr<TrialSpace<Name, FieldShape::Scalar>>> {
   static constexpr FieldShape value = FieldShape::Vector;
};

template<StaticString Name>
struct field_shape_impl<GradientExpr<TestSpace<Name, FieldShape::Scalar>>> {
   static constexpr FieldShape value = FieldShape::Vector;
};

// grad(vector field) → matrix (gradient of vector is matrix)
template<StaticString Name>
struct field_shape_impl<GradientExpr<TrialSpace<Name, FieldShape::Vector>>> {
   static constexpr FieldShape value = FieldShape::Matrix;
};

template<StaticString Name>
struct field_shape_impl<GradientExpr<TestSpace<Name, FieldShape::Vector>>> {
   static constexpr FieldShape value = FieldShape::Matrix;
};

// =============================================================================
// FieldShape propagation for shape-preserving wrapper expressions
// =============================================================================

/**
 * @brief NegExpr preserves the field shape of its operand.
 *
 * Negating a scalar/vector/matrix expression does not change its shape.
 */
template<FieldExpr Expr>
struct field_shape_impl<NegExpr<Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

template<FieldExpr Expr>
struct field_shape_impl<MinusTraceExpr<Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

template<FieldExpr Expr>
struct field_shape_impl<PlusTraceExpr<Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

/**
 * @brief AverageExpr preserves the field shape of its operand.
 *
 * Averaging a scalar/vector/matrix expression does not change its shape.
 * Facet semantics (averaging across facet) are lowering concerns, not shape.
 */
template<FieldExpr Expr>
struct field_shape_impl<AverageExpr<Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

/**
 * @brief JumpExpr preserves the field shape of its operand.
 *
 * Taking a jump of a scalar/vector/matrix expression does not change its shape.
 * Facet semantics (jump across facet) are lowering concerns, not shape.
 */
template<FieldExpr Expr>
struct field_shape_impl<JumpExpr<Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

/**
 * @brief UpwindExpr preserves the field shape of its operand.
 *
 * Taking an upwind of a scalar/vector/matrix expression does not change its shape.
 * Facet semantics (upwind across facet) are lowering concerns, not shape.
 *
 */
template<FieldExpr AdvExpr, FieldExpr Expr>
struct field_shape_impl<UpwindExpr<AdvExpr, Expr>>
{
   static constexpr FieldShape value = field_shape_v<std::remove_cvref_t<Expr>>;
};

// =============================================================================
// Pairing operations (always produce scalars)
// =============================================================================

// Dot product: vector · vector → scalar
template<typename LHS, typename RHS>
struct field_shape_impl<DotExpr<LHS, RHS>> {
   // Note: Operand validation is enforced in DotExpr class definition via static_assert.
   // If DotExpr is instantiated with invalid operands, the error triggers there.
   // This specialization assumes valid operands and always returns Scalar.
   static constexpr FieldShape value = FieldShape::Scalar;
};

// Note: OuterExpr field_shape_impl is defined later in this file,
// after is_valid_outer_expr_v is defined (needed for validation).

// Note: InnerExpr field_shape_impl is defined later in this file,
// after is_valid_inner_expr_v is defined (needed for validation).

// =============================================================================
// SumExpr shape validation
// =============================================================================

/**
 * @brief SumExpr requires all terms to have the same FieldShape.
 *
 * The shape of a sum is the shape of its first term (Head), provided all
 * other terms (Tail...) have the same shape. If any term has a different
 * shape, requesting field_shape_v<SumExpr<...>> fails with a static_assert.
 *
 * Examples:
 *   u + v               → Scalar (both scalar)
 *   grad(v) + beta      → Vector (both vector)
 *   grad(vv) + A        → Matrix (both matrix)
 *
 * Invalid (static_assert failure):
 *   v + grad(v)         → Scalar + Vector (incompatible shapes)
 *   beta + A            → Vector + Matrix (incompatible shapes)
 */
template<FieldExpr Head, FieldExpr... Tail>
struct field_shape_impl<SumExpr<Head, Tail...>>
{
   static constexpr FieldShape head_shape =
      field_shape_v<std::remove_cvref_t<Head>>;

   static_assert(
      ((field_shape_v<std::remove_cvref_t<Tail>> == head_shape) && ...),
      "SumExpr terms must all have the same FieldShape.");

   static constexpr FieldShape value = head_shape;
};

// =============================================================================
// SFINAE-safe has_field_shape trait
// =============================================================================

// Detects whether an expression type T has a field_shape_v specialization.
// This uses SFINAE to avoid hard errors on types without field_shape_impl.
template<class T, class = void>
struct has_field_shape : std::false_type {};

template<class T>
struct has_field_shape<T, std::void_t<
   decltype(field_shape_impl<std::remove_cvref_t<T>>::value)
>> : std::true_type {};

template<class T>
inline constexpr bool has_field_shape_v = has_field_shape<std::remove_cvref_t<T>>::value;

// =============================================================================
// InnerExpr operand validation
// =============================================================================

/**
 * @brief SFINAE-safe validation trait for InnerExpr operands.
 *
 * Valid combinations:
 *   - Vector × Vector → inner product (e.g., inner(grad(scalar_u), grad(scalar_v)))
 *   - Matrix × Matrix → Frobenius inner product (e.g., inner(grad(vector_u), grad(vector_v)))
 *
 * Invalid combinations:
 *   - Scalar × Scalar → Use operator* for scalar multiplication
 *   - Matrix × Vector, Vector × Matrix → Use operator* for matvec products
 *   - Any combination involving scalar → Use operator* for scalar products
 *
 * This two-level SFINAE-safe pattern prevents hard errors when checking field_shape_v
 * on types that don't have has_field_shape_v.
 */
template<class LHS, class RHS, bool HasShape =
   has_field_shape_v<LHS> &&
   has_field_shape_v<RHS>>
struct is_valid_inner_expr_impl : std::false_type {};

template<class LHS, class RHS>
struct is_valid_inner_expr_impl<LHS, RHS, true>
   : std::bool_constant<
        (
           field_shape_v<std::remove_cvref_t<LHS>> == FieldShape::Vector &&
           field_shape_v<std::remove_cvref_t<RHS>> == FieldShape::Vector
        ) ||
        (
           field_shape_v<std::remove_cvref_t<LHS>> == FieldShape::Matrix &&
           field_shape_v<std::remove_cvref_t<RHS>> == FieldShape::Matrix
        )>
{};

template<class LHS, class RHS>
inline constexpr bool is_valid_inner_expr_v =
   is_valid_inner_expr_impl<LHS, RHS>::value;

// =============================================================================
// OuterExpr validation
// =============================================================================

/**
 * @brief SFINAE-safe trait to check if OuterExpr operands are valid.
 *
 * OuterExpr represents an outer product: Vector × Vector → Matrix.
 *
 * Valid operands must:
 * - Both have field_shape_v specializations (HasFieldShape)
 * - Both have FieldShape::Vector
 *
 */
template<class LHS, class RHS, bool HasShape =
   has_field_shape_v<LHS> &&
   has_field_shape_v<RHS>>
struct is_valid_outer_expr_impl : std::false_type {};

template<class LHS, class RHS>
struct is_valid_outer_expr_impl<LHS, RHS, true>
   : std::bool_constant<
      field_shape_v<std::remove_cvref_t<LHS>> == FieldShape::Vector &&
      field_shape_v<std::remove_cvref_t<RHS>> == FieldShape::Vector>
{};

template<class LHS, class RHS>
inline constexpr bool is_valid_outer_expr_v =
   is_valid_outer_expr_impl<LHS, RHS>::value;

// =============================================================================
// OuterExpr field_shape_impl (after validation trait is defined)
// =============================================================================

// Outer product: vector × vector → matrix
// Note: Validation is enforced both in OuterExpr class AND here in field_shape_impl.
template<typename LHS, typename RHS>
struct field_shape_impl<OuterExpr<LHS, RHS>> {
   static_assert(is_valid_outer_expr_v<LHS, RHS>,
      "OuterExpr requires Vector×Vector operands.");
   static constexpr FieldShape value = FieldShape::Matrix;
};

// =============================================================================
// InnerExpr field_shape_impl (after validation trait is defined)
// =============================================================================

// Inner product: inner(vector·vector) or inner(matrix:matrix) → scalar
// Note: Validation is enforced both in InnerExpr class templates AND here in field_shape_impl.
// Querying field_shape_v may not instantiate the InnerExpr class body, so we validate here too.
template<typename LHS, typename RHS>
struct field_shape_impl<InnerExpr<LHS, RHS>> {
   static_assert(is_valid_inner_expr_v<LHS, RHS>,
      "InnerExpr operands must be matching non-scalar shapes (Vector·Vector or Matrix:Matrix).");
   static constexpr FieldShape value = FieldShape::Scalar;
};

// =============================================================================
// Field-shape expression concepts
// =============================================================================

// Concept: expression has a field_shape_v specialization
template<class T>
concept HasFieldShape =
   has_field_shape_v<std::remove_cvref_t<T>>;

// Concept: expression is a scalar field
template<class T>
concept ScalarExpr =
   FieldExpr<std::remove_cvref_t<T>> &&
   HasFieldShape<T> &&
   field_shape_v<std::remove_cvref_t<T>> == FieldShape::Scalar;

// Concept: expression is a vector field
template<class T>
concept VectorExpr =
   FieldExpr<std::remove_cvref_t<T>> &&
   HasFieldShape<T> &&
   field_shape_v<std::remove_cvref_t<T>> == FieldShape::Vector;

// Concept: expression is a matrix field
template<class T>
concept MatrixExpr =
   FieldExpr<std::remove_cvref_t<T>> &&
   HasFieldShape<T> &&
   field_shape_v<std::remove_cvref_t<T>> == FieldShape::Matrix;

} // namespace gendil
