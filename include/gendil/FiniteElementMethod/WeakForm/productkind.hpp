// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include <type_traits>

namespace gendil
{

// =============================================================================
// ProductKind classification for algebraic products
// =============================================================================

// ProductKind classifies the algebraic product operation implied by two
// FieldShape operands. This is purely shape-based and does NOT encode
// test-linearity, trial-dependence, coefficient role, or runtime behavior.
enum class ProductKind
{
   ScalarTimes,  // At least one operand is Scalar (S×S, S×V, V×S, S×M, M×S)
   MatVec,       // Matrix × Vector
   MatMat,       // Matrix × Matrix (standard matrix-matrix product/composition)
   Unsupported   // Vector × Vector, Vector × Matrix, or missing field_shape_v
};

// Note: has_field_shape_v is now defined in fieldshapetraits.hpp

// =============================================================================
// product_kind_v classification trait
// =============================================================================

// Base case: at least one operand lacks field_shape_v → Unsupported
template<class LHS, class RHS, bool HasShape =
   has_field_shape_v<LHS> && has_field_shape_v<RHS>>
struct product_kind
{
   static constexpr ProductKind value = ProductKind::Unsupported;
};

// Specialized case: both operands have field_shape_v → classify by shape
template<class LHS, class RHS>
struct product_kind<LHS, RHS, true>
{
   static constexpr auto lhs_shape =
      field_shape_v<std::remove_cvref_t<LHS>>;
   static constexpr auto rhs_shape =
      field_shape_v<std::remove_cvref_t<RHS>>;

   static constexpr ProductKind value =
      // Rule 2: If either is Scalar → ScalarTimes
      (lhs_shape == FieldShape::Scalar || rhs_shape == FieldShape::Scalar)
         ? ProductKind::ScalarTimes
      // Rule 3: Matrix × Vector → MatVec
      : (lhs_shape == FieldShape::Matrix && rhs_shape == FieldShape::Vector)
         ? ProductKind::MatVec
      // Rule 4: Matrix × Matrix → MatMat
      : (lhs_shape == FieldShape::Matrix && rhs_shape == FieldShape::Matrix)
         ? ProductKind::MatMat
      // Rule 5: Everything else → Unsupported
      : ProductKind::Unsupported;
};

// Variable template accessor (follows GenDiL convention: _v suffix)
template<class LHS, class RHS>
inline constexpr ProductKind product_kind_v =
   product_kind<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>::value;

// =============================================================================
// product_field_shape_v: Result shape of a product operation
// =============================================================================

// Computes the FieldShape resulting from a product classified by ProductKind.
// Specialized by ProductKind to avoid instantiating field_shape_v for
// unsupported combinations.
template<class LHS, class RHS, ProductKind Kind = product_kind_v<LHS, RHS>>
struct product_field_shape;

// ScalarTimes: Result shape is the non-scalar operand, or Scalar if both Scalar
template<class LHS, class RHS>
struct product_field_shape<LHS, RHS, ProductKind::ScalarTimes>
{
   static constexpr auto lhs_shape =
      field_shape_v<std::remove_cvref_t<LHS>>;
   static constexpr auto rhs_shape =
      field_shape_v<std::remove_cvref_t<RHS>>;

   static constexpr FieldShape value =
      lhs_shape == FieldShape::Scalar ? rhs_shape : lhs_shape;
};

// MatVec: Matrix × Vector → Vector
template<class LHS, class RHS>
struct product_field_shape<LHS, RHS, ProductKind::MatVec>
{
   static constexpr FieldShape value = FieldShape::Vector;
};

// MatMat: Matrix × Matrix → Matrix
template<class LHS, class RHS>
struct product_field_shape<LHS, RHS, ProductKind::MatMat>
{
   static constexpr FieldShape value = FieldShape::Matrix;
};

// Unsupported: Trigger clear error when explicitly requested
template<class LHS, class RHS>
struct product_field_shape<LHS, RHS, ProductKind::Unsupported>
{
   static_assert(dependent_false_v<LHS, RHS>,
      "product_field_shape_v requested for an unsupported product.");
};

// Variable template accessor
template<class LHS, class RHS>
inline constexpr FieldShape product_field_shape_v =
   product_field_shape<std::remove_cvref_t<LHS>,
                       std::remove_cvref_t<RHS>>::value;

// =============================================================================
// field_shape_v specialization for ProductExpr
// =============================================================================

// Note: ProductExpr is forward-declared in dslbase.hpp with FieldExpr constraints

// ProductExpr result shape is determined by product_field_shape_v
template<FieldExpr LHS, FieldExpr RHS>
struct field_shape_impl<ProductExpr<LHS, RHS>>
{
   static constexpr FieldShape value = product_field_shape_v<LHS, RHS>;
};

// =============================================================================
// Specialized MatVecExpr exclusion helper
// =============================================================================

/**
 * @brief Predicate for the specialized MatVecExpr normal contraction syntax.
 *
 * Returns true iff the expression matches:
 *   grad(VectorTestSpace) * Normal  OR  grad(VectorTrialSpace) * Normal
 *
 * This specialized path returns MatVecExpr and must be excluded from generic
 * ProductExpr MatVec syntax.
 */
template<class LHS, class RHS>
inline constexpr bool is_specialized_matvec_syntax_v = false;

// Specialization for GradientExpr<VectorField> * Normal
template<class E>
   requires is_vector_field_reference_v<std::remove_cvref_t<E>>
inline constexpr bool is_specialized_matvec_syntax_v<GradientExpr<E>, Normal> = true;

// =============================================================================
// Predicate: should ProductExpr handle this operator* syntax?
// =============================================================================

/**
 * @brief Predicate identifying ProductExpr syntax that ProductExpr handles.
 *
 * True when:
 * - Both operands are FieldExpr
 * - Both operands have field_shape_v
 * - ProductKind is ScalarTimes OR MatMat OR (MatVec except specialized normal contraction)
 *
 * This predicate is used to make ProductExpr operator* and MultFieldExpr operator*
 * mutually exclusive.
 */
template<class LHS, class RHS>
inline constexpr bool is_productexpr_syntax_candidate_v =
   FieldExpr<std::remove_cvref_t<LHS>> &&
   FieldExpr<std::remove_cvref_t<RHS>> &&
   HasFieldShape<std::remove_cvref_t<LHS>> &&
   HasFieldShape<std::remove_cvref_t<RHS>> &&
   (product_kind_v<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>
      == ProductKind::ScalarTimes ||
    product_kind_v<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>
      == ProductKind::MatMat ||
    (product_kind_v<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>
      == ProductKind::MatVec &&
     !is_specialized_matvec_syntax_v<std::remove_cvref_t<LHS>, std::remove_cvref_t<RHS>>));

// Backward compatibility alias (deprecated, for transition only)
template<class LHS, class RHS>
inline constexpr bool is_productexpr_scalar_syntax_candidate_v =
   is_productexpr_syntax_candidate_v<LHS, RHS>;

} // namespace gendil
