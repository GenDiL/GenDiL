// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymappingtranspose.hpp"
#include "gendil/Utilities/MathHelperFunctions/inner.hpp"

namespace gendil
{

// =============================================================================
// Helper trait to detect GradientExpr<TestSpace<...>>
// =============================================================================

// TODO: This is a temporary implementation detail for disambiguating the
// InnerExpr partial specializations below. Once a general pullback/lowering
// mechanism exists, this helper should be removed along with the specializations.

template<typename T>
struct is_gradient_of_test_space : std::false_type {};

template<StaticString Name, FieldShape Shape>
struct is_gradient_of_test_space<GradientExpr<TestSpace<Name, Shape>>> : std::true_type {};

template<typename T>
inline constexpr bool is_gradient_of_test_space_v = is_gradient_of_test_space<std::remove_cvref_t<T>>::value;

// =============================================================================
// InnerExpr expression template
// =============================================================================

// Generic inner product expression
// Valid: inner(Vector, Vector) or inner(Matrix, Matrix)
// Invalid: Scalar operands or mixed ranks
template<FieldExpr LHS, FieldExpr RHS>
struct InnerExpr : FieldBase
{
   static_assert(is_valid_inner_expr_v<LHS, RHS>,
      "InnerExpr requires matching non-scalar operands: Vector·Vector or Matrix:Matrix. "
      "Use operator* for scalar multiplication and matrix-vector/matrix-matrix products.");

   LHS lhs;
   RHS rhs;

   GENDIL_HOST_DEVICE
   InnerExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template<typename... Args>
   GENDIL_HOST_DEVICE
   auto operator()(Args&&... args) const
   {
      return Inner(lhs(std::forward<Args>(args)...), rhs(std::forward<Args>(args)...));
   }
};

// =============================================================================
// Temporary compatibility specializations for test-gradient lowering
// =============================================================================

// TODO: These test-gradient specializations are a temporary compatibility bridge
// for the current lowering path. Once a general pullback/lowering mechanism exists,
// InnerExpr should be reduced to a single generic algebraic node and these
// specializations should be removed.

// Specialization for inner(lhs, grad(v)) where v is a test space and lhs is NOT grad(test)
// Evaluates lhs and applies mapping, but does NOT evaluate grad(v)
// The requires clause makes this specialization mutually exclusive with the symmetric one below.
template<FieldExpr LHS, StaticString Name, FieldShape Shape>
   requires (!is_gradient_of_test_space_v<LHS>)
struct InnerExpr<LHS, GradientExpr<TestSpace<Name, Shape>>> : FieldBase
{
   static_assert(is_valid_inner_expr_v<LHS, GradientExpr<TestSpace<Name, Shape>>>,
      "InnerExpr requires matching non-scalar operands: Vector·Vector or Matrix:Matrix. "
      "Use operator* for scalar multiplication and matrix-vector/matrix-matrix products.");

   LHS lhs;
   using RHS = GradientExpr<TestSpace<Name, Shape>>;
   RHS rhs;

   GENDIL_HOST_DEVICE
   InnerExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template<
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields>
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const ElementContext& element_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields) const
   {
      auto grad_q = lhs(kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields);
      ApplyMappingTranspose(quad_pt_context.inv_J_mesh, grad_q);
      return grad_q;
   }
};

// Specialization for inner(grad(v), rhs) - symmetric case
// where v is a test space and rhs is NOT grad(test)
// The requires clause makes this specialization mutually exclusive with the one above.
template<FieldExpr RHS, StaticString Name, FieldShape Shape>
   requires (!is_gradient_of_test_space_v<RHS>)
struct InnerExpr<GradientExpr<TestSpace<Name, Shape>>, RHS> : FieldBase
{
   static_assert(is_valid_inner_expr_v<GradientExpr<TestSpace<Name, Shape>>, RHS>,
      "InnerExpr requires matching non-scalar operands: Vector·Vector or Matrix:Matrix. "
      "Use operator* for scalar multiplication and matrix-vector/matrix-matrix products.");

   using LHS = GradientExpr<TestSpace<Name, Shape>>;
   LHS lhs;
   RHS rhs;

   GENDIL_HOST_DEVICE
   InnerExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template<
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields>
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const ElementContext& element_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields) const
   {
      auto grad_q = rhs(kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields);
      ApplyMappingTranspose(quad_pt_context.inv_J_mesh, grad_q);
      return grad_q;
   }
};

template<FieldExpr LHS, FieldExpr RHS>
std::ostream& operator<<(std::ostream& os, const InnerExpr<LHS, RHS>& inner)
{
   return os << "(" << inner.lhs << ":" << inner.rhs << ")";
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto inner(const LHS& lhs, const RHS& rhs)
{
   return InnerExpr<LHS, RHS>(lhs, rhs);
}

} // namespace gendil
