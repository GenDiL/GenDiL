// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymappingtranspose.hpp"

namespace gendil
{

// =============================================================================
// DotExpr operand validation
// =============================================================================

/**
 * @brief SFINAE-safe trait to check if DotExpr operands are valid.
 *
 * DotExpr represents a vector dot product: Vector · Vector → Scalar.
 *
 * Valid operands must:
 * - Both have field_shape_v specializations (HasFieldShape)
 * - Both have FieldShape::Vector
 *
 * Invalid operand combinations that should use other operators:
 * - Scalar × Scalar → Use operator* / ProductExpr for scalar multiplication
 * - Matrix × Vector → Use operator* for matrix-vector ProductExpr
 * - Matrix × Matrix → Use inner() for Frobenius inner product if supported
 */
template<class LHS, class RHS, bool HasShape =
   HasFieldShape<std::remove_cvref_t<LHS>> &&
   HasFieldShape<std::remove_cvref_t<RHS>>>
struct is_valid_dot_expr_impl : std::false_type {};

template<class LHS, class RHS>
struct is_valid_dot_expr_impl<LHS, RHS, true>
   : std::bool_constant<
      field_shape_v<std::remove_cvref_t<LHS>> == FieldShape::Vector &&
      field_shape_v<std::remove_cvref_t<RHS>> == FieldShape::Vector>
{};

template<class LHS, class RHS>
inline constexpr bool is_valid_dot_expr_v =
   is_valid_dot_expr_impl<LHS, RHS>::value;

// =============================================================================
// DotExpr definition with operand validation
// =============================================================================

template < FieldExpr LHS, FieldExpr RHS >
struct DotExpr : FieldBase
{
   static_assert(is_valid_dot_expr_v<LHS, RHS>,
      "DotExpr requires vector-shaped operands (Vector · Vector → Scalar). "
      "Use operator* for scalar multiplication, matrix-vector product, or matrix-matrix product; "
      "use inner() for matrix inner products.");

   LHS lhs;
   RHS rhs;

   DotExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template < typename... Args >
   GENDIL_HOST_DEVICE
   auto operator()( Args&&... args ) const
   {
      return Dot( lhs(std::forward<Args>(args)...), rhs(std::forward<Args>(args)...) );
   }
};

template < FieldExpr LHS, StaticString Name, FieldShape Shape >
struct DotExpr< LHS, GradientExpr< TestSpace<Name, Shape> > > : FieldBase
{
   LHS lhs;
   using RHS = GradientExpr< TestSpace<Name, Shape> >;
   RHS rhs;

   DotExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const ElementContext & element_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      auto grad_q = lhs(kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields);
      ApplyMappingTranspose(quad_pt_context.inv_J_mesh, grad_q);
      return grad_q;
   }
};

template < FieldExpr RHS, StaticString Name, FieldShape Shape >
struct DotExpr< GradientExpr< TestSpace<Name, Shape> >, RHS > : FieldBase
{
   using LHS = GradientExpr< TestSpace<Name, Shape> >;
   LHS lhs;
   RHS rhs;

   DotExpr(const LHS& lhs_, const RHS& rhs_)
      : lhs(lhs_), rhs(rhs_)
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename ElementContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const ElementContext & element_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      auto grad_q = rhs(kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields);
      ApplyMappingTranspose(quad_pt_context.inv_J_mesh, grad_q);
      return grad_q;
   }
};

template < FieldExpr LHS, FieldExpr RHS >
std::ostream& operator<<(std::ostream& os, const DotExpr<LHS, RHS>& dot)
{
   return os << "(" << dot.lhs << "." << dot.rhs << ")";
}

template < FieldExpr LHS, FieldExpr RHS >
auto dot(const LHS& lhs, const RHS& rhs)
{
   return DotExpr<LHS, RHS>(lhs, rhs);
}

} // namespace gendil
