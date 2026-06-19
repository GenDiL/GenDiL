// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

template < FieldExpr Expr >
struct MinusTraceExpr : FieldBase
{
   Expr expr;

   MinusTraceExpr(const Expr& expr_)
      : expr(expr_)
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename FaceContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const FaceContext& face_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields) const
   {
      static_assert(
         requires { quad_pt_context.MinusSide(); } &&
         requires { fields.minus_fields; },
         "minus(expr) is an interior-facet trace operator. Boundary and cell "
         "contexts do not provide a canonical minus/plus trace pair.");

      return expr(
         kernel_context,
         weak_form_context,
         operator_context,
         face_context,
         quad_pt_context.MinusSide(),
         fields.minus_fields);
   }
};

template < FieldExpr Expr >
struct PlusTraceExpr : FieldBase
{
   Expr expr;

   PlusTraceExpr(const Expr& expr_)
      : expr(expr_)
   {}

   template <
      typename KernelContext,
      typename WeakFormContext,
      typename OperatorContext,
      typename FaceContext,
      typename QuadPtContext,
      typename Fields >
   GENDIL_HOST_DEVICE
   auto operator()(
      const KernelContext& kernel_context,
      const WeakFormContext& weak_form_context,
      const OperatorContext& operator_context,
      const FaceContext& face_context,
      const QuadPtContext& quad_pt_context,
      const Fields& fields) const
   {
      static_assert(
         requires { quad_pt_context.PlusSide(); } &&
         requires { fields.plus_fields; },
         "plus(expr) is an interior-facet trace operator. Boundary and cell "
         "contexts do not provide a canonical minus/plus trace pair.");

      return expr(
         kernel_context,
         weak_form_context,
         operator_context,
         face_context,
         quad_pt_context.PlusSide(),
         fields.plus_fields);
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const MinusTraceExpr<Expr>& expr)
{
   return os << "minus(" << expr.expr << ")";
}

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const PlusTraceExpr<Expr>& expr)
{
   return os << "plus(" << expr.expr << ")";
}

template < FieldExpr Expr >
auto minus(const Expr& expr)
{
   return MinusTraceExpr<Expr>(expr);
}

template < FieldExpr Expr >
auto plus(const Expr& expr)
{
   return PlusTraceExpr<Expr>(expr);
}

} // namespace gendil
