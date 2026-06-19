// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"

namespace gendil
{

template < FieldExpr Expr >
struct NegExpr : FieldBase
{
   Expr expr;

   NegExpr(const Expr& expr_)
      : expr(expr_)
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
      return -expr( kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields );
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const NegExpr<Expr>& neg)
{
   return os << "-(" << neg.expr << ")";
}

template < FieldExpr Expr >
auto MakeNegExpr(const Expr& expr)
{
   return NegExpr<Expr>(expr);
}

template < FieldExpr Expr >
auto MakeNegExpr(const NegExpr<Expr>& neg)
{
   return neg.expr;
}

template < FieldExpr Expr >
auto operator-(const Expr& expr)
{
   return MakeNegExpr<Expr>(expr);
}

} // namespace gendil
