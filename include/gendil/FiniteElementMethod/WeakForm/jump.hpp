// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"

namespace gendil
{

template < FieldExpr Expr>
struct JumpExpr : FieldBase
{
   Expr expr;

   GENDIL_HOST_DEVICE
   JumpExpr(const Expr& expr_)
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
      if constexpr ( is_side_evaluable_v<Expr> )
      {
         return minus(expr)(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields
         ) - plus(expr)(
            kernel_context,
            weak_form_context,
            operator_context,
            element_context,
            quad_pt_context,
            fields
         );
      }
      else
      {
         return expr( kernel_context, weak_form_context, operator_context, element_context, quad_pt_context, fields.minus_fields );
      }
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const JumpExpr<Expr>& jump)
{
   return os << "[[ " << jump.expr << " ]]";
}

template < FieldExpr Expr >
GENDIL_HOST_DEVICE
auto jump(const Expr& expr)
{
   return JumpExpr<Expr>(expr);
}

} // namespace gendil
