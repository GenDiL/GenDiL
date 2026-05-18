// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/add.hpp"
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointIO/readquadraturelocalvalues.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/applymapping.hpp"

namespace gendil
{

template < FieldExpr Expr >
struct GradientExpr : FieldBase
{
   Expr expr;

   GradientExpr(const Expr& expr_)
      : expr(expr_)
   {}
};

template < StaticString Name, FieldShape Shape >
struct GradientExpr< TrialSpace<Name, Shape> > : FieldBase
{
   using expr_type = TrialSpace<Name, Shape>;
   expr_type expr;

   GradientExpr(const expr_type& expr_)
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
      auto grad_q = ReadQuadratureLocalGradients(kernel_context, quad_pt_context.quad_index, fields.template get<NameTag<Name>>().gradients);
      ApplyMapping(quad_pt_context.inv_J_mesh, grad_q);
      return grad_q;
   }
};

template < StaticString Name, FieldShape Shape >
struct GradientExpr< TestSpace<Name, Shape> > : FieldBase
{
   using expr_type = TestSpace<Name, Shape>;
   expr_type expr;

   GradientExpr(const expr_type& expr_)
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
      static_assert( dependent_false_v<expr_type>,
         "Gradient of test space functions is not supported. Test function gradients are not stored at quadrature points." );
      return std::array{quad_pt_context.inv_J_mesh};
   }
};

template < FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const GradientExpr<Expr>& grad)
{
   return os << "grad(" << grad.expr << ")";
}

template < FieldExpr LHSExpr, FieldExpr RHSExpr >
auto grad(const AddFieldExpr<LHSExpr, RHSExpr>& expr)
{
   return grad(expr.lhs) + grad(expr.rhs);
}

template < FieldExpr LHSExpr, FieldExpr RHSExpr >
auto grad(const MultFieldExpr<LHSExpr, RHSExpr>& expr)
{
   return (grad(expr.lhs) * expr.rhs) + (expr.lhs * grad(expr.rhs));
}

template < FieldExpr Expr >
auto grad(const Expr& expr)
{
   return GradientExpr<Expr>(expr);
}

} // namespace gendil
