// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"

namespace gendil
{

struct ScaleExpr : FieldBase
{
   Real scale_factor;

   GENDIL_HOST_DEVICE
   ScaleExpr(Real scale_factor_)
      : scale_factor(scale_factor_)
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
      return scale_factor;
   }
};

std::ostream& operator<<(std::ostream& os, const ScaleExpr& Scale)
{
   return os << Scale.scale_factor;
}

template < FieldExpr Expr >
GENDIL_HOST_DEVICE
auto operator*(const Real scale_factor, const Expr& expr)
{
   return ScaleExpr(scale_factor) * expr;
}

} // namespace gendil
