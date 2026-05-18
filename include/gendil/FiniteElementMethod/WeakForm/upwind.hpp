// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformtraits.hpp"

namespace gendil
{

template < FieldExpr AdvExpr, FieldExpr Expr >
struct UpwindExpr : FieldBase
{
   AdvExpr adv_expr;
   Expr expr;

   UpwindExpr(const AdvExpr& adv_expr_, const Expr& expr_)
      : adv_expr(adv_expr_), expr(expr_)
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
      const KernelContext & kernel_context,
      const WeakFormContext & weak_form_context,
      const OperatorContext & operator_context,
      const FaceContext & face_context,
      const QuadPtContext & quad_pt_context,
      const Fields & fields ) const
   {
      static_assert( facet_role<Expr>::is_trial, "The advection expression in upwind() must be a trial facet expression." );
      static_assert( !need_trial_grads_v<Expr>, "UpwindExpr: trial gradients on facets not supported. ");

      // Check if quad_pt_context has physical_normal field (FacetQuadraturePointContext)
      if constexpr (requires { quad_pt_context.physical_normal; })
      {
         auto adv_value = adv_expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields );
         const auto & physical_normal = quad_pt_context.physical_normal;
         
         Real dot = Dot( adv_value, physical_normal );
         
         Real result = 0.0;
         if( dot >= 0 )
         {
            result = dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.minus_fields );
         }
         else
         {
            result = dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.plus_fields );
         }

         return result;
      }
      else
      {
         auto adv_value = adv_expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields );
         
         const auto reference_normal = GetReferenceNormal( face_context );
         const auto physical_normal = ComputePhysicalNormal( quad_pt_context.inv_J_mesh, reference_normal );
         
         Real dot = Dot( adv_value, physical_normal );
         
         Real result = 0.0;
         if( dot >= 0 )
         {
            result = dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.minus_fields );
         }
         else
         {
            result = dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.plus_fields );
         }

         return result;
      }
   }
};

template < FieldExpr AdvExpr, FieldExpr Expr >
std::ostream& operator<<(std::ostream& os, const UpwindExpr<AdvExpr, Expr>& Upwind)
{
   return os << "upwind(" << Upwind.adv_expr << ", " << Upwind.expr << ")";
}

template < FieldExpr AdvExpr, FieldExpr Expr >
auto upwind(const AdvExpr& adv_expr, const Expr& expr)
{
   return UpwindExpr<AdvExpr, Expr>(adv_expr, expr);
}

} // namespace gendil
