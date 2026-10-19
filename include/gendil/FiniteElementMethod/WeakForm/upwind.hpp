// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/accessors.hpp"
#include "gendil/Algebra/staticvectorops.hpp"
#include "gendil/Utilities/MathHelperFunctions/dot.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/fielddependencies.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/zero.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/GenericOperator/Context/quadraturepointcontexttraits.hpp"

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
      static_assert(!is_physical_boundary_v<QuadPtContext>,
         "upwind(beta, expr) requires two interior traces; "
         "physical-boundary inflow data must be specified explicitly.");
      static_assert(is_side_evaluable_v<Expr>,
         "The transported expression in upwind() must be test-free.");
      // Keep direct trial-gradient admission separate from coefficient inputs;
      // merging their masks by name would reject valid gradient callbacks.
      static_assert([]<class... Reads>(type_list<Reads...>) consteval {
         return ((!has_provenance(Reads::provenance, NamedFieldProvenance::ActiveTrial) ||
                  !need_gradients(Reads::mask)) && ...);
      }(typename unqualified_side_dependent_named_field_requirements<Expr>::type{}),
         "UpwindExpr: trial gradients on facets not supported. ");

      // Check if quad_pt_context has physical_normal field (FacetQuadraturePointContext)
      if constexpr (requires { quad_pt_context.physical_normal; })
      {
         auto adv_value = adv_expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields );
         const auto & physical_normal = quad_pt_context.physical_normal;
         
         auto dot = Dot( adv_value, physical_normal );

         if( dot >= 0 )
         {
            if constexpr (requires { quad_pt_context.MinusSide(); })
               return dot * expr(kernel_context, weak_form_context, operator_context,
                  face_context, quad_pt_context.MinusSide(), fields.minus_fields);
            else
               return dot * expr(kernel_context, weak_form_context, operator_context,
                  face_context, quad_pt_context, fields.minus_fields);
         }
         else
         {
            if constexpr (requires { quad_pt_context.PlusSide(); })
               return dot * expr(kernel_context, weak_form_context, operator_context,
                  face_context, quad_pt_context.PlusSide(), fields.plus_fields);
            else
               return dot * expr(kernel_context, weak_form_context, operator_context,
                  face_context, quad_pt_context, fields.plus_fields);
         }
      }
      else
      {
         auto adv_value = adv_expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields );
         
         const auto reference_normal = GetReferenceNormal( face_context );
         const auto physical_normal = ComputePhysicalNormal( quad_pt_context.inv_J_mesh, reference_normal );
         
         auto dot = Dot( adv_value, physical_normal );

         if( dot >= 0 )
         {
            return dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.minus_fields );
         }
         else
         {
            return dot * expr( kernel_context, weak_form_context, operator_context, face_context, quad_pt_context, fields.plus_fields );
         }
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

namespace weak_form_detail
{
template<class Value>
consteval auto BoundaryUpwindScalarType()
{
   using T = std::remove_cvref_t<Value>;
   if constexpr (static_num_rows_v<T> != std::dynamic_extent &&
                 static_num_cols_v<T> != std::dynamic_extent)
      return std::type_identity<typename vector_element_type<
         std::remove_cvref_t<decltype(GetValue(std::declval<const T&>(), 0, 0))>>::type>{};
   else if constexpr (static_extent_v<T> != std::dynamic_extent)
      return std::type_identity<typename vector_element_type<
         std::remove_cvref_t<decltype(GetValue(std::declval<const T&>(), 0))>>::type>{};
   else
      return std::type_identity<typename vector_element_type<T>::type>{};
}

template<class Speed, class InteriorValue, class ExteriorValue>
consteval auto BoundaryUpwindResultType()
{
   using I = std::remove_cvref_t<InteriorValue>;
   using E = std::remove_cvref_t<ExteriorValue>;
   static_assert(!std::is_same_v<I, Empty> || !std::is_same_v<E, Empty>,
      "A boundary upwind expression must retain at least one state.");
   // Reuse the live type when a branch vanished; no missing state is evaluated.
   using Inside = std::conditional_t<std::is_same_v<I, Empty>, E, I>;
   using Outside = std::conditional_t<std::is_same_v<E, Empty>, Inside, E>;
   using InsideScalar = typename decltype(BoundaryUpwindScalarType<Inside>())::type;
   using OutsideScalar = typename decltype(BoundaryUpwindScalarType<Outside>())::type;
   using Scalar = std::common_type_t<
      decltype(std::declval<Speed>() * std::declval<InsideScalar>()),
      decltype(std::declval<Speed>() * std::declval<OutsideScalar>())>;

   if constexpr (static_num_rows_v<Inside> != std::dynamic_extent &&
                 static_num_cols_v<Inside> != std::dynamic_extent)
   {
      static_assert(static_num_rows_v<Inside> == static_num_rows_v<Outside> &&
                    static_num_cols_v<Inside> == static_num_cols_v<Outside>,
         "Boundary upwind states must have matching matrix dimensions.");
      return std::type_identity<SerialRecursiveArray<Scalar,
         static_num_rows_v<Inside>, static_num_cols_v<Inside>>>{};
   }
   else if constexpr (static_extent_v<Inside> != std::dynamic_extent)
   {
      static_assert(static_num_rows_v<Outside> == std::dynamic_extent &&
                    static_num_cols_v<Outside> == std::dynamic_extent &&
                    static_extent_v<Inside> == static_extent_v<Outside>,
         "Boundary upwind states must have matching vector dimensions.");
      return std::type_identity<SerialRecursiveArray<Scalar, static_extent_v<Inside>>>{};
   }
   else
   {
      static_assert(static_num_rows_v<Outside> == std::dynamic_extent &&
                    static_num_cols_v<Outside> == std::dynamic_extent &&
                    static_extent_v<Outside> == std::dynamic_extent,
         "Boundary upwind states must have matching scalar shapes.");
      return std::type_identity<Scalar>{};
   }
}

template<class Result, class Speed, class Value>
GENDIL_HOST_DEVICE Result ScaleBoundaryUpwindState(const Speed& speed, const Value& value)
{
   if constexpr (static_num_rows_v<Result> != std::dynamic_extent &&
                 static_num_cols_v<Result> != std::dynamic_extent)
   {
      Result result;
      for (Integer i = 0; i < static_num_rows_v<Result>; ++i)
         for (Integer j = 0; j < static_num_cols_v<Result>; ++j)
            result(i, j) = speed * GetValue(value, i, j);
      return result;
   }
   else if constexpr (static_extent_v<Result> != std::dynamic_extent)
   {
      Result result;
      for (Integer i = 0; i < static_extent_v<Result>; ++i)
         result(i) = speed * GetValue(value, i);
      return result;
   }
   else
      return static_cast<Result>(speed * value);
}
} // namespace weak_form_detail

/**
 * @brief Physical-boundary upwind flux with an explicit exterior state.
 *
 * With outward normal n and a = beta.n, the flux is a*u on outflow and a*g
 * on inflow. All fields are evaluated on the adjacent cell; g is an expression
 * supplied by the caller, not a ghost-cell lookup. Velocity is evaluated once
 * and only the selected state expression is evaluated. Zero normal speed
 * evaluates neither state.
 *
 * Affine extraction may replace one state with an empty zero marker. That
 * state retains its rank but no expression, captures, or field dependencies.
 */
template < FieldExpr AdvExpr, typename InteriorExpr, typename ExteriorExpr >
struct BoundaryUpwindExpr : FieldBase
{
   static_assert(FieldExpr<InteriorExpr> || weak_form_traits::is_boundary_upwind_zero_v<InteriorExpr>);
   static_assert(FieldExpr<ExteriorExpr> || weak_form_traits::is_boundary_upwind_zero_v<ExteriorExpr>);
   static_assert(FieldExpr<InteriorExpr> || FieldExpr<ExteriorExpr>);

   AdvExpr adv_expr;
   [[no_unique_address]] InteriorExpr interior_expr;
   [[no_unique_address]] ExteriorExpr exterior_expr;

   GENDIL_HOST_DEVICE
   BoundaryUpwindExpr(
      const AdvExpr & adv_expr,
      const InteriorExpr & interior_expr,
      const ExteriorExpr & exterior_expr )
      : adv_expr(adv_expr), interior_expr(interior_expr), exterior_expr(exterior_expr)
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
      static_assert(is_physical_boundary_v<QuadPtContext>,
         "upwind(beta, u, g) requires a physical boundary and its outward normal.");
      static_assert(is_test_free_v<BoundaryUpwindExpr>,
         "upwind(beta, u, g) requires test-free velocity and state expressions.");

      if constexpr ( is_physical_boundary_v<QuadPtContext> &&
                     is_test_free_v<BoundaryUpwindExpr> )
      {
         const auto velocity = adv_expr(
            kernel_context, weak_form_context, operator_context,
            face_context, quad_pt_context, fields);
         const auto & normal = quad_pt_context.physical_normal;
         constexpr auto Dim = static_extent_v<std::remove_cvref_t<decltype(normal)>>;
         static_assert(Dim != std::dynamic_extent &&
                       static_extent_v<decltype(velocity)> == Dim,
            "Boundary upwind velocity must match the physical domain dimension.");
         Real normal_speed = 0;
         for (Integer d = 0; d < Dim; ++d)
         {
            normal_speed += GetValue(velocity, d) * GetValue(normal, d);
         }

         // Result storage is shared by both branches, including vector states
         // represented by different fixed-size containers.
         auto evaluate = [&] (const auto & state)
         {
            // NVCC needs unconditional references for these captures.
            (void)kernel_context;
            (void)weak_form_context;
            (void)operator_context;
            (void)face_context;
            (void)quad_pt_context;
            (void)fields;
            if constexpr ( FieldExpr<std::remove_cvref_t<decltype(state)>> )
            {
               return state(kernel_context, weak_form_context, operator_context,
                            face_context, quad_pt_context, fields);
            }
            else
            {
               return Empty{};
            }
         };
         using InteriorValue = decltype(evaluate(interior_expr));
         using ExteriorValue = decltype(evaluate(exterior_expr));
         using Result = typename decltype(weak_form_detail::BoundaryUpwindResultType<
            decltype(normal_speed), InteriorValue, ExteriorValue>())::type;

         if ( normal_speed == 0 ) { return Result{}; }
         if ( normal_speed > 0 )
         {
            if constexpr ( FieldExpr<InteriorExpr> )
            {
               return weak_form_detail::ScaleBoundaryUpwindState<Result>(
                  normal_speed, evaluate(interior_expr));
            }
         }
         else
         {
            if constexpr ( FieldExpr<ExteriorExpr> )
            {
               return weak_form_detail::ScaleBoundaryUpwindState<Result>(
                  normal_speed, evaluate(exterior_expr));
            }
         }
         return Result{};
      }
   }
};

template < FieldExpr AdvExpr, typename InteriorExpr, typename ExteriorExpr >
std::ostream & operator<<(
   std::ostream & os,
   const BoundaryUpwindExpr<AdvExpr, InteriorExpr, ExteriorExpr> & expr )
{
   os << "upwind(" << expr.adv_expr << ", ";
   if constexpr ( FieldExpr<InteriorExpr> ) { os << expr.interior_expr; }
   else { os << "0"; }
   os << ", ";
   if constexpr ( FieldExpr<ExteriorExpr> ) { os << expr.exterior_expr; }
   else { os << "0"; }
   return os << ")";
}

/**
 * @brief Return the outward normal upwind flux on a physical boundary.
 *
 * Use integrate(boundary, upwind(beta, u, g) * v), where u is the interior
 * state and g supplies the inflow state. Scalar and vector states may use
 * different finite-element spaces but must have matching value shapes.
 * Both can be bound fields, active fields, or test-free expressions. An
 * active-dependent velocity is valid for residual evaluation, but cannot
 * be split by lhs/rhs. The two-argument overload remains an interior trace.
 */
template < FieldExpr AdvExpr, FieldExpr InteriorExpr, FieldExpr ExteriorExpr >
GENDIL_HOST_DEVICE
auto upwind(
   const AdvExpr & adv_expr,
   const InteriorExpr & interior_expr,
   const ExteriorExpr & exterior_expr )
{
   return BoundaryUpwindExpr<AdvExpr, InteriorExpr, ExteriorExpr>{
      adv_expr, interior_expr, exterior_expr};
}

} // namespace gendil
