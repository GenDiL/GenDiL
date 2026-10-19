// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/staticmap.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"
#include "gendil/FiniteElementMethod/WeakForm/gradient.hpp"
#include "gendil/FiniteElementMethod/WeakForm/curl.hpp"
#include "gendil/FiniteElementMethod/WeakForm/divergence.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/WeakFormTraits/testlineartraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dot.hpp"
#include "gendil/FiniteElementMethod/WeakForm/inner.hpp"
#include "gendil/FiniteElementMethod/WeakForm/mult.hpp"
#include "gendil/FiniteElementMethod/WeakForm/sumfieldexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/transpose.hpp"
#include "gendil/FiniteElementMethod/WeakForm/symmetry.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/outer.hpp"
#include "gendil/FiniteElementMethod/WeakForm/neg.hpp"
#include "gendil/FiniteElementMethod/WeakForm/normal.hpp"
#include "gendil/FiniteElementMethod/WeakForm/facetprojection.hpp"
#include "gendil/FiniteElementMethod/WeakForm/scale.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/jump.hpp"
#include "gendil/FiniteElementMethod/WeakForm/average.hpp"
#include "gendil/FiniteElementMethod/WeakForm/upwind.hpp"

namespace gendil
{

// =============================================================================
// Design NOTE: Seed-Based Pullback with Compile-Time Channels
// =============================================================================
//
// This header implements the pullback mechanism for weak-form expressions.
//
// **Pullback API:**
//    pullback(expr, seed) → PullbackResult<Channels...>
//
// Where:
//  - expr is a test-linear weak-form expression
//  - seed is the upstream coefficient/cotangent multiplying expr
//  - Result is a compile-time set of channel contributions
//
// **For a complete scalar integrand:**
//    lower_integrand(expr) = pullback(expr, ScalarIdentity{})
//
// **Coordinate frame convention:**
//    GradientChannel stores physical-gradient coefficients produced by
//    weak-form pullback. Mapping transpose from physical to reference
//    coordinates is not applied in this skeleton. It will be handled later
//    by the channel-application stage that writes GradientChannel
//    contributions into the quadrature data consumed by
//    ApplyGradientTestFunctions.
//
// =============================================================================

// =============================================================================
// Channel Application Design
// =============================================================================

// =============================================================================
// Channel Tags
// =============================================================================

/**
 * @brief Value channel: contributions multiplying test function values.
 *
 * For scalar test space TestSpace<Name>: seed is Scalar
 * For vector test space VectorTestSpace<Name>: seed is Vector
 */
struct ValueChannel {};

/**
 * @brief Gradient channel: contributions multiplying test function gradients.
 *
 * For scalar test space gradient grad(TestSpace<Name>): seed is Vector
 * For vector test space gradient grad(VectorTestSpace<Name>): seed is Matrix
 *
 * COORDINATE FRAME: Stores physical-gradient coefficients.
 * Mapping transpose to reference frame is deferred to channel-application stage.
 */
struct GradientChannel {};

/**
 * @brief Canonical global-interior value channel for the minus test row.
 */
struct ValueMinusChannel {};

/**
 * @brief Canonical global-interior value channel for the plus test row.
 */
struct ValuePlusChannel {};

/**
 * @brief Canonical global-interior gradient channel for the minus test row.
 */
struct GradientMinusChannel {};

/**
 * @brief Canonical global-interior gradient channel for the plus test row.
 */
struct GradientPlusChannel {};

template<class Channels>
inline constexpr bool contains_ordinary_test_channel_v =
   std::remove_cvref_t<Channels>::template Contains<ValueChannel>() ||
   std::remove_cvref_t<Channels>::template Contains<GradientChannel>();

template<class Channels>
inline constexpr bool contains_side_qualified_interior_test_channel_v =
   std::remove_cvref_t<Channels>::template Contains<ValueMinusChannel>() ||
   std::remove_cvref_t<Channels>::template Contains<ValuePlusChannel>() ||
   std::remove_cvref_t<Channels>::template Contains<GradientMinusChannel>() ||
   std::remove_cvref_t<Channels>::template Contains<GradientPlusChannel>();

// =============================================================================
// PullbackResult: StaticMap-Based Channel Storage
// =============================================================================

/**
 * @brief Compile-time map from channel tags to seed expressions.
 *
 * PullbackResult is a thin wrapper around StaticMap that stores channel
 * contributions with compile-time presence/absence guarantees.
 *
 * Usage:
 *    auto result = pullback(v, c);
 *    static_assert(decltype(result)::Contains<ValueChannel>());
 *    auto& seed = result.template Get<ValueChannel>();
 *
 * @tparam Entries Pack of Entry<ChannelTag, SeedExpr> types
 */
template<class... Entries>
struct PullbackResult : StaticMap<Entries...>
{
   using StaticMap<Entries...>::StaticMap;

   // Constructor from StaticMap (for MakeStaticMap compatibility)
   GENDIL_HOST_DEVICE
   constexpr PullbackResult(StaticMap<Entries...>&& map)
      : StaticMap<Entries...>(std::move(map))
   {}

   GENDIL_HOST_DEVICE
   constexpr PullbackResult(const StaticMap<Entries...>& map)
      : StaticMap<Entries...>(map)
   {}
};

// Form lowering records where its channels may be delivered. This adds no
// row storage: ordinary entities still carry only current-row coefficients,
// while a canonical interior facet retains its minus/plus channel keys.
enum class PullbackEntity
{
   Cell,
   BoundaryFacet,
   LocalInteriorFacet,
   GlobalInteriorFacet
};

enum class PullbackRows { Current, MinusPlus };

namespace pullback_detail
{

template<PullbackRows Rows, class... Entries>
consteval bool ChannelsMatchRows(const std::tuple<Entries...>*)
{
   if constexpr (Rows == PullbackRows::Current)
   {
      return ((std::is_same_v<typename Entries::key_type, ValueChannel> ||
               std::is_same_v<typename Entries::key_type, GradientChannel>) && ...);
   }
   else
   {
      return ((std::is_same_v<typename Entries::key_type, ValueMinusChannel> ||
               std::is_same_v<typename Entries::key_type, ValuePlusChannel> ||
               std::is_same_v<typename Entries::key_type, GradientMinusChannel> ||
               std::is_same_v<typename Entries::key_type, GradientPlusChannel>) && ...);
   }
}

} // namespace pullback_detail

template<class Channels, PullbackRows Rows>
inline constexpr bool pullback_channels_match_rows_v =
   pullback_detail::ChannelsMatchRows<Rows>(
      static_cast<decltype(std::declval<std::remove_cvref_t<Channels>>().entries)*>(nullptr));

template<PullbackEntity Entity, class Channels>
struct EntityPullbackResult : Channels
{
   using channel_map_type = Channels;
   static constexpr auto entity = Entity;
   static constexpr auto rows = Entity == PullbackEntity::GlobalInteriorFacet
      ? PullbackRows::MinusPlus : PullbackRows::Current;
   static constexpr int output_row_count = rows == PullbackRows::Current ? 1 : 2;

   static_assert(pullback_channels_match_rows_v<Channels, rows>,
      "Pullback channels do not match the entity's output rows.");

   GENDIL_HOST_DEVICE
   constexpr explicit EntityPullbackResult(Channels channels)
      : Channels(std::move(channels))
   {}
};

template<PullbackEntity Entity, class Channels>
GENDIL_HOST_DEVICE
constexpr auto MakeEntityPullbackResult(Channels channels)
{
   return EntityPullbackResult<Entity, Channels>(std::move(channels));
}

// =============================================================================
// Seed Shape Validation Helper
// =============================================================================

/**
 * @brief SFINAE-safe validation that seed shape matches expression shape.
 *
 * pullback(expr, seed) requires:
 *    field_shape_v<Seed> == field_shape_v<Expr>
 *
 * This two-level pattern avoids hard errors on types without has_field_shape_v.
 */
template<class Expr, class Seed, bool BothHaveShape =
   has_field_shape_v<Expr> && has_field_shape_v<Seed>>
struct is_seed_shape_compatible_impl : std::false_type {};

template<class Expr, class Seed>
struct is_seed_shape_compatible_impl<Expr, Seed, true>
   : std::bool_constant<
        field_shape_v<std::remove_cvref_t<Expr>> ==
        field_shape_v<std::remove_cvref_t<Seed>>>
{};

template<class Expr, class Seed>
inline constexpr bool is_seed_shape_compatible_v =
   is_seed_shape_compatible_impl<Expr, Seed>::value;

namespace pullback_detail
{

// Name-only bound fields deliberately have no context-free shape. Scalar
// multiplication retains the seed's known shape while the selected context
// validates the expression's deferred shape obligations. Check the evaluated
// factor too, so using the lowerer directly cannot turn a vector into a scalar.
template<class Seed, class Multiplier>
struct ScalarMultiplierSeed : FieldBase
{
   Seed seed;
   Multiplier multiplier;

   GENDIL_HOST_DEVICE
   ScalarMultiplierSeed(const Seed& s, const Multiplier& m)
      : seed(s), multiplier(m) {}

   template<class... Args>
   GENDIL_HOST_DEVICE
   auto operator()(const Args&... args) const
   {
      auto factor = multiplier(args...);
      static_assert(std::is_convertible_v<decltype(factor), Real>,
         "Multiplication pullback requires a scalar test-free multiplier.");
      return seed(args...) * factor;
   }
};

// The adjoint of trace(grad(v)) is a physical diagonal matrix. Only the
// coordinate type supplies its extent; no coordinate values are evaluated.
template<class Seed>
struct DivergenceSeed : FieldBase
{
   Seed seed;

   GENDIL_HOST_DEVICE
   explicit DivergenceSeed(const Seed& value) : seed(value) {}

   template<class KernelContext, class WeakFormContext, class OperatorContext,
            class ElementContext, class QuadPtContext, class Fields>
   GENDIL_HOST_DEVICE
   auto operator()(const KernelContext& kernel_context, const WeakFormContext& weak_form_context,
                   const OperatorContext& operator_context, const ElementContext& element_context,
                   const QuadPtContext& quad_pt_context, const Fields& fields) const
   {
      constexpr auto Dim = std::tuple_size_v<std::remove_cvref_t<decltype(quad_pt_context.X)>>;
      const auto value = seed(kernel_context, weak_form_context, operator_context,
                              element_context, quad_pt_context, fields);
      static_assert(std::is_convertible_v<decltype(value), Real>,
         "Divergence pullback requires a scalar seed.");
      SerialRecursiveArray<Real, Dim, Dim> result{};
      for (size_t d = 0; d < Dim; ++d) { result(d, d) = value; }
      return result;
   }
};

// Curl's adjoint acts on the physical gradient. The channel writer applies
// the selected cell/side mapping afterwards, exactly as for grad and div.
template<class Seed>
struct CurlSeed : FieldBase
{
   Seed seed;

   GENDIL_HOST_DEVICE
   explicit CurlSeed(const Seed& value) : seed(value) {}

   template<class... Args>
   GENDIL_HOST_DEVICE
   auto operator()(const Args&... args) const
   {
      const auto value = seed(args...);
      static_assert(static_extent_v<decltype(value)> == 3,
         "Curl pullback requires a three-component seed.");
      const Real x = GetValue(value, 0);
      const Real y = GetValue(value, 1);
      const Real z = GetValue(value, 2);
      SerialRecursiveArray<Real, 3, 3> result{};
      result(0, 1) = -z; result(0, 2) = y;
      result(1, 0) = z;  result(1, 2) = -x;
      result(2, 0) = -y; result(2, 1) = x;
      return result;
   }
};

} // namespace pullback_detail

template<class Seed>
struct field_shape_impl<pullback_detail::CurlSeed<Seed>>
{
   static constexpr auto value = FieldShape::Matrix;
};

template<class Seed>
struct test_linearity<pullback_detail::CurlSeed<Seed>>
{
   static_assert(is_test_free_v<Seed>);
   static constexpr auto value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<class Seed>
struct field_shape_impl<pullback_detail::DivergenceSeed<Seed>>
{
   static constexpr auto value = FieldShape::Matrix;
};

template<class Seed>
struct test_linearity<pullback_detail::DivergenceSeed<Seed>>
{
   static_assert(is_test_free_v<Seed>);
   static constexpr auto value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

template<class Seed, class Multiplier>
struct field_shape_impl<pullback_detail::ScalarMultiplierSeed<Seed, Multiplier>>
{
   static constexpr auto value = field_shape_v<Seed>;
};

template<class Seed, class Multiplier>
struct test_linearity<pullback_detail::ScalarMultiplierSeed<Seed, Multiplier>>
{
   static_assert(is_test_free_v<Seed> && is_test_free_v<Multiplier>);
   static constexpr auto value = TestLinearity::TestFree;
   static constexpr auto test_name = StaticString{""};
};

// =============================================================================
// Primitive Pullback Rules
// =============================================================================

/**
 * @brief Pullback for test space value (scalar or vector).
 *
 * pullback(v, seed) → ValueChannel with seed
 *
 * For scalar test space (Shape=Scalar): seed must be Scalar
 * For vector test space (Shape=Vector): seed must be Vector
 *
 * @tparam Name Test space name
 * @tparam Shape Field shape (Scalar or Vector)
 * @tparam Seed Seed expression type (must match Shape)
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const TestSpace<Name, Shape>& /*v*/, const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<ValueChannel>(seed))};
}

/**
 * @brief Pullback for test space gradient (scalar or vector).
 *
 * pullback(grad(v), seed) → GradientChannel with seed (physical frame)
 *
 * For scalar test space gradient (grad(TestSpace<Scalar>)): seed must be Vector
 * For vector test space gradient (grad(TestSpace<Vector>)): seed must be Matrix
 *
 * @tparam Name Test space name
 * @tparam Shape Field shape (Scalar or Vector)
 * @tparam Seed Seed expression type (must match grad(Shape))
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const GradientExpr<TestSpace<Name, Shape>>& /*grad_v*/, const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<GradientChannel>(seed))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const DivergenceExpr<TestSpace<Name, FieldShape::Vector>>&, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<DivergenceExpr<VectorTestSpace<Name>>, Seed>,
      "Divergence pullback requires a scalar seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientChannel>(
      pullback_detail::DivergenceSeed<Seed>{seed}))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const CurlExpr<TestSpace<Name, FieldShape::Vector>>&, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<CurlExpr<VectorTestSpace<Name>>, Seed>,
      "Curl pullback requires a vector seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientChannel>(
      pullback_detail::CurlSeed<Seed>{seed}))};
}

template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const MinusTraceExpr<E>&, const Seed&)
{
   static_assert(
      dependent_false_v<E, Seed>,
      "minus(expr) is a canonical interior-facet trace operator. The local/"
      "current-row and boundary pullback paths do not assign row-relative "
      "semantics to public minus(...); use jump/average in the local path or "
      "the canonical global-interior lowering path.");
   return PullbackResult<>{};
}

template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const PlusTraceExpr<E>&, const Seed&)
{
   static_assert(
      dependent_false_v<E, Seed>,
      "plus(expr) is a canonical interior-facet trace operator. The local/"
      "current-row and boundary pullback paths do not assign row-relative "
      "semantics to public plus(...); use jump/average in the local path or "
      "the canonical global-interior lowering path.");
   return PullbackResult<>{};
}

// =============================================================================
// Placeholder Overloads for Deferred Expression Types
// =============================================================================
//
// These overloads produce clear compile-time errors when pullback is called
// on expression types not yet implemented. The static_assert is dependent
// and fires only when pullback is actually invoked.

// Forward declarations needed for placeholder overloads
// (Already declared in dslbase.hpp with FieldExpr constraints)

/**
 * @brief Pullback for ProductExpr (algebraic product).
 *
 * Currently implemented for ProductKind::ScalarTimes and the two linear
 * ProductKind::MatVec adjoints.
 *
 * ScalarTimes adjoint rule:
 *   pullback(scalar_free * test_expr, seed)
 *      = pullback(test_expr, seed * scalar_free)
 *
 * Valid combinations:
 *   - test-linear × test-free scalar
 *   - test-free scalar × test-linear
 *
 * Invalid combinations (will fail with clear static_assert):
 *   - Both test-free → no test contribution
 *   - Both test-linear → NonlinearInTest
 *   - ProductKind::MatVec → supported when exactly one operand is test-linear
 *   - ProductKind::MatMat → requires product-specific adjoint (not implemented)
 *
 * @tparam LHS Left operand type
 * @tparam RHS Right operand type
 * @tparam Seed Seed expression type (must match ProductExpr shape)
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const ProductExpr<LHS, RHS>& expr, const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = ProductExpr<LHS, RHS>;

   // Validate seed shape
   static_assert(is_seed_shape_compatible_v<Expr, S>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   // Branch on ProductKind
   if constexpr (Expr::product_kind == ProductKind::ScalarTimes)
   {
      // ScalarTimes: at least one operand must be Scalar
      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         // LHS is test-linear, RHS is test-free
         static_assert(field_shape_v<R> == FieldShape::Scalar,
            "ScalarTimes ProductExpr pullback requires the test-free multiplier to be Scalar.");

         return pullback(expr.lhs, seed * expr.rhs);
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         // LHS is test-free, RHS is test-linear
         static_assert(field_shape_v<L> == FieldShape::Scalar,
            "ScalarTimes ProductExpr pullback requires the test-free multiplier to be Scalar.");

         return pullback(expr.rhs, seed * expr.lhs);
      }
      else
      {
         static_assert(dependent_false_v<L, R, S>,
            "ScalarTimes ProductExpr pullback requires exactly one TestLinear operand and one TestFree scalar operand. "
            "Both test-free → no test contribution; both test-linear → NonlinearInTest.");
      }
   }
   else if constexpr (Expr::product_kind == ProductKind::MatVec)
   {
      static_assert(field_shape_v<L> == FieldShape::Matrix,
         "MatVec ProductExpr pullback requires the left operand to be Matrix-shaped.");
      static_assert(field_shape_v<R> == FieldShape::Vector,
         "MatVec ProductExpr pullback requires the right operand to be Vector-shaped.");
      static_assert(field_shape_v<S> == FieldShape::Vector,
         "MatVec ProductExpr pullback requires a Vector-shaped seed.");
      static_assert(is_test_free_v<S>,
         "MatVec ProductExpr pullback requires a test-free seed.");

      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         return pullback(
            expr.lhs,
            outer(seed, expr.rhs));
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         return pullback(
            expr.rhs,
            transpose(expr.lhs) * seed);
      }
      else
      {
         static_assert(dependent_false_v<L, R, S>,
            "MatVec ProductExpr pullback requires exactly one TestLinear operand. "
            "Supported directions are test-linear matrix * test-free vector "
            "and test-free matrix * test-linear vector.");
      }
   }
   else
   {
      static_assert(dependent_false_v<L, R, S>,
         "ProductExpr pullback is implemented for ProductKind::ScalarTimes and "
         "the linear ProductKind::MatVec adjoints. "
         "MatMat and broader product-specific adjoints are not implemented.");
   }
}

// The fallback multiplication node is used when a named bound multiplier's
// shape is deferred until context selection. Do not change that field's public
// shape metadata to admit it here.
template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const MultFieldExpr<LHS, RHS>& expr, const Seed& seed)
{
   auto multiply = [&]<class Factor>(const Factor& factor)
   {
      if constexpr (has_field_shape_v<Factor>)
      {
         static_assert(field_shape_v<Factor> == FieldShape::Scalar,
            "Multiplication pullback requires a scalar test-free multiplier.");
         return seed * factor;
      }
      else
      {
         return pullback_detail::ScalarMultiplierSeed<Seed, Factor>{seed, factor};
      }
   };
   if constexpr (is_test_linear_v<LHS> && is_test_free_v<RHS>)
   {
      return pullback(expr.lhs, multiply(expr.rhs));
   }
   else if constexpr (is_test_free_v<LHS> && is_test_linear_v<RHS>)
   {
      return pullback(expr.rhs, multiply(expr.lhs));
   }
   else
   {
      static_assert(dependent_false_v<LHS, RHS, Seed>,
         "Multiplication pullback requires exactly one test-linear operand and one test-free scalar operand.");
   }
}

/**
 * @brief Pullback for DotExpr (vector dot product).
 *
 * pullback(dot(a, b), seed) where exactly one operand is test-linear.
 *
 * Bilinear adjoint rule:
 *   pullback(dot(test_expr, free_expr), seed)
 *      = pullback(test_expr, seed * free_expr)
 *
 * Valid combinations:
 *   - dot(test-linear vector, test-free vector)
 *   - dot(test-free vector, test-linear vector)
 *
 * Invalid combinations (will fail with clear static_assert):
 *   - dot(test-free, test-free) → no test contribution
 *   - dot(test-linear, test-linear) → NonlinearInTest
 *
 * @tparam LHS Left operand type
 * @tparam RHS Right operand type
 * @tparam Seed Seed expression type (must be Scalar)
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const DotExpr<LHS, RHS>& expr, const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = DotExpr<LHS, RHS>;

   // Validate seed shape (DotExpr is Scalar, so seed must be Scalar)
   static_assert(is_seed_shape_compatible_v<Expr, S>,
      "pullback(expr, seed) requires seed to have the same FieldShape as expr.");

   // Branch on test-linearity
   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      // LHS is test-linear, RHS is test-free
      return pullback(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      // LHS is test-free, RHS is test-linear
      return pullback(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(dependent_false_v<L, R, S>,
         "DotExpr pullback requires exactly one test-linear operand and one test-free operand. "
         "Both test-free → no test contribution; both test-linear → NonlinearInTest.");
   }
}

/**
 * @brief Pullback for NegExpr.
 */
template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const NegExpr<Expr>& expr, const Seed& seed)
{
   return pullback(expr.expr, -seed);
}

// Symmetric and skew-symmetric projection are self-adjoint under the real
// Frobenius pairing. Their adjoints keep the existing physical matrix seed.
template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const SymExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<SymExpr<Expr>, Seed>,
      "Symmetric projection pullback requires a matrix seed.");
   return pullback(expr.expr, sym(seed));
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const SkewExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<SkewExpr<Expr>, Seed>,
      "Skew-symmetric projection pullback requires a matrix seed.");
   return pullback(expr.expr, skew(seed));
}

// Orthogonal facet projections are self-adjoint. Project the incoming seed
// without changing the side selections of any fields it already contains.
template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const NormalProjectionExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<NormalProjectionExpr<Expr>, Seed>,
      "Normal projection pullback requires a vector seed.");
   return pullback(expr.expr, normal_projection(seed));
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const TangentialProjectionExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<TangentialProjectionExpr<Expr>, Seed>,
      "Tangential projection pullback requires a vector seed.");
   return pullback(expr.expr, tangential_projection(seed));
}

// =============================================================================
// Channel Merge Utilities
// =============================================================================

/**
 * @brief Helper to merge one entry from b into the accumulating map.
 *
 * Base case: no more entries to process.
 */
template<class...EntriesA>
GENDIL_HOST_DEVICE
constexpr auto merge_entries_impl(StaticMap<EntriesA...> acc_map)
{
   return acc_map;
}

/**
 * @brief Helper to merge one entry from b into the accumulating map.
 *
 * Recursive case: process first entry, then recurse on rest.
 */
template<class... EntriesA, class FirstEntry, class... RestEntries>
GENDIL_HOST_DEVICE
constexpr auto merge_entries_impl(
   StaticMap<EntriesA...> acc_map,
   const FirstEntry& first_entry,
   const RestEntries&... rest_entries)
{
   using Key = typename FirstEntry::key_type;
   auto new_map = MapInsertOrMerge<Key>(
      std::move(acc_map),
      first_entry.value,
      [](const auto& old_val, const auto& new_val)
      {
         return old_val + new_val;
      });

   return merge_entries_impl(std::move(new_map), rest_entries...);
}

/**
 * @brief Merge two PullbackResult maps by combining duplicate channels.
 *
 * If a channel exists in only one result: keep it.
 * If the same channel exists in both results: combine values with operator+.
 *
 * This preserves the test-free invariant and FieldShape compatibility.
 *
 * @tparam EntriesA Entries from first PullbackResult
 * @tparam EntriesB Entries from second PullbackResult
 */
template<class... EntriesA, class... EntriesB>
GENDIL_HOST_DEVICE
constexpr auto merge_two_pullback_results(
   const PullbackResult<EntriesA...>& a,
   const PullbackResult<EntriesB...>& b)
{
   // Unpack entries from b and recursively insert/merge into a's map
   auto result_map = std::apply(
      [&](const auto&... b_entries)
      {
         return merge_entries_impl(StaticMap{a.entries}, b_entries...);
      },
      b.entries);

   return PullbackResult{std::move(result_map)};
}

/**
 * @brief Variadic merge of multiple PullbackResult maps.
 *
 * Base case: single result.
 */
template<class Result>
GENDIL_HOST_DEVICE
constexpr auto merge_pullback_results(Result&& result)
{
   return std::forward<Result>(result);
}

/**
 * @brief Variadic merge: recursively merge pairwise.
 */
template<class ResultA, class ResultB, class... Rest>
GENDIL_HOST_DEVICE
constexpr auto merge_pullback_results(ResultA&& a, ResultB&& b, Rest&&... rest)
{
   auto merged = merge_two_pullback_results(
      std::forward<ResultA>(a),
      std::forward<ResultB>(b));
   return merge_pullback_results(std::move(merged), std::forward<Rest>(rest)...);
}

// =============================================================================
// SumExpr Pullback
// =============================================================================

/**
 * @brief Pullback for SumExpr (sum of field expressions).
 *
 * pullback(t1 + t2 + ..., seed) = merge_channels(
 *    pullback(t1, seed),
 *    pullback(t2, seed),
 *    ...
 * )
 *
 * The same seed is propagated to all terms.
 * Channel maps from each term are merged, combining duplicate channels with operator+.
 *
 * Examples:
 *   pullback(u*v + dot(grad(u), grad(v)), c)
 *      → ValueChannel: c*u
 *      → GradientChannel: c*grad(u)
 *
 *   pullback(u*v + mu*v, c)
 *      → ValueChannel: (c*u) + (c*mu)
 *
 * @tparam Head First term type
 * @tparam Tail Additional term types
 * @tparam Seed Seed expression type (must match SumExpr shape)
 */
template<FieldExpr Head, FieldExpr... Tail, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const SumExpr<Head, Tail...>& expr, const Seed& seed)
{
   // Each term checks its seed shape recursively. Querying the aggregate's
   // shape here would prematurely reject named bound multipliers whose shape
   // remains an obligation until the execution context selects their spaces.
   return std::apply(
      [&](const auto&... terms)
      {
         return merge_pullback_results(pullback(terms, seed)...);
      },
      expr.terms);
}

// =============================================================================
// InnerExpr Pullback
// =============================================================================
//
// InnerExpr represents Frobenius inner product for pairing operations.
//
// Pullback rule (narrow scope - exactly one test-linear operand):
//    pullback(inner(coeff_expr, test_expr), seed)
//       = pullback(test_expr, seed * coeff_expr)
//
// Valid patterns:
//    inner(A, outer(jump(V), normal)) - coeff matrix, test outer product
//    inner(outer(jump(U), normal), average(grad(V))) - test trial, test gradient
//
// NARROW SCOPE:
//    - Only handles exactly one test-linear operand
//    - Rejects both-test-dependent cases (would be nonlinear/quadratic)
//    - Uses ProductExpr (seed * coeff_expr) for seed multiplication
//

/**
 * @brief InnerExpr pullback: coeff-test pattern.
 *
 * For inner(coeff_expr, test_expr) where RHS is test-linear:
 *    Variation: coeff_expr : delta_test_expr
 *    Pullback seed for test_expr: seed * coeff_expr
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_free_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Frobenius inner: A : B where B is test-linear
   // Pullback seed for B: seed * A
   auto rhs_seed = seed * expr.lhs;
   return pullback(expr.rhs, rhs_seed);
}

/**
 * @brief InnerExpr pullback: test-coeff pattern (symmetric case).
 *
 * For inner(test_expr, coeff_expr) where LHS is test-linear:
 *    Variation: delta_test_expr : coeff_expr
 *    Pullback seed for test_expr: seed * coeff_expr
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_free_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Symmetric case: inner(test_expr, coeff_expr)
   // Pullback seed for test_expr: seed * coeff_expr
   auto lhs_seed = seed * expr.rhs;
   return pullback(expr.lhs, lhs_seed);
}

/**
 * @brief InnerExpr pullback: reject both-test-dependent case.
 *
 * If both operands are test-dependent, the expression is quadratic in test
 * function (nonlinear). Such expressions are invalid for weak forms.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const InnerExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "InnerExpr pullback: both operands test-dependent not supported. "
      "inner(test_expr1, test_expr2) is quadratic in test function (nonlinear/invalid).");
   return PullbackResult<>{};
}

// =============================================================================
// OuterExpr Pullback
// =============================================================================
//
// OuterExpr represents the outer product of two vector expressions.
//
// Pullback rule (narrow scope - test-linear LHS only):
//    pullback(outer(test_expr, test_free_vector), tensor_seed)
//       = pullback(test_expr, contracted_seed)
//
// Where:
//    contracted_seed[c] = sum_j tensor_seed[c][j] * test_free_vector[j]
//
// This contraction is implemented via ProductExpr with MatVec kind:
//    contracted_seed = tensor_seed * test_free_vector
//
// Target use case (vector SIPDG):
//    pullback(outer(jump(V), normal), A)
//       → pullback(jump(V), A * normal)
//
// NARROW SCOPE:
//    - Only implements outer(test_expr, test_free_vector)
//    - Does NOT implement outer(test_free, test_expr) unless compiler proves needed
//    - Rejects both-test-dependent cases (quadratic, invalid)
//

/**
 * @brief OuterExpr pullback: test-linear LHS, test-free RHS.
 *
 * For outer(test_expr, test_free_vector):
 *    Variation: outer(delta_test_expr, test_free_vector)
 *    Pullback seed for test_expr: contracted_seed = tensor_seed * test_free_vector
 *
 * The contraction uses ProductExpr (Matrix×Vector → Vector).
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_free_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& expr, const Seed& seed)
{
   // Contract tensor seed with RHS vector:
   // contracted_seed[c] = sum_j seed[c][j] * expr.rhs[j]
   //
   // ProductExpr with MatVec kind implements this contraction.
   // Verified: product_kind_v<Matrix, Vector> == ProductKind::MatVec.
   auto contracted_seed = seed * expr.rhs;  // Matrix×Vector → Vector
   return pullback(expr.lhs, contracted_seed);
}

/**
 * @brief OuterExpr pullback: test-free LHS, test-linear RHS (not implemented).
 *
 * For outer(test_free_vector, test_expr):
 *    Not needed for vector SIPDG (RHS is always Normal/geometry).
 *    Reserved for future extension if compiler proves necessary.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_free_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "OuterExpr pullback: test-linear RHS not yet implemented. "
      "Only outer(test_expr, test_free_vector) is supported for vector SIPDG.");
   return PullbackResult<>{};
}

/**
 * @brief OuterExpr pullback: reject both-test-dependent case.
 *
 * If both operands are test-dependent, the expression is quadratic in test
 * function (nonlinear). Such expressions are invalid for weak forms.
 */
template<FieldExpr LHS, FieldExpr RHS, class Seed>
   requires (is_test_linear_v<LHS> && is_test_linear_v<RHS>)
GENDIL_HOST_DEVICE
auto pullback(const OuterExpr<LHS, RHS>& /*expr*/, const Seed& /*seed*/)
{
   static_assert(dependent_false_v<LHS, RHS, Seed>,
      "OuterExpr pullback: both operands test-dependent (quadratic in test function, invalid). "
      "outer(test_expr1, test_expr2) represents a nonlinear weak form.");
   return PullbackResult<>{};
}

// =============================================================================
// AverageExpr Pullback for Test Functions
// =============================================================================
//
// For interior facets, test functions only use minus side with 1/2 factor:
//    average(v) = 0.5 * v_minus
//
// Pullback rule:
//    pullback(average(v), seed) = pullback(v, 0.5 * seed)
//
// Trial functions (not implemented here):
//    average(u) = 0.5 * (u_minus + u_plus)  (handled by AverageExpr::operator())
//


/**
 * @brief Pullback for average(E) on interior facets when E is test-linear.
 *
 * With the current/minus-side convention used by the pullback operator:
 *
 *     average(E) = 0.5 * E_minus
 *
 * Therefore:
 *
 *     pullback(average(E), seed)
 *       = pullback(E, 0.5 * seed)
 *
 * This handles average(v), average(grad(v)), and more complex
 * test-linear expressions such as average(coef * grad(v)), provided
 * the inner expression E has a valid pullback rule.
 */
template<FieldExpr E, class Seed>
   requires (is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const AverageExpr<E>& expr, const Seed& seed)
{
   return pullback(expr.expr, 0.5 * seed);
}

/**
 * @brief Reject pullback of average(E) when E is not test-linear.
 *
 * Trial-only/test-free average expressions should be evaluated as coefficients
 * and should not enter the pullback path directly.
 */
template<FieldExpr E, class Seed>
   requires (!is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const AverageExpr<E>&, const Seed&)
{
   static_assert(dependent_false_v<E, Seed>,
      "AverageExpr pullback is only implemented for test-linear expressions. "
      "Trial-only/test-free average expressions should be evaluated, not pulled back.");
   return PullbackResult<>{};
}

// =============================================================================
// JumpExpr Pullback for Test Functions
// =============================================================================
//
// For interior facets, test functions only use minus side:
//    jump(v) = v_minus
//
// Pullback rule:
//    pullback(jump(v), seed) = pullback(v, seed)
//
// Trial functions (not implemented here):
//    jump(u) = u_minus - u_plus  (handled by JumpExpr::operator())
//
// The same current-side rule applies to any supported test-linear expression.
//

/**
 * @brief Pullback for jump of test space (interior facets).
 *
 * For test functions on interior facets:
 *    jump(v) = v_minus (only current/minus side)
 *
 * Therefore:
 *    pullback(jump(v), seed) = pullback(v, seed) = ValueChannel with seed
 *
 * @tparam Name Test space name
 * @tparam Shape Test space field shape (Scalar or Vector)
 * @tparam Seed Seed expression type
 */
template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback(const JumpExpr<TestSpace<Name, Shape>>& jump_expr, const Seed& seed)
{
   // For test functions, jump only uses minus side
   // Extract underlying test space and pullback directly
   const auto& test_space = jump_expr.expr;
   return pullback(test_space, seed);
}

/**
 * @brief Pull back a jump of a test-linear expression on the current side.
 *
 * Each local facet visit contributes only its current test row. Thus jump(E)
 * has the same pullback as E, including gradients and div(v). The opposite
 * row is handled by the neighboring visit, with its own trial jump.
 */
template<FieldExpr E, class Seed>
   requires (is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const JumpExpr<E>& expr, const Seed& seed)
{
   return pullback(expr.expr, seed);
}

template<FieldExpr E, class Seed>
   requires (!is_test_linear_v<E>)
GENDIL_HOST_DEVICE
auto pullback(const JumpExpr<E>&, const Seed&)
{
   static_assert(dependent_false_v<E, Seed>,
      "JumpExpr pullback requires a test-linear expression. Trial-only/test-free "
      "jumps must be evaluated as coefficients, not pulled back.");
   return PullbackResult<>{};
}

// =============================================================================
// Physical-boundary trace lowering
// =============================================================================

namespace boundary_pullback_detail
{
// Rebuild only the numerical expression. The original form, declarations, and
// integrand signature remain unchanged. No coefficient callback is evaluated.
template<class E>
struct lower
{
   GENDIL_HOST_DEVICE
   static auto apply(const E& expr) { return expr; }
};
template<FieldExpr E>
struct lower<JumpExpr<E>>
{
   GENDIL_HOST_DEVICE
   static auto apply(const JumpExpr<E>& expr) { return lower<E>::apply(expr.expr); }
};
template<FieldExpr E>
struct lower<AverageExpr<E>>
{
   GENDIL_HOST_DEVICE
   static auto apply(const AverageExpr<E>& expr) { return lower<E>::apply(expr.expr); }
};
#define GENDIL_BOUNDARY_UNARY(Node) \
template<FieldExpr E> struct lower<Node<E>> { \
   GENDIL_HOST_DEVICE static auto apply(const Node<E>& expr) { \
      auto child = lower<E>::apply(expr.expr); \
      return Node<std::remove_cvref_t<decltype(child)>>{child}; \
   } \
};
GENDIL_BOUNDARY_UNARY(NegExpr)
GENDIL_BOUNDARY_UNARY(TransposeExpr)
GENDIL_BOUNDARY_UNARY(SymExpr)
GENDIL_BOUNDARY_UNARY(SkewExpr)
GENDIL_BOUNDARY_UNARY(NormalProjectionExpr)
GENDIL_BOUNDARY_UNARY(TangentialProjectionExpr)
#undef GENDIL_BOUNDARY_UNARY
#define GENDIL_BOUNDARY_BINARY(Node) \
template<FieldExpr L, FieldExpr R> struct lower<Node<L, R>> { \
   GENDIL_HOST_DEVICE static auto apply(const Node<L, R>& expr) { \
      auto lhs = lower<L>::apply(expr.lhs); \
      auto rhs = lower<R>::apply(expr.rhs); \
      return Node<std::remove_cvref_t<decltype(lhs)>, std::remove_cvref_t<decltype(rhs)>>{lhs, rhs}; \
   } \
};
GENDIL_BOUNDARY_BINARY(ProductExpr)
GENDIL_BOUNDARY_BINARY(MultFieldExpr)
GENDIL_BOUNDARY_BINARY(DotExpr)
GENDIL_BOUNDARY_BINARY(InnerExpr)
GENDIL_BOUNDARY_BINARY(OuterExpr)
#undef GENDIL_BOUNDARY_BINARY

template<FieldExpr Head, FieldExpr... Tail>
struct lower<SumExpr<Head, Tail...>>
{
   GENDIL_HOST_DEVICE
   static auto apply(const SumExpr<Head, Tail...>& expr)
   {
      return std::apply([] (const auto&... terms)
      {
         return SumExpr{lower<std::remove_cvref_t<decltype(terms)>>::apply(terms)...};
      }, expr.terms);
   }
};
#define GENDIL_BOUNDARY_REJECT_TRACE(Node) \
template<FieldExpr E> struct lower<Node<E>> { \
   GENDIL_HOST_DEVICE static auto apply(const Node<E>&) { \
      static_assert(dependent_false_v<E>, \
         "Explicit minus/plus traces are interior-only and are invalid on physical boundaries."); \
   } \
};
GENDIL_BOUNDARY_REJECT_TRACE(MinusTraceExpr)
GENDIL_BOUNDARY_REJECT_TRACE(PlusTraceExpr)
#undef GENDIL_BOUNDARY_REJECT_TRACE

template<FieldExpr Adv, FieldExpr E>
struct lower<UpwindExpr<Adv, E>>
{
   GENDIL_HOST_DEVICE
   static auto apply(const UpwindExpr<Adv, E>&)
   {
      static_assert(dependent_false_v<Adv, E>,
         "upwind(beta, expr) requires two interior traces; "
         "physical-boundary inflow data must be specified explicitly.");
   }
};

template<FieldExpr Adv, class Interior, class Exterior>
struct lower<BoundaryUpwindExpr<Adv, Interior, Exterior>>
{
   GENDIL_HOST_DEVICE
   static auto apply(const BoundaryUpwindExpr<Adv, Interior, Exterior>& expr)
   {
      auto velocity = lower<Adv>::apply(expr.adv_expr);
      auto interior = lower<Interior>::apply(expr.interior_expr);
      auto exterior = lower<Exterior>::apply(expr.exterior_expr);
      return BoundaryUpwindExpr<decltype(velocity), decltype(interior), decltype(exterior)>{
         velocity, interior, exterior};
   }
};
} // namespace boundary_pullback_detail

/** Apply one-sided jump/average identities before the ordinary algebra pullback. */
template<FieldExpr E, class Seed>
GENDIL_HOST_DEVICE
auto pullback_boundary(const E& expr, const Seed& seed)
{
   return pullback(boundary_pullback_detail::lower<E>::apply(expr), seed);
}

// =============================================================================
// Canonical Global Interior Pullback
// =============================================================================
//
// The ordinary pullback rules above are row-relative: jump(v) contributes to
// the current/minus row only. Canonical global-face lowering evaluates both
// rows from one canonical face context, so test-side dependence is lowered into
// side-qualified channel keys instead. Trace selection is pushed structurally
// through the expression tree before pullback; this keeps public minus/plus as
// semantic DSL nodes without carrying a side template parameter through every
// pullback overload.

namespace details
{

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const Expr& expr)
{
   return minus(expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const Expr& expr)
{
   return plus(expr);
}

GENDIL_HOST_DEVICE
inline auto TraceToMinusSide(const ScaleExpr& expr)
{
   return expr;
}

GENDIL_HOST_DEVICE
inline auto TraceToPlusSide(const ScaleExpr& expr)
{
   return expr;
}

GENDIL_HOST_DEVICE
inline auto TraceToMinusSide(const Normal& expr)
{
   return expr;
}

GENDIL_HOST_DEVICE
inline auto TraceToPlusSide(const Normal& expr)
{
   return expr;
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const MinusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as minus(minus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const PlusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as minus(plus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const MinusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as plus(minus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const PlusTraceExpr<Expr>&)
{
   static_assert(
      dependent_false_v<Expr>,
      "Nested interior trace expressions such as plus(plus(expr)) are not "
      "part of the current canonical global-interior lowering.");
   return PullbackResult<>{};
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const NegExpr<Expr>& expr)
{
   return -TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const NegExpr<Expr>& expr)
{
   return -TraceToPlusSide(expr.expr);
}

template<FieldExpr Head, FieldExpr... Tail>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const SumExpr<Head, Tail...>& expr)
{
   return std::apply(
      [] (const auto&... terms)
      {
         return (TraceToMinusSide(terms) + ...);
      },
      expr.terms);
}

template<FieldExpr Head, FieldExpr... Tail>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const SumExpr<Head, Tail...>& expr)
{
   return std::apply(
      [] (const auto&... terms)
      {
         return (TraceToPlusSide(terms) + ...);
      },
      expr.terms);
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const DotExpr<LHS, RHS>& expr)
{
   return dot(TraceToMinusSide(expr.lhs), TraceToMinusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const DotExpr<LHS, RHS>& expr)
{
   return dot(TraceToPlusSide(expr.lhs), TraceToPlusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const InnerExpr<LHS, RHS>& expr)
{
   return inner(TraceToMinusSide(expr.lhs), TraceToMinusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const InnerExpr<LHS, RHS>& expr)
{
   return inner(TraceToPlusSide(expr.lhs), TraceToPlusSide(expr.rhs));
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const ProductExpr<LHS, RHS>& expr)
{
   return TraceToMinusSide(expr.lhs) * TraceToMinusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const ProductExpr<LHS, RHS>& expr)
{
   return TraceToPlusSide(expr.lhs) * TraceToPlusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const MultFieldExpr<LHS, RHS>& expr)
{
   return TraceToMinusSide(expr.lhs) * TraceToMinusSide(expr.rhs);
}

template<FieldExpr LHS, FieldExpr RHS>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const MultFieldExpr<LHS, RHS>& expr)
{
   return TraceToPlusSide(expr.lhs) * TraceToPlusSide(expr.rhs);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const JumpExpr<Expr>& expr)
{
   return TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const JumpExpr<Expr>& expr)
{
   return - TraceToPlusSide(expr.expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const AverageExpr<Expr>& expr)
{
   return 0.5 * TraceToMinusSide(expr.expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const AverageExpr<Expr>& expr)
{
   return 0.5 * TraceToPlusSide(expr.expr);
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const SymExpr<Expr>& expr)
{
   return sym(TraceToMinusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const SymExpr<Expr>& expr)
{
   return sym(TraceToPlusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const SkewExpr<Expr>& expr)
{
   return skew(TraceToMinusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const SkewExpr<Expr>& expr)
{
   return skew(TraceToPlusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const NormalProjectionExpr<Expr>& expr)
{
   return normal_projection(TraceToMinusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const NormalProjectionExpr<Expr>& expr)
{
   return normal_projection(TraceToPlusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToMinusSide(const TangentialProjectionExpr<Expr>& expr)
{
   return tangential_projection(TraceToMinusSide(expr.expr));
}

template<FieldExpr Expr>
GENDIL_HOST_DEVICE
auto TraceToPlusSide(const TangentialProjectionExpr<Expr>& expr)
{
   return tangential_projection(TraceToPlusSide(expr.expr));
}

} // namespace details

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<TestSpace<Name, Shape>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<ValueMinusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<TestSpace<Name, Shape>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<TestSpace<Name, Shape>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<ValuePlusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<GradientExpr<TestSpace<Name, Shape>>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<GradientMinusChannel>(seed))};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<GradientExpr<TestSpace<Name, Shape>>>&,
   const Seed& seed)
{
   using SeedType = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<GradientExpr<TestSpace<Name, Shape>>, SeedType>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return PullbackResult{MakeStaticMap(MakeEntry<GradientPlusChannel>(seed))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<DivergenceExpr<TestSpace<Name, FieldShape::Vector>>>&,
   const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<DivergenceExpr<VectorTestSpace<Name>>, Seed>,
      "Divergence pullback requires a scalar seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientMinusChannel>(
      pullback_detail::DivergenceSeed<Seed>{seed}))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<DivergenceExpr<TestSpace<Name, FieldShape::Vector>>>&,
   const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<DivergenceExpr<VectorTestSpace<Name>>, Seed>,
      "Divergence pullback requires a scalar seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientPlusChannel>(
      pullback_detail::DivergenceSeed<Seed>{seed}))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<CurlExpr<TestSpace<Name, FieldShape::Vector>>>&,
   const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<CurlExpr<VectorTestSpace<Name>>, Seed>,
      "Curl pullback requires a vector seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientMinusChannel>(
      pullback_detail::CurlSeed<Seed>{seed}))};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<CurlExpr<TestSpace<Name, FieldShape::Vector>>>&,
   const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<CurlExpr<VectorTestSpace<Name>>, Seed>,
      "Curl pullback requires a vector seed.");
   return PullbackResult{MakeStaticMap(MakeEntry<GradientPlusChannel>(
      pullback_detail::CurlSeed<Seed>{seed}))};
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MinusTraceExpr<Expr>& expr,
   const Seed& seed)
{
   auto side_expr = details::TraceToMinusSide(expr.expr);
   using SideExpr = std::remove_cvref_t<decltype(side_expr)>;

   static_assert(
      !std::is_same_v<SideExpr, MinusTraceExpr<Expr>>,
      "Canonical global interior pullback cannot pull back a pure minus-side "
      "test-free leaf. The integrand must be test-linear.");

   return pullback_global_interior(side_expr, seed);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const PlusTraceExpr<Expr>& expr,
   const Seed& seed)
{
   auto side_expr = details::TraceToPlusSide(expr.expr);
   using SideExpr = std::remove_cvref_t<decltype(side_expr)>;

   static_assert(
      !std::is_same_v<SideExpr, PlusTraceExpr<Expr>>,
      "Canonical global interior pullback cannot pull back a pure plus-side "
      "test-free leaf. The integrand must be test-linear.");

   return pullback_global_interior(side_expr, seed);
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const TestSpace<Name, Shape>&,
   const Seed&)
{
   static_assert(
      dependent_false_v<Seed>,
      "Unqualified TestSpace on canonical global interior facets is "
      "ambiguous. Use minus(v), plus(v), jump(v), or average(v).");
   return PullbackResult<>{};
}

template<StaticString Name, FieldShape Shape, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const GradientExpr<TestSpace<Name, Shape>>&,
   const Seed&)
{
   static_assert(
      dependent_false_v<Seed>,
      "Unqualified grad(v) on canonical global interior facets is ambiguous. "
      "Use minus(grad(v)), plus(grad(v)), jump(grad(v)), or "
      "average(grad(v)).");
   return PullbackResult<>{};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const DivergenceExpr<TestSpace<Name, FieldShape::Vector>>&,
   const Seed&)
{
   static_assert(dependent_false_v<Seed>,
      "Unqualified div(v) on canonical global interior facets is ambiguous. "
      "Use minus(div(v)), plus(div(v)), jump(div(v)), or average(div(v)).");
   return PullbackResult<>{};
}

template<StaticString Name, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const CurlExpr<TestSpace<Name, FieldShape::Vector>>&,
   const Seed&)
{
   static_assert(dependent_false_v<Seed>,
      "Unqualified curl(v) on canonical global interior facets is ambiguous. "
      "Use minus(curl(v)), plus(curl(v)), jump(curl(v)), or average(curl(v)).");
   return PullbackResult<>{};
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const JumpExpr<Expr>& expr,
   const Seed& seed)
{
   auto minus_channels =
      pullback_global_interior(details::TraceToMinusSide(expr.expr), seed);
   auto plus_channels =
      pullback_global_interior(details::TraceToPlusSide(expr.expr), -seed);
   return merge_pullback_results(minus_channels, plus_channels);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const AverageExpr<Expr>& expr,
   const Seed& seed)
{
   auto half_seed = 0.5 * seed;
   auto minus_channels =
      pullback_global_interior(details::TraceToMinusSide(expr.expr), half_seed);
   auto plus_channels =
      pullback_global_interior(details::TraceToPlusSide(expr.expr), half_seed);
   return merge_pullback_results(minus_channels, plus_channels);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const NegExpr<Expr>& expr,
   const Seed& seed)
{
   return pullback_global_interior(expr.expr, -seed);
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const SymExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<SymExpr<Expr>, Seed>,
      "Symmetric projection pullback requires a matrix seed.");
   return pullback_global_interior(expr.expr, sym(seed));
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const SkewExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<SkewExpr<Expr>, Seed>,
      "Skew-symmetric projection pullback requires a matrix seed.");
   return pullback_global_interior(expr.expr, skew(seed));
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const NormalProjectionExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<NormalProjectionExpr<Expr>, Seed>,
      "Normal projection pullback requires a vector seed.");
   return pullback_global_interior(expr.expr, normal_projection(seed));
}

template<FieldExpr Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const TangentialProjectionExpr<Expr>& expr, const Seed& seed)
{
   static_assert(is_seed_shape_compatible_v<TangentialProjectionExpr<Expr>, Seed>,
      "Tangential projection pullback requires a vector seed.");
   return pullback_global_interior(expr.expr, tangential_projection(seed));
}

template<FieldExpr Head, FieldExpr... Tail, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const SumExpr<Head, Tail...>& expr,
   const Seed& seed)
{
   using Expr = SumExpr<Head, Tail...>;
   using S = std::remove_cvref_t<Seed>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   return std::apply(
      [&](const auto&... terms)
      {
         return merge_pullback_results(
            pullback_global_interior(terms, seed)...);
      },
      expr.terms);
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const DotExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = DotExpr<LHS, RHS>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      return pullback_global_interior(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      return pullback_global_interior(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior DotExpr pullback requires exactly one "
         "TestLinear operand and one TestFree operand.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const InnerExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;

   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      return pullback_global_interior(expr.lhs, seed * expr.rhs);
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      return pullback_global_interior(expr.rhs, seed * expr.lhs);
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior InnerExpr pullback requires exactly one "
         "TestLinear operand and one TestFree operand.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const ProductExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;
   using Expr = ProductExpr<LHS, RHS>;

   static_assert(
      is_seed_shape_compatible_v<Expr, S>,
      "pullback_global_interior(expr, seed) requires seed to have the same "
      "FieldShape as expr.");

   if constexpr (Expr::product_kind == ProductKind::ScalarTimes)
   {
      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         static_assert(
            field_shape_v<R> == FieldShape::Scalar,
            "Canonical global interior scalar-times pullback requires the "
            "test-free multiplier to be Scalar.");
         return pullback_global_interior(expr.lhs, seed * expr.rhs);
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         static_assert(
            field_shape_v<L> == FieldShape::Scalar,
            "Canonical global interior scalar-times pullback requires the "
            "test-free multiplier to be Scalar.");
         return pullback_global_interior(expr.rhs, seed * expr.lhs);
      }
      else
      {
         static_assert(
            dependent_false_v<L, R, S>,
            "Canonical global interior ProductExpr pullback requires exactly "
            "one TestLinear operand and one TestFree scalar operand.");
      }
   }
   else if constexpr (Expr::product_kind == ProductKind::MatVec)
   {
      static_assert(
         field_shape_v<L> == FieldShape::Matrix,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Matrix-shaped left operand.");
      static_assert(
         field_shape_v<R> == FieldShape::Vector,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Vector-shaped right operand.");
      static_assert(
         field_shape_v<S> == FieldShape::Vector,
         "Canonical global interior MatVec ProductExpr pullback requires a "
         "Vector-shaped seed.");

      if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
      {
         return pullback_global_interior(
            expr.lhs,
            outer(seed, expr.rhs));
      }
      else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
      {
         return pullback_global_interior(
            expr.rhs,
            transpose(expr.lhs) * seed);
      }
      else
      {
         static_assert(
            dependent_false_v<L, R, S>,
            "Canonical global interior MatVec ProductExpr pullback requires "
            "exactly one TestLinear operand. Supported directions are "
            "test-linear matrix * test-free vector and test-free matrix * "
            "test-linear vector.");
      }
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior ProductExpr pullback currently supports "
         "scalar-times products and linear MatVec products.");
   }
}

template<FieldExpr LHS, FieldExpr RHS, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(
   const MultFieldExpr<LHS, RHS>& expr,
   const Seed& seed)
{
   using L = std::remove_cvref_t<LHS>;
   using R = std::remove_cvref_t<RHS>;
   using S = std::remove_cvref_t<Seed>;

   auto multiply = [&]<class Factor>(const Factor& factor)
   {
      if constexpr (has_field_shape_v<Factor>)
      {
         static_assert(field_shape_v<Factor> == FieldShape::Scalar,
            "Canonical global interior multiplication pullback requires a scalar test-free multiplier.");
         return seed * factor;
      }
      else return pullback_detail::ScalarMultiplierSeed<Seed, Factor>{seed, factor};
   };
   if constexpr (is_test_linear_v<L> && is_test_free_v<R>)
   {
      return pullback_global_interior(expr.lhs, multiply(expr.rhs));
   }
   else if constexpr (is_test_free_v<L> && is_test_linear_v<R>)
   {
      return pullback_global_interior(expr.rhs, multiply(expr.lhs));
   }
   else
   {
      static_assert(
         dependent_false_v<L, R, S>,
         "Canonical global interior multiplication pullback requires exactly "
         "one TestLinear operand and one TestFree scalar operand.");
   }
}

template<class Expr, class Seed>
GENDIL_HOST_DEVICE
auto pullback_global_interior(const Expr&, const Seed&)
{
   static_assert(
      dependent_false_v<Expr, Seed>,
      "Canonical global interior pullback cannot lower this traced expression "
      "yet. The expression is structurally valid, but the canonical pullback "
      "must provide a side-qualified channel rule for its test-linear part.");
   return PullbackResult<>{};
}

} // namespace gendil
