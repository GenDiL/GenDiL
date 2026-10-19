// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include "gendil/Algebra/staticalgebraconcepts.hpp"
#include "gendil/Utilities/RecursiveArray/recursivearray.hpp"
#include "gendil/FiniteElementMethod/WeakForm/fieldshapetraits.hpp"
#include "gendil/FiniteElementMethod/WeakForm/normal.hpp"

namespace gendil
{
namespace weak_form_detail
{
// Facet geometry supplies a unit normal. Materialize readable entries before
// computing the orthogonal projection so proxies and inputs are each read once.
template<bool Tangential, class Value, class Direction>
GENDIL_HOST_DEVICE constexpr auto ProjectFacetVector(const Value& value, const Direction& normal)
{
   static_assert(StaticVector<Value> && StaticVector<Direction>,
      "Facet projection requires readable static vectors with real arithmetic entries.");
   if constexpr (StaticVector<Value> && StaticVector<Direction>)
   {
      constexpr size_t size = static_extent_v<Value>;
      static_assert(size > 0 && size == static_extent_v<Direction>,
         "Facet projection requires matching nonempty vector and normal dimensions.");
      if constexpr (size > 0 && size == static_extent_v<Direction>)
      {
         using Scalar = vector_value_t<Value>;
         using NormalScalar = vector_value_t<Direction>;
         using Entry = std::common_type_t<Scalar, NormalScalar>;
         SerialRecursiveArray<Entry, size> result{};
         std::array<Entry, size> direction{};
         Entry normal_component = 0;
         for (size_t i = 0; i < size; ++i)
         {
            const Entry entry = static_cast<Entry>(static_cast<Scalar>(GetValue(value, i)));
            result(i) = entry;
            direction[i] = static_cast<Entry>(static_cast<NormalScalar>(GetValue(normal, i)));
            normal_component += entry*direction[i];
         }
         for (size_t i = 0; i < size; ++i)
         {
            const Entry projected = normal_component*direction[i];
            if constexpr (Tangential) result(i) = static_cast<Entry>(result(i)) - projected;
            else result(i) = projected;
         }
         return result;
      }
   }
}
} // namespace weak_form_detail

template<FieldExpr Expr>
struct NormalProjectionExpr : FieldBase
{
   static_assert(VectorExpr<Expr>, "normal_projection requires a vector-shaped expression.");
   Expr expr;

   GENDIL_HOST_DEVICE constexpr explicit NormalProjectionExpr(const Expr& operand) : expr(operand) {}

   template<class... Args>
   GENDIL_HOST_DEVICE constexpr auto operator()(Args&&... args) const
   {
      const auto& value = expr(std::forward<Args>(args)...);
      const auto normal = Normal{}(std::forward<Args>(args)...);
      return weak_form_detail::ProjectFacetVector<false>(value, normal);
   }
};

template<FieldExpr Expr>
struct TangentialProjectionExpr : FieldBase
{
   static_assert(VectorExpr<Expr>, "tangential_projection requires a vector-shaped expression.");
   Expr expr;

   GENDIL_HOST_DEVICE constexpr explicit TangentialProjectionExpr(const Expr& operand) : expr(operand) {}

   template<class... Args>
   GENDIL_HOST_DEVICE constexpr auto operator()(Args&&... args) const
   {
      const auto& value = expr(std::forward<Args>(args)...);
      const auto normal = Normal{}(std::forward<Args>(args)...);
      return weak_form_detail::ProjectFacetVector<true>(value, normal);
   }
};

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const NormalProjectionExpr<Expr>& value)
{ return os << "normal_projection(" << value.expr << ")"; }

template<FieldExpr Expr>
std::ostream& operator<<(std::ostream& os, const TangentialProjectionExpr<Expr>& value)
{ return os << "tangential_projection(" << value.expr << ")"; }

/**
 * @brief Keep the vector component perpendicular to a physical facet.
 *
 * normal_projection(u) is (u dot n)*n for the physical unit normal n. For example,
 * dot(normal_projection(u), normal_projection(v)) penalizes normal motion.
 * The result is a vector of the same dimension, distinct from the scalar
 * normal flux dot(u, Normal{}) and from normal_derivative(u).
 *
 * The operand and physical normal are each evaluated once. Facet quadrature
 * contexts supply a unit normal; reversing its direction preserves the projector.
 * Readable static-vector inputs produce an owning SerialRecursiveArray using
 * their common arithmetic scalar type with the normal.
 *
 * This is a facet-only operation on supported vector expressions, including
 * coefficients and scalar gradients. Vector length must match the physical
 * normal. Apply existing minus/plus, jump, or average operators to select
 * interior traces. Both sides use the existing common facet normal, so projection
 * commutes with jump and average. Test pullback applies the same projection to
 * its vector seed and retains the operand's existing value or gradient channels.
 */
template<VectorExpr Expr>
GENDIL_HOST_DEVICE constexpr auto normal_projection(const Expr& expr)
{ return NormalProjectionExpr<Expr>{expr}; }

/**
 * @brief Keep the vector component tangent to a physical facet.
 *
 * tangential_projection(u) is u - normal_projection(u), evaluated with one
 * operand evaluation. For example, jump(tangential_projection(u)) expresses
 * tangential discontinuity without choosing a tangent basis or cross-product
 * convention. The dimension, storage, normal, and facet rules of
 * normal_projection apply. Its real-vector pullback is also self-adjoint.
 */
template<VectorExpr Expr>
GENDIL_HOST_DEVICE constexpr auto tangential_projection(const Expr& expr)
{ return TangentialProjectionExpr<Expr>{expr}; }

} // namespace gendil
