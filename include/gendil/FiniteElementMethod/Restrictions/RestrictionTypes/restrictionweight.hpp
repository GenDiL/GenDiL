// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Scalar-linear restriction weights and their composition.
 *
 * A restriction entry associates an algebraic coordinate @f$j@f$ with a
 * scalar-linear weight @f$w@f$. Gathering applies @f$w@f$ to the algebraic
 * value, while adjoint scattering applies @f$w^*@f$ to the local value. For
 * the real weights currently supported here, @f$w^*=w@f$.
 *
 * `RestrictionUnitWeight` identifies the zero-storage identity fast path.
 * `RestrictionSign` stores a runtime orientation sign as one Boolean: false
 * represents @f$+1@f$ and true represents @f$-1@f$. Arithmetic values
 * represent general scalar weights. A raw `bool` is deliberately not an
 * arithmetic restriction weight because its ordinary zero/one interpretation
 * is different from the sign convention.
 *
 * `ComposeRestrictionWeights(lhs, rhs)` represents applying `rhs` first and
 * then `lhs`. Its adjoint therefore applies the corresponding adjoints in the
 * reverse order. Keeping this operation explicit lets vector, tensor-product,
 * and signed restriction adaptors combine weights without materializing
 * stencils.
 */

#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/restrictionunitweight.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace gendil {

struct RestrictionSign
{
   /** @brief Whether this weight represents the value @f$-1@f$. */
   bool is_negative = false;
};

/**
 * @brief Arithmetic scalar accepted as a restriction weight.
 *
 * Boolean values are intentionally excluded: use `RestrictionSign` when a
 * Boolean flag denotes an orientation sign.
 */
template < class Weight >
concept ArithmeticRestrictionWeight =
   std::is_arithmetic_v< std::remove_cv_t< Weight > > &&
   !std::same_as< std::remove_cv_t< Weight >, bool >;

template < class Value >
GENDIL_HOST_DEVICE
constexpr decltype(auto) ApplyRestrictionWeight(
   RestrictionUnitWeight,
   Value && value )
{
   return std::forward< Value >( value );
}

template < class Value >
GENDIL_HOST_DEVICE
constexpr decltype(auto) ApplyAdjointRestrictionWeight(
   RestrictionUnitWeight,
   Value && value )
{
   return std::forward< Value >( value );
}

template < class Value >
GENDIL_HOST_DEVICE
constexpr auto ApplyRestrictionWeight(
   const RestrictionSign sign,
   const Value & value )
{
   return sign.is_negative ? -value : value;
}

template < class Value >
GENDIL_HOST_DEVICE
constexpr auto ApplyAdjointRestrictionWeight(
   const RestrictionSign sign,
   const Value & value )
{
   return ApplyRestrictionWeight( sign, value );
}

template < class Weight, class Value >
   requires ArithmeticRestrictionWeight< Weight >
GENDIL_HOST_DEVICE
constexpr auto ApplyRestrictionWeight(
   const Weight weight,
   const Value & value )
{
   return weight * value;
}

template < class Weight, class Value >
   requires ArithmeticRestrictionWeight< Weight >
GENDIL_HOST_DEVICE
constexpr auto ApplyAdjointRestrictionWeight(
   const Weight weight,
   const Value & value )
{
   return weight * value;
}

GENDIL_HOST_DEVICE
constexpr RestrictionUnitWeight ComposeRestrictionWeights(
   RestrictionUnitWeight,
   RestrictionUnitWeight )
{
   return {};
}

template < class Weight >
GENDIL_HOST_DEVICE
constexpr Weight ComposeRestrictionWeights(
   RestrictionUnitWeight,
   const Weight & weight )
{
   return weight;
}

template < class Weight >
GENDIL_HOST_DEVICE
constexpr Weight ComposeRestrictionWeights(
   const Weight & weight,
   RestrictionUnitWeight )
{
   return weight;
}

GENDIL_HOST_DEVICE
constexpr RestrictionSign ComposeRestrictionWeights(
   const RestrictionSign left,
   const RestrictionSign right )
{
   return RestrictionSign{
      left.is_negative != right.is_negative };
}

template < class Weight >
   requires ArithmeticRestrictionWeight< Weight >
GENDIL_HOST_DEVICE
constexpr auto ComposeRestrictionWeights(
   const RestrictionSign sign,
   const Weight weight )
{
   return sign.is_negative ? -weight : weight;
}

template < class Weight >
   requires ArithmeticRestrictionWeight< Weight >
GENDIL_HOST_DEVICE
constexpr auto ComposeRestrictionWeights(
   const Weight weight,
   const RestrictionSign sign )
{
   return sign.is_negative ? -weight : weight;
}

template < class Left, class Right >
   requires (
      ArithmeticRestrictionWeight< Left > &&
      ArithmeticRestrictionWeight< Right > )
GENDIL_HOST_DEVICE
constexpr auto ComposeRestrictionWeights(
   const Left left,
   const Right right )
{
   return left * right;
}

} // namespace gendil
