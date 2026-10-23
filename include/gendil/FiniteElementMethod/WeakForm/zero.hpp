// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{
// A vanished residual term still identifies its output rows. It owns neither
// the discarded expression nor its coefficients, and is never evaluated.
// Negation and literal scaling preserve this wrapper; it is not a general
// zero-algebra node for arbitrary coefficient products or new expressions.
template<class Declarations, class LogicalOutputSides>
struct ZeroExpr : FieldBase
{
   using declarations = Declarations;
   using logical_output_sides = LogicalOutputSides;
};

template<class Declarations, class LogicalOutputSides>
std::ostream& operator<<(std::ostream& os, const ZeroExpr<Declarations, LogicalOutputSides>&)
{ return os << "0"; }

namespace weak_form_detail
{

// A removed boundary-upwind state retains only its rank. This keeps the live
// state's shape contract when affine extraction discards the other expression.
// It is a branch marker, not a separately evaluable field expression.
template<FieldShape Shape>
struct BoundaryUpwindZero {};

} // namespace weak_form_detail

namespace weak_form_traits
{

template<class T> inline constexpr bool is_boundary_upwind_zero_v = false;
template<> inline constexpr bool is_boundary_upwind_zero_v<Empty> = true;
template<FieldShape Shape>
inline constexpr bool is_boundary_upwind_zero_v<weak_form_detail::BoundaryUpwindZero<Shape>> = true;

template<class T> struct is_zero_expr : std::false_type {};
template<class Declarations, class LogicalOutputSides>
struct is_zero_expr<ZeroExpr<Declarations, LogicalOutputSides>> : std::true_type {};
template<class T>
inline constexpr bool is_zero_expr_v = is_zero_expr<std::remove_cvref_t<T>>::value;

} // namespace weak_form_traits
} // namespace gendil
