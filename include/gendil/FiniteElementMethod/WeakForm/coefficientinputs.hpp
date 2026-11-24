// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/dslbase.hpp"

namespace gendil
{

// Describe field and geometry values supplied to coefficient callbacks.
// These tags carry no runtime data; extraction is implemented in coefficient.hpp.
struct ElementIndex          : CoefficientInputBase {};
struct QuadraturePointIndex  : CoefficientInputBase {};
struct PhysicalCoordinates   : CoefficientInputBase {};
struct ReferenceCoordinates  : CoefficientInputBase {};
struct JacobianDeterminant   : CoefficientInputBase {};
struct Jacobian              : CoefficientInputBase {};
struct JacobianInverse       : CoefficientInputBase {};

/**
 * @brief Inverse cell size normal to a facet: ||J_K^{-T} n_ref||.
 *
 * Uses the [0,1]^d reference cell and its unit reference facet normal.
 * Evaluated at the facet quadrature point using the selected cell's Jacobian;
 * on curved cells this is a local scale, not a single geometric cell height.
 * The two adjacent cells can have different values. On interior facets,
 * select minus(...), plus(...), or average(...) of the coefficient; for
 * example, average(tau) gives a shared SIPDG penalty when tau depends on
 * this input. Boundary facets use their sole adjacent cell.
 */
struct InverseNormalCellSize : CoefficientInputBase {};

namespace weak_form_detail
{

template<bool Gradient, FieldShape Shape>
struct FieldInputShape
{
   static_assert(Shape == FieldShape::Scalar || Shape == FieldShape::Vector,
      "Coefficient field inputs support scalar or componentwise vector fields.");
   static constexpr auto field_shape = Shape;
   static constexpr unsigned argument_rank =
      (Shape == FieldShape::Vector ? 1u : 0u) + (Gradient ? 1u : 0u);
};

} // namespace weak_form_detail

// Read a scalar field by default; use VectorFieldValue for vector fields.
// The declared shape is checked when constructing an operator. Extents always
// come from the selected FE space; field_shape and argument_rank describe rank.
template <StaticString Name, FieldShape Shape>
struct FieldValue : CoefficientInputBase,
                    weak_form_detail::FieldInputShape<false, Shape>
{
   static constexpr auto name = Name;
};

// Shape describes the underlying field: a scalar gradient has rank one;
// a vector gradient has rank two, indexed by (component, physical direction).
template <StaticString Name, FieldShape Shape>
struct FieldGradient : CoefficientInputBase,
                       weak_form_detail::FieldInputShape<true, Shape>
{
   static constexpr auto name = Name;
};

template <StaticString Name>
using VectorFieldValue = FieldValue<Name, FieldShape::Vector>;

template <StaticString Name>
using VectorFieldGradient = FieldGradient<Name, FieldShape::Vector>;

} // namespace gendil
