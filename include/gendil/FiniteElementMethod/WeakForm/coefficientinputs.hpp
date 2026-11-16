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

template <StaticString Name>
struct FieldValue : CoefficientInputBase
{
   static constexpr auto name = Name;
};

template <StaticString Name>
struct FieldGradient : CoefficientInputBase
{
   static constexpr auto name = Name;
};

} // namespace gendil
