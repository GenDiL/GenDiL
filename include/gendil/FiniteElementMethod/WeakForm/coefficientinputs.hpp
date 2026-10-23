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
 * @brief Inverse facet size coefficient input: ||J^{-T} n_ref||
 *
 * Available only with FacetQuadraturePointContext.
 * For Cartesian mesh with spacing h, returns 1/h.
 *
 * Used for SIPDG penalty scaling: tau = kappa * h^{-1}
 */
struct InverseFacetSize : CoefficientInputBase {};

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
