// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/tensorproductdata.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/dot.hpp"
#include "gendil/FiniteElementMethod/WeakForm/normal.hpp"
#include "gendil/FiniteElementMethod/WeakForm/productexpr.hpp"
#include "gendil/FiniteElementMethod/WeakForm/testspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/trialspace.hpp"

namespace gendil
{

/**
 * @brief Differentiate a scalar or vector trial/test field along a facet normal.
 *
 * Use normal_derivative(u)*v in a scalar boundary flux term, or
 * average(normal_derivative(u))*jump(v) on interior facets. Apply trace,
 * jump, and average wrappers after taking the derivative.
 *
 * Scalar fields return dot(grad(field), Normal{}). Vector fields return
 * grad(field)*Normal{}, with gradient(component, direction), so the result
 * has one entry per field component. These are the existing expression types
 * and inherit their normal orientation, geometry requirements, and pullbacks.
 * The full physical gradient is evaluated; no separate derivative channel or
 * expression node is introduced. Normal{} supplies the facet normal, including
 * its existing normalization convention on the selected execution path.
 *
 * Only named scalar/vector trial and test fields are supported. Prescribed
 * FiniteElementField gradients need shape and evaluation support before they
 * can use this convenience function. Coefficients, compound operands, matrix
 * fields, and nested derivatives are not admitted. For tensor diffusion, write
 * dot(K*grad(u), Normal{}) to express the coefficient-weighted normal flux.
 */
template<FieldExpr Field>
   requires (is_scalar_field_reference_v<Field> || is_vector_field_reference_v<Field>)
auto normal_derivative(const Field& field)
{
   if constexpr (is_scalar_field_reference_v<Field>)
   {
      return dot(grad(field), Normal{});
   }
   else
   {
      return grad(field) * Normal{};
   }
}

} // namespace gendil
