// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesserial.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolatevaluesthreaded.hpp"

namespace gendil
{

/**
 * @brief generic implementation of an operator interpolating values at quadrature points
 * from the given degrees-of-freedom.
 * 
 * @tparam KernelContext The kernel context.
 * @tparam ElementDofToQuad A tuple of cached 1D basis at quadrature points.
 * @tparam DofTensor The input degrees-of-freedom tensor.
 * @param ctx The kernel context.
 * @param element_quad_data The tuple containing data at quadrature point for each dimension.
 * @param u The input degrees-of-freedom.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template <
   typename KernelContext,
   typename ElementDofToQuad,
   typename DofTensor >
GENDIL_HOST_DEVICE
auto InterpolateValues( const KernelContext & ctx,
                        const ElementDofToQuad & element_quad_data,
                        const DofTensor & u )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      // serial interpolation
      return InterpolateValuesSerial( element_quad_data, u );
   }
   else
   {
      // threaded interpolation
      return InterpolateValuesThreaded( ctx, element_quad_data, u );
   }
}

} // namespace gendil
