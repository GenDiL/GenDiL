// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctionsserial.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctionsthreaded.hpp"

namespace gendil {

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename ValuesInput,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctions(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const ValuesInput & Duq,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   if constexpr ( is_serial_v< KernelContext > )
      ApplyValuesAndGradientTestFunctions< Add >( element_quad_data, Duq, DGuq, dofs_out );
   else
   {
      ApplyValuesAndGradientTestFunctionsThreaded< Add >( thread, element_quad_data, Duq, DGuq, dofs_out );
   }
}

}
