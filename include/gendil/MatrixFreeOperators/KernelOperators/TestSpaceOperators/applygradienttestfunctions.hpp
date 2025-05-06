// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctionsserial.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctionsthreaded.hpp"

namespace gendil
{

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename Input,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctions(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const Input & DGuq,
   Output & dofs_out )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      ApplyGradientTestFunctions( element_quad_data, DGuq, dofs_out );
   }
   else
   {
      ApplyGradientTestFunctionsThreaded( thread, element_quad_data, DGuq, dofs_out );
   }
}

}
