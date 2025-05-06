// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionsserial.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionsthreaded.hpp"

namespace gendil {

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & quad_point_values )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      using dof_shape  = make_contraction_input_shape< ProductOperator >;
      auto dofs_out = MakeSerialRecursiveArray< Real >( dof_shape{} );
      // auto dofs_out = MakeStaticFIFOView< Real >( dof_shape{} );
      ApplyTestFunctionsSerial< false, DiffDim >( element_quad_data, quad_point_values, dofs_out );
      return dofs_out;
   }
   else
   {
      return ApplyTestFunctionsThreaded< DiffDim >( thread, element_quad_data, quad_point_values );
   }
}

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor,
   typename OutputTensor >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & quad_point_values,
   OutputTensor & dofs_out )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      ApplyTestFunctionsSerial< true, DiffDim >( element_quad_data, quad_point_values, dofs_out );
   }
   else
   {
      dofs_out += ApplyTestFunctionsThreaded< DiffDim >( thread, element_quad_data, quad_point_values );
   }
}

}
