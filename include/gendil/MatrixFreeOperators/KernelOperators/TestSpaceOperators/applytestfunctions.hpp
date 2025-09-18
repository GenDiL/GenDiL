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
   typename ... InputTensors,
   size_t ... I >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   const KernelContext & thread,
   const ProductOperator & quad_data,
   const std::tuple< InputTensors ... > & quad_point_values,
   std::index_sequence< I... > )
{
   return std::make_tuple( ApplyTestFunctions< DiffDim >( thread, std::get< I >( quad_data ), std::get< I >( quad_point_values ) )... );
}

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename ... InputTensors >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   const KernelContext & thread,
   const ProductOperator & quad_data,
   const std::tuple< InputTensors ... > & quad_point_values )
{
   return ApplyTestFunctions< DiffDim >( thread, quad_data, quad_point_values, std::make_index_sequence< sizeof...( InputTensors ) >{} );
}

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
   typename ... InputTensors,
   typename ... OutputTensors >
GENDIL_HOST_DEVICE
auto ApplyAddTestFunctions(
   const KernelContext & thread,
   const ProductOperator & quad_data,
   const std::tuple< InputTensors ... > & quad_point_values,
   std::tuple< OutputTensors ... > & dofs_out  )
{
   // TODO assert size of input output
   ConstexprLoop< sizeof...( InputTensors ) >( [&]( auto i ){
      ApplyAddTestFunctions< DiffDim >( thread, std::get< i >( quad_data ), std::get< i >( quad_point_values ), std::get< i >( dofs_out ) );
   });
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
