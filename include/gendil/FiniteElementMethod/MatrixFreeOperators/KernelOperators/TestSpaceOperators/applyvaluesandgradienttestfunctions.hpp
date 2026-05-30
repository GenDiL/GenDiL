// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctionsserial.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applyvaluesandgradienttestfunctionsthreaded.hpp"

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
   KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const ValuesInput & Duq,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   if constexpr ( is_serial_v< KernelContext > )
      ApplyValuesAndGradientTestFunctions< Add >( element_quad_data, Duq, DGuq, dofs_out );
   else if constexpr ( KernelContext::thread_block_dim == 0 )
   {
      // Device register-only configurations have one logical thread per work
      // item and no shared-memory staging dimensions.
      ApplyValuesAndGradientTestFunctions< Add >(
         element_quad_data,
         Duq,
         DGuq,
         dofs_out );
   }
   else
   {
      ApplyValuesAndGradientTestFunctionsThreaded< Add >( thread, element_quad_data, Duq, DGuq, dofs_out );
   }
}

template <
   bool Add,
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename ValuesInput,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
auto ApplyValuesAndGradientTestFunctions(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const ValuesInput & Duq,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   constexpr Integer local_face_index = Face::local_face_index_type::value;
   const auto & local_face_quad_data = std::get< local_face_index >( face_quad_data );
   if constexpr ( Face::is_conforming )
   {
      return ApplyValuesAndGradientTestFunctions< Add >( ctx, local_face_quad_data, Duq, DGuq, dofs_out );
   }
   else
   {
      auto non_conforming_face_quad_data = MakeNonconformingDofToQuadData( face, local_face_quad_data );
      return ApplyValuesAndGradientTestFunctions< Add >( ctx, non_conforming_face_quad_data, Duq, DGuq, dofs_out );
   }
}

}
