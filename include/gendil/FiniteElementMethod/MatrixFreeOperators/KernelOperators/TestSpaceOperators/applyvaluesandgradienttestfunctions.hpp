// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/facetquaddata.hpp"
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
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      // Register-only configurations have no shared-memory staging dimensions.
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
   auto&& facet_qd = GetFacetQuadData( face_quad_data, face );
   return ApplyValuesAndGradientTestFunctions< Add >( ctx, facet_qd, Duq, DGuq, dofs_out );
}

}
