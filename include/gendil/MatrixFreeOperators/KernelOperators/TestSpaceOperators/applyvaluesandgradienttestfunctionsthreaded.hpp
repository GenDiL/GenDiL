// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil {

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename ValuesInput,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctionsThreaded(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const ValuesInput & Duq,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   constexpr bool face_interp = is_face_interpolation_v< ElementDofToQuad >;

   if constexpr ( face_interp )
   {
      if constexpr ( Add )
         dofs_out += ApplyTestFunctionsThreaded( thread, element_quad_data, Duq );
      else
         dofs_out  = ApplyTestFunctionsThreaded( thread, element_quad_data, Duq );
      constexpr Integer Rank = std::tuple_size_v< ElementDofToQuad >;
      ConstexprLoop<Rank>([&]( auto dim )
      {
         dofs_out += ApplyTestFunctionsThreaded< dim >( thread, element_quad_data, DGuq );
      } );
   }
   else
   {
      using IntegrationRule = typename ElementDofToQuad::integration_rule;
      auto uq = MakeQuadraturePointValuesContainer( thread, IntegrationRule{} );
      ApplyGradientTestFunctionsAtQPoints( thread, element_quad_data, DGuq, uq );
      uq += Duq;
      if constexpr ( Add )
         dofs_out += ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
      else
         dofs_out  = ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
   }
}

}
