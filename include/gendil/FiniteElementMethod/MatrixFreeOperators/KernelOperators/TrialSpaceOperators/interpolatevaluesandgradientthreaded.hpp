// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/TensorContraction/contractionhelper.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/IndexSequenceHelperFunctions/indexsequencehelperfunctions.hpp"
#include "gendil/Utilities/VariadicHelperFunctions/variadichelperfunctions.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"

namespace gendil
{

template <
    typename KernelContext,
    typename ProductOperator,
    typename InputTensor,
    typename ValuesOutputTensor,
    typename GradientOutputTensor >
GENDIL_HOST_DEVICE
void InterpolateValuesAndGradientsThreaded(
   const KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & u,
   ValuesOutputTensor & Bu,
   GradientOutputTensor & Gu )
{
   auto Bu = InterpolateValuesThreaded( thread, element_quad_data, u );

   constexpr bool face_interp = is_face_interpolation_v< ProductOperator >;
   if constexpr ( face_interp )
   {
      constexpr Integer Rank = std::tuple_size_v< ProductOperator >;
      ConstexprLoop< Rank >( [&] ( auto ActiveDim )
      {
         auto gu = InterpolateValuesThreaded< ActiveDim >( thread, element_quad_data, u );
         using quad_shape = make_contraction_output_shape< ProductOperator >;
         using RegisterDimensions = typename KernelContext::template register_dimensions< Rank >;
         using reg_shape = subsequence_t< quad_shape, RegisterDimensions >;
         UnitLoop< reg_shape >([&]( auto ... indices )
         {
            Gu( indices..., ActiveDim ) = gu( indices... );
         });
      } );
   }
   else
   {
      InterpolateGradientAtQPointsThreaded( thread, element_quad_data, Bu, Gu );
   }
}

} // namespace gendil
