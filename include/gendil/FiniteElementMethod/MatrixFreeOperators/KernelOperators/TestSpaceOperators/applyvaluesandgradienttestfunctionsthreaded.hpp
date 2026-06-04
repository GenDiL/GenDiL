// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/finiteelementmethod.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/quadraturepointvalues.hpp"

namespace gendil {

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename... ValueInputTypes,
   typename... GradientInputTypes,
   typename... OutputTypes >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctionsThreaded(
   KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const std::tuple<ValueInputTypes...> & values_tuple,
   const std::tuple<GradientInputTypes...> & gradients_tuple,
   std::tuple<OutputTypes...> & output_tuple )
{
   static_assert(sizeof...(ValueInputTypes) == sizeof...(GradientInputTypes),
      "Vector ApplyValuesAndGradientTestFunctionsThreaded: values and gradients "
      "must have the same number of components.");

   static_assert(sizeof...(ValueInputTypes) == sizeof...(OutputTypes),
      "Vector ApplyValuesAndGradientTestFunctionsThreaded: output must have the "
      "same number of components as input.");

   constexpr Integer NumComponents = sizeof...(ValueInputTypes);

   static_assert(
      std::tuple_size_v<std::remove_cvref_t<ElementDofToQuad>> == NumComponents,
      "Vector ApplyValuesAndGradientTestFunctionsThreaded: element_quad_data must "
      "have one DofToQuad tuple per vector component.");

   ConstexprLoop<NumComponents>([&](auto c)
   {
      constexpr Integer C = decltype(c)::value;

      ApplyValuesAndGradientTestFunctionsThreaded<Add>(
         thread,
         std::get<C>(element_quad_data),
         std::get<C>(values_tuple),
         std::get<C>(gradients_tuple),
         std::get<C>(output_tuple));
   });
}

template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename ValuesInput,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
void ApplyValuesAndGradientTestFunctionsThreaded(
   KernelContext & thread,
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
      auto uq = MakeQuadraturePointValuesContainer( thread, element_quad_data );
      ApplyGradientTestFunctionsAtQPoints( thread, element_quad_data, DGuq, uq );
      uq += Duq;
      if constexpr ( Add )
         dofs_out += ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
      else
         dofs_out  = ApplyTestFunctionsThreaded( thread, element_quad_data, uq );
   }
}

}
