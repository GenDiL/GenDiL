// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctionsserial.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applygradienttestfunctionsthreaded.hpp"

namespace gendil
{

/**
 * @brief Scalar ApplyGradientTestFunctions.
 */
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
      ApplyGradientTestFunctions<Add>( element_quad_data, DGuq, dofs_out );
   }
   else
   {
      ApplyGradientTestFunctionsThreaded<Add>( thread, element_quad_data, DGuq, dofs_out );
   }
}

/**
 * @brief Helper: Apply gradient test functions componentwise for tuple storage.
 */
template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename... InputTensors,
   typename... OutputTensors,
   size_t... I >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctionsTupleImpl(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const std::tuple< InputTensors... > & DGuq_tuple,
   std::tuple< OutputTensors... > & dofs_out_tuple,
   std::index_sequence< I... > )
{
   static_assert(sizeof...(InputTensors) == sizeof...(OutputTensors),
      "Input and output tuple sizes must match");

   // Apply gradient test functions for each component
   (ApplyGradientTestFunctions<Add>(
      thread,
      std::get<I>(element_quad_data),
      std::get<I>(DGuq_tuple),
      std::get<I>(dofs_out_tuple)), ...);
}

/**
 * @brief Tuple overload for ApplyGradientTestFunctions (vector FE gradients).
 * Applies gradient test functions componentwise for tuple-backed gradient storage.
 */
template <
   bool Add,
   typename KernelContext,
   typename ElementDofToQuad,
   typename... InputTensors,
   typename... OutputTensors >
GENDIL_HOST_DEVICE
void ApplyGradientTestFunctions(
   const KernelContext & thread,
   const ElementDofToQuad & element_quad_data,
   const std::tuple< InputTensors... > & DGuq_tuple,
   std::tuple< OutputTensors... > & dofs_out_tuple )
{
   ApplyGradientTestFunctionsTupleImpl<Add>(
      thread,
      element_quad_data,
      DGuq_tuple,
      dofs_out_tuple,
      std::make_index_sequence<sizeof...(InputTensors)>{});
}

template <
   bool Add,
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename GradientsInput,
   typename Output >
GENDIL_HOST_DEVICE
auto ApplyGradientTestFunctions(
   const KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const GradientsInput & DGuq,
   Output & dofs_out )
{
   constexpr Integer local_face_index = Face::local_face_index_type::value;
   const auto & local_face_quad_data = std::get< local_face_index >( face_quad_data );
   if constexpr ( Face::is_conforming )
   {
      return ApplyGradientTestFunctions< Add >( ctx, local_face_quad_data, DGuq, dofs_out );
   }
   else
   {
      auto non_conforming_face_quad_data = MakeNonconformingDofToQuadData( face, local_face_quad_data );
      return ApplyGradientTestFunctions< Add >( ctx, non_conforming_face_quad_data, DGuq, dofs_out );
   }
}

}
