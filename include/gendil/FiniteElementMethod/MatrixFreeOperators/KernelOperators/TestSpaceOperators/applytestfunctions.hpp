// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/facetquaddata.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionsserial.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TestSpaceOperators/applytestfunctionsthreaded.hpp"

namespace gendil {

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename ... InputTensors,
   size_t ... I >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   KernelContext & thread,
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
   KernelContext & thread,
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
   KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & quad_point_values )
{
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      using dof_shape  = make_contraction_input_shape< ProductOperator >;
      auto dofs_out = MakeSerialRecursiveArray< Real >( dof_shape{} );
      ApplyTestFunctionsSerial< false, DiffDim >(
         element_quad_data,
         quad_point_values,
         dofs_out );
      return dofs_out;
   }
   else
   {
      return ApplyTestFunctionsThreaded< DiffDim >( thread, element_quad_data, quad_point_values );
   }
}


template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const InputTensor & quad_point_values )
{
   auto&& facet_qd = GetFacetQuadData( face_quad_data, face );
   return ApplyTestFunctions( ctx, facet_qd, quad_point_values );
}

/**
 * @brief Helper for component-wise application after face extraction (vector FE).
 *
 * IMPORTANT: face_quad_data is indexed face-first for vector FE:
 *   face_quad_data[local_face_index] = tuple<comp0_dofquad, comp1_dofquad>
 *
 * The caller (vector FE overload below) extracts the correct face first, then
 * this helper applies the scalar implementation component-wise.
 *
 * @param ctx Kernel execution context
 * @param this_face_data Tuple of DofToQuad data for THIS face (one per component)
 * @param quad_point_values Tuple of quadrature point values (one per component)
 * @param seq Index sequence over components
 * @return Tuple of DOF contributions (one per component)
 */
template <
   typename KernelContext,
   typename ThisFaceData,
   typename... InputTensors,
   size_t... I >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions_ComponentWise(
   KernelContext & ctx,
   const ThisFaceData & this_face_data,
   const std::tuple< InputTensors... > & quad_point_values,
   std::index_sequence< I... > )
{
   // Static assert: number of components must match
   static_assert(
      sizeof...(InputTensors) == std::tuple_size_v<std::remove_cvref_t<ThisFaceData>>,
      "Vector face ApplyTestFunctions: face data and value tuple must have the same number of components.");

   // Apply scalar ApplyTestFunctions per component
   // No Face parameter - face already extracted by caller
   return std::make_tuple(
      ApplyTestFunctions( ctx, std::get< I >( this_face_data ), std::get< I >( quad_point_values ) )...
   );
}

/**
 * @brief Vector FE overload: Apply test functions for boundary/interior facets.
 *
 * CRITICAL DATA LAYOUT:
 * For vector FE, face_quad_data is indexed FACE-FIRST:
 *   face_quad_data = tuple<
 *     tuple<comp0, comp1>,  // face 0
 *     tuple<comp0, comp1>,  // face 1
 *     ...
 *   >
 *
 * This overload:
 * 1. Extracts THIS face's data: face_quad_data[local_face_index]
 * 2. Applies scalar ApplyTestFunctions component-wise
 *
 * @param ctx Kernel execution context
 * @param face Face view containing local_face_index
 * @param face_quad_data Face quadrature data (face-first, then component-indexed)
 * @param quad_point_values Tuple of quadrature point values (one per component)
 * @return Tuple of DOF contributions (one per component)
 */
template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename... InputTensors >
GENDIL_HOST_DEVICE
auto ApplyTestFunctions(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const std::tuple< InputTensors... > & quad_point_values )
{
   auto&& this_face_data = GetFacetQuadData( face_quad_data, face );

   // Apply component-wise using extracted face data
   return ApplyTestFunctions_ComponentWise(
      ctx, this_face_data, quad_point_values,
      std::make_index_sequence< sizeof...( InputTensors ) >{}
   );
}

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename InputTensor,
   typename OutputTensor >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext & thread,
   const ProductOperator & element_quad_data,
   const InputTensor & quad_point_values,
   OutputTensor & dofs_out )
{
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      ApplyTestFunctionsSerial< true, DiffDim >(
         element_quad_data,
         quad_point_values,
         dofs_out );
   }
   else
   {
      dofs_out += ApplyTestFunctionsThreaded< DiffDim >( thread, element_quad_data, quad_point_values );
   }
}

/**
 * @brief Apply and accumulate test functions for vector fields with tuple storage.
 *
 * For vector FE with tuple-per-component storage, applies test functions component-wise
 * and accumulates into tuple DOF output. Mirrors ApplyTestFunctions tuple overload pattern.
 *
 * @param quad_data Test-space quadrature data (tuple-indexed for vector FE)
 * @param quad_point_values Tuple of per-component quadrature values
 * @param dofs_out Tuple of per-component DOF outputs (accumulated)
 */
template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename... InputTensors,
   typename... OutputTensors,
   size_t... I >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext & thread,
   const ProductOperator & quad_data,
   const std::tuple< InputTensors... > & quad_point_values,
   std::tuple< OutputTensors... > & dofs_out,
   std::index_sequence< I... > )
{
   // Component-wise accumulation via fold expression
   // Do NOT rely on tuple += tuple
   ( ApplyAddTestFunctions< DiffDim >( thread, std::get< I >( quad_data ), std::get< I >( quad_point_values ), std::get< I >( dofs_out ) ), ... );
}

template <
   Integer DiffDim = std::numeric_limits< GlobalIndex >::max(),
   typename KernelContext,
   typename ProductOperator,
   typename... InputTensors,
   typename... OutputTensors >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext & thread,
   const ProductOperator & quad_data,
   const std::tuple< InputTensors... > & quad_point_values,
   std::tuple< OutputTensors... > & dofs_out )
{
   static_assert( sizeof...( InputTensors ) == sizeof...( OutputTensors ),
      "ApplyAddTestFunctions: Input and output tuple sizes must match" );
   ApplyAddTestFunctions< DiffDim >( thread, quad_data, quad_point_values, dofs_out, std::make_index_sequence< sizeof...( InputTensors ) >{} );
}

template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename InputTensor,
   typename OutputTensor >
GENDIL_HOST_DEVICE
auto ApplyAddTestFunctions(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const InputTensor & quad_point_values,
   OutputTensor & dofs_out )
{
   dofs_out += ApplyTestFunctions( ctx, face, face_quad_data, quad_point_values );
}

/**
 * @brief Tuple-aware face ApplyAddTestFunctions for vector finite elements.
 *
 * For vector FE, both quad_point_values and dofs_out are tuples.
 * This overload dispatches componentwise to scalar face ApplyAddTestFunctions.
 *
 * Face quad data is tuple-indexed by local face, shared across all spatial components.
 */
template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename... InputTensors,
   typename... OutputTensors,
   size_t... I >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions_Tuple_Impl(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const std::tuple< InputTensors... > & quad_point_values,
   std::tuple< OutputTensors... > & dofs_out,
   std::index_sequence< I... > )
{
   static_assert( sizeof...( InputTensors ) == sizeof...( OutputTensors ),
      "ApplyAddTestFunctions (face tuple): Input and output tuple sizes must match" );

   // For vector FE, face_quad_data is indexed FACE-FIRST:
   //   face_quad_data = tuple<tuple<comp0,comp1>, tuple<comp0,comp1>, ...>
   // Must extract THIS face first, then apply component-wise.

   auto&& this_face_data = GetFacetQuadData( face_quad_data, face );

   // Component-wise application using fold expression
   // Extract component I from THIS face's data
   ( ApplyAddTestFunctions( ctx, std::get< I >( this_face_data ), std::get< I >( quad_point_values ), std::get< I >( dofs_out ) ), ... );
}

template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename... InputTensors,
   typename... OutputTensors >
GENDIL_HOST_DEVICE
void ApplyAddTestFunctions(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const std::tuple< InputTensors... > & quad_point_values,
   std::tuple< OutputTensors... > & dofs_out )
{
   ApplyAddTestFunctions_Tuple_Impl( ctx, face, face_quad_data, quad_point_values, dofs_out, std::make_index_sequence< sizeof...( InputTensors ) >{} );
}

}
