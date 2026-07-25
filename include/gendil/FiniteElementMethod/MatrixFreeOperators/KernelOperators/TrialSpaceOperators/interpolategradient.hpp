// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/facetquaddata.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradientserial.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/TrialSpaceOperators/interpolategradientthreaded.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename ... ScalarMaps,
   typename ... ScalarDofTensors,
   size_t ... I >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   KernelContext & ctx,
   const VectorDofToQuad<ScalarMaps...>& quad_data,
   const std::tuple< ScalarDofTensors ... > & u,
   std::index_sequence< I... > )
{
   return std::make_tuple(
      InterpolateGradient(
         ctx,
         GetVectorComponent<I>(quad_data),
         std::get<I>(u))...);
}

template <
   typename KernelContext,
   typename ... ScalarMaps,
   typename ... ScalarDofTensors >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   KernelContext & ctx,
   const VectorDofToQuad<ScalarMaps...>& quad_data,
   const std::tuple< ScalarDofTensors ... > & u )
{
   static_assert(
      sizeof...(ScalarMaps) == sizeof...(ScalarDofTensors),
      "InterpolateGradient requires one input DoF tensor per "
      "VectorDofToQuad component.");
   return InterpolateGradient(
      ctx,
      quad_data,
      u,
      std::make_index_sequence<sizeof...(ScalarDofTensors)>{});
}

template <
   typename KernelContext,
   typename ... Entries,
   typename InputTensor >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   KernelContext& thread,
   const TensorProductData<Entries...>& element_quad_data,
   const InputTensor& u)
{
   using ProductOperator =
      TensorProductData<Entries...>;
   constexpr Integer dim =
      tensor_product_entry_count_v<ProductOperator>;
   using quad_shape = make_contraction_output_shape<ProductOperator>;
   using rdims = typename KernelContext::template register_dimensions< dim >;
   using rshape = subsequence_t< quad_shape, rdims >;
   using shape = cat_t< rshape, std::index_sequence< dim > >;
   // auto Gu = MakeSerialRecursiveArray< Real >( shape{} );
   auto Gu = MakeStaticFIFOView< Real >( shape{} );

   if constexpr ( !is_threaded_v< KernelContext > )
   {
      // Register-only configurations have no shared-memory staging dimensions.
      InterpolateGradientSerial( element_quad_data, u, Gu );
   }
   else
   {
      // threaded interpolation
      InterpolateGradientThreaded( thread, element_quad_data, u, Gu );
   }

   return Gu;
}

template <
   typename KernelContext,
   CellFaceView Face,
   typename FaceQuadData,
   typename DofTensor >
GENDIL_HOST_DEVICE
auto InterpolateGradient(
   KernelContext & ctx,
   const Face & face,
   const FaceQuadData & face_quad_data,
   const DofTensor & u )
{
   auto&& facet_qd = GetFacetQuadData( face_quad_data, face );
   return InterpolateGradient( ctx, facet_qd, u );
}

}
