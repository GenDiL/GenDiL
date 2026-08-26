// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

namespace gendil {

namespace details
{

template<class T>
struct TensorProductDimension : std::integral_constant<Integer, 1>
{
};

template<class... Entries>
struct TensorProductDimension<TensorProductData<Entries...>>
   : std::integral_constant<
        Integer,
        (TensorProductDimension<std::remove_cvref_t<Entries>>::value +
         ... + 0)>
{
};

template<class QData1D>
GENDIL_HOST_DEVICE
Real GetWeight1D(const QData1D& qdata, Integer q)
{
   using QData = std::remove_cvref_t<QData1D>;
   if constexpr (requires { qdata.weights(q); })
   {
      return qdata.weights(q);
   }
   else if constexpr (requires { QData::GetWeight(q); })
   {
      return QData::GetWeight(q);
   }
   else if constexpr (requires { qdata.GetWeight(q); })
   {
      return qdata.GetWeight(q);
   }
   else
   {
      static_assert(
         dependent_false_v<QData1D>,
         "GetWeight(q, qdata) requires 1D qdata with weights(q) or "
         "a static GetWeight(q) point-set interface.");
   }
}

template<Integer Dim, class QData>
GENDIL_HOST_DEVICE
Real GetTensorProductWeightImpl(
   const TensorIndex<Dim>& index,
   const QData& qdata,
   Integer& dimension);

template<Integer Dim, class... Entries, size_t... Is>
GENDIL_HOST_DEVICE
Real GetTensorProductWeightImpl(
   const TensorIndex<Dim>& index,
   const TensorProductData<Entries...>& qdata,
   Integer& dimension,
   std::index_sequence<Is...>)
{
   Real weight = 1.0;
   (
      (weight *= GetTensorProductWeightImpl(
         index,
         GetTensorProductEntry<Is>(qdata),
         dimension)),
      ...);
   return weight;
}

template<Integer Dim, class QData>
GENDIL_HOST_DEVICE
Real GetTensorProductWeightImpl(
   const TensorIndex<Dim>& index,
   const QData& qdata,
   Integer& dimension)
{
   if constexpr (is_tensor_product_data_v<QData>)
   {
      return GetTensorProductWeightImpl(
         index,
         qdata,
         dimension,
         std::make_index_sequence<
            tensor_product_entry_count_v<QData>>{});
   }
   else
   {
      return GetWeight1D(qdata, index[dimension++]);
   }
}

} // namespace details

/**
 * @brief Get the quadrature weight at a given quadrature point.
 * 
 * @tparam Dim The dimension of the space.
 * @tparam Entries Immediate one-dimensional or nested tensor-product qdata
 * entries.
 * @param index The index of the quadrature point.
 * @param quad_data Strongly typed tensor-product quadrature data.
 * @return The quadrature weight.
 * 
 * @note Assumes tensor finite element with tensor integration rule.
 */
template<Integer Dim, class... Entries>
GENDIL_HOST_DEVICE
Real GetWeight(
   const TensorIndex<Dim>& index,
   const TensorProductData<Entries...>& quad_data)
{
   static_assert(
      details::TensorProductDimension<
         TensorProductData<Entries...>>::value == Dim,
      "GetWeight requires the flattened tensor-product quadrature dimension "
      "to match TensorIndex::Dim.");
   Integer dimension = 0;
   return details::GetTensorProductWeightImpl(
      index,
      quad_data,
      dimension);
}

template<Integer Dim, class... ScalarMaps>
GENDIL_HOST_DEVICE
Real GetWeight(
   const TensorIndex<Dim>& index,
   const VectorDofToQuad<ScalarMaps...>& quad_data)
{
   return GetWeight(index, GetVectorComponent<0>(quad_data));
}

template<Integer Dim, class... QData>
GENDIL_HOST_DEVICE
Real GetWeight(
   const TensorIndex<Dim>&,
   const std::tuple<QData...>&)
{
   static_assert(
      dependent_false_v<std::tuple<QData...>>,
      "GetWeight volume qdata requires TensorProductData; plain "
      "std::tuple volume qdata is unsupported.");
   return 0.0;
}

template < CellFaceView FaceInfo, Integer Dim, typename... DofToQuads >
GENDIL_HOST_DEVICE
Real GetWeight(
   const FaceInfo & face_info,
   const TensorIndex< Dim > & index,
   const std::tuple< DofToQuads... > & face_quad_data )
{
   static_assert(
      2*Dim == sizeof...( DofToQuads ),
      "Dimension mismatch between FaceInfo and face_quad_data"
   );
   constexpr Integer local_face_index = FaceInfo::local_face_index_type::value;
   const auto & local_face_quad_data = std::get< local_face_index >( face_quad_data );
   return face_info.Measure() * GetWeight( index, local_face_quad_data );
}

}
