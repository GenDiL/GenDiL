// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/nonconformingpoints.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/nonconformingfacetquaddatatraits.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace nonconforming_facet_qdata_detail
{
struct NonconformingDofToQuadLeafTransform
{
   template<Integer Offset, class Face, class QData>
   GENDIL_HOST_DEVICE
   auto operator()(const Face& face, const QData& qdata) const
   {
      return MakeNonconformingDofToQuadData(
         face,
         qdata,
         std::integral_constant<Integer, Offset>{});
   }
};

struct NonconformingMappedPointSetLeafTransform
{
   template<Integer Offset, class Face, class QData>
   GENDIL_HOST_DEVICE
   auto operator()(const Face& face, const QData&) const
   {
      return NonconformingMappedPointSet1D<
         std::remove_cvref_t<QData>,
         Face,
         Offset>{face};
   }
};

template<
   Integer Offset,
   class LeafTransform,
   class Face,
   class QData>
GENDIL_HOST_DEVICE
auto TransformTensorProductQDataAtOffset(
   const Face& face,
   const QData& qdata,
   const LeafTransform& transform);

template<
   Integer Offset,
   class LeafTransform,
   class Face,
   class... Entries,
   size_t... I>
GENDIL_HOST_DEVICE
auto TransformTensorProductQDataAtOffsetImpl(
   const Face& face,
   const TensorProductData<Entries...>& qdata,
   const LeafTransform& transform,
   std::index_sequence<I...>)
{
   return MakeTensorProductData(
      TransformTensorProductQDataAtOffset<
         Offset + TensorProductQDataEntryOffset<I, Entries...>::value>(
            face,
            GetTensorProductEntry<I>(qdata),
            transform)...);
}

template<
   Integer Offset,
   class LeafTransform,
   class Face,
   class QData>
GENDIL_HOST_DEVICE
auto TransformTensorProductQDataAtOffset(
   const Face& face,
   const QData& qdata,
   const LeafTransform& transform)
{
   using Data = std::remove_cvref_t<QData>;
   if constexpr (is_tensor_product_data_v<Data>)
   {
      return TransformTensorProductQDataAtOffsetImpl<Offset>(
         face,
         qdata,
         transform,
         std::make_index_sequence<tensor_product_entry_count_v<Data>>{});
   }
   else
   {
      return transform.template operator()<Offset>(face, qdata);
   }
}

} // namespace nonconforming_facet_qdata_detail

template<class Face, class QData>
GENDIL_HOST_DEVICE
auto MakeNestedNonconformingDofToQuadData(
   const Face& face,
   const QData& qdata)
{
   constexpr bool Supported =
      nonconforming_facet_qdata_detail::EveryTensorProductQDataLeaf<
         IsSupportedNonconformingFacetDofToQuad1D,
         std::remove_cvref_t<QData>>::value;
   static_assert(
      Supported,
      "Nested nonconforming finite-element facet qdata must terminate in "
      "CachedDofToQuad entries.");
   if constexpr (Supported)
   {
      return nonconforming_facet_qdata_detail::
         TransformTensorProductQDataAtOffset<0>(
            face,
            qdata,
            nonconforming_facet_qdata_detail::
               NonconformingDofToQuadLeafTransform{});
   }
   else
   {
      return Empty{};
   }
}

template<class Face, class QData>
GENDIL_HOST_DEVICE
auto MakeNestedNonconformingMappedPointSetData(
   const Face& face,
   const QData& qdata)
{
   constexpr bool Supported =
      nonconforming_facet_qdata_detail::EveryTensorProductQDataLeaf<
         IsStaticPointSetFacetQData1D,
         std::remove_cvref_t<QData>>::value;
   static_assert(
      Supported,
      "Nested nonconforming mesh facet qdata must terminate in static "
      "point-set entries.");
   if constexpr (Supported)
   {
      return nonconforming_facet_qdata_detail::
         TransformTensorProductQDataAtOffset<0>(
            face,
            qdata,
            nonconforming_facet_qdata_detail::
               NonconformingMappedPointSetLeafTransform{});
   }
   else
   {
      return Empty{};
   }
}

template<class Face, class... PointSets, size_t... Is>
GENDIL_HOST_DEVICE
auto MakeNonconformingMappedPointSetTupleImpl(
   const Face& face,
   const TensorProductData<PointSets...>&,
   std::index_sequence<Is...>)
{
   using Conformity = typename std::remove_cvref_t<Face>::conformity_type;
   static_assert(
      is_embedded_cell_reference_face_map_v<Conformity>,
      "Nonconforming affine mesh facet qdata requires an embedded "
      "cell-reference face map. Currently only "
      "NonconformingHyperCubeFaceMap<Dim> is supported.");

   return MakeTensorProductData(
      NonconformingMappedPointSet1D<
         PointSets,
         Face,
         static_cast<Integer>(Is)>{face}...);
}

template<class Face, class... PointSets>
GENDIL_HOST_DEVICE
auto MakeNonconformingMappedPointSetTuple(
   const Face& face,
   const TensorProductData<PointSets...>& point_sets)
{
   return MakeNonconformingMappedPointSetTupleImpl(
      face,
      point_sets,
      std::make_index_sequence<sizeof...(PointSets)>{});
}

template<class Face, class... ScalarMaps, size_t... Component>
GENDIL_HOST_DEVICE
auto MakeNonconformingVectorDofToQuadDataImpl(
   const Face& face,
   const VectorDofToQuad<ScalarMaps...>& local_face_qd,
   std::index_sequence<Component...>)
{
   return MakeVectorDofToQuad(
      MakeNestedNonconformingDofToQuadData(
         face,
         GetVectorComponent<Component>(local_face_qd))...);
}

template<class Face, class... ScalarMaps>
GENDIL_HOST_DEVICE
auto MakeNonconformingVectorDofToQuadData(
   const Face& face,
   const VectorDofToQuad<ScalarMaps...>& local_face_qd)
{
   return MakeNonconformingVectorDofToQuadDataImpl(
      face,
      local_face_qd,
      std::index_sequence_for<ScalarMaps...>{});
}

template<class Face, class LocalFaceQData>
GENDIL_HOST_DEVICE
auto MakeNonconformingFacetQuadData(
   const Face& face,
   const LocalFaceQData& local_face_qd)
{
   if constexpr (is_vector_dof_to_quad_v<LocalFaceQData>)
   {
      return MakeNonconformingVectorDofToQuadData(
         face,
         local_face_qd);
   }
   else if constexpr (
      IsSupportedNonconformingFacetDofToQuadTuple_v<LocalFaceQData>)
   {
      return MakeNestedNonconformingDofToQuadData(face, local_face_qd);
   }
   else if constexpr (
      IsStaticPointSetFacetQDataTuple_v<LocalFaceQData>)
   {
      using Conformity = typename std::remove_cvref_t<Face>::conformity_type;
      static_assert(
         is_embedded_cell_reference_face_map_v<Conformity>,
         "Nonconforming affine mesh facet qdata requires an embedded "
         "cell-reference face map. Currently only "
         "NonconformingHyperCubeFaceMap<Dim> is supported.");
      return MakeNestedNonconformingMappedPointSetData(
         face,
         local_face_qd);
   }
   else
   {
      static_assert(
         dependent_false_v<Face, LocalFaceQData>,
         "Nonconforming facet qdata is not implemented for this "
         "finite-element or mesh cell/qdata type.");
   }
}

} // namespace gendil
