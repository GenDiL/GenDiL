// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <concepts>
#include <tuple>
#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/NumericalIntegration/QuadraturePoints/nonconformingpoints.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/doftoquad.hpp"

namespace gendil
{

template<class T>
struct IsEmptyFacetQuadData : std::false_type {};

template<>
struct IsEmptyFacetQuadData<Empty> : std::true_type {};

template<class T>
inline constexpr bool IsEmptyFacetQuadData_v =
   IsEmptyFacetQuadData<std::remove_cvref_t<T>>::value;

template<class T>
struct IsSupportedNonconformingFacetDofToQuad1D : std::false_type {};

template<class ShapeFunctions, class IntegrationRule>
struct IsSupportedNonconformingFacetDofToQuad1D<
   CachedDofToQuad<ShapeFunctions, IntegrationRule>> : std::true_type {};

template<class T>
inline constexpr bool IsSupportedNonconformingFacetDofToQuad1D_v =
   IsSupportedNonconformingFacetDofToQuad1D<
      std::remove_cvref_t<T>>::value;

template<class T>
struct IsSupportedNonconformingFacetDofToQuadTuple : std::false_type {};

template<class... DofToQuads>
struct IsSupportedNonconformingFacetDofToQuadTuple<
   std::tuple<DofToQuads...>>
   : std::bool_constant<
        (IsSupportedNonconformingFacetDofToQuad1D_v<DofToQuads> && ...)> {};

template<class T>
inline constexpr bool IsSupportedNonconformingFacetDofToQuadTuple_v =
   IsSupportedNonconformingFacetDofToQuadTuple<
      std::remove_cvref_t<T>>::value;

template<class T>
struct IsStaticPointSetFacetQData1D : std::false_type {};

template<class T>
   requires requires(Integer q)
   {
      { std::remove_cvref_t<T>::GetCoord(q) } -> std::convertible_to<Real>;
      { std::remove_cvref_t<T>::GetWeight(q) } -> std::convertible_to<Real>;
   }
struct IsStaticPointSetFacetQData1D<T> : std::true_type {};

template<class T>
inline constexpr bool IsStaticPointSetFacetQData1D_v =
   IsStaticPointSetFacetQData1D<std::remove_cvref_t<T>>::value;

template<class T>
struct IsStaticPointSetFacetQDataTuple : std::false_type {};

template<class... PointSets>
struct IsStaticPointSetFacetQDataTuple<std::tuple<PointSets...>>
   : std::bool_constant<
        (IsStaticPointSetFacetQData1D_v<PointSets> && ...)> {};

template<class T>
inline constexpr bool IsStaticPointSetFacetQDataTuple_v =
   IsStaticPointSetFacetQDataTuple<std::remove_cvref_t<T>>::value;

template<class Face, class... PointSets, size_t... Is>
GENDIL_HOST_DEVICE
auto MakeNonconformingMappedPointSetTupleImpl(
   const Face& face,
   const std::tuple<PointSets...>&,
   std::index_sequence<Is...>)
{
   using Conformity = typename std::remove_cvref_t<Face>::conformity_type;
   static_assert(
      is_embedded_cell_reference_face_map_v<Conformity>,
      "Nonconforming affine mesh facet qdata requires an embedded "
      "cell-reference face map. Currently only "
      "NonconformingHyperCubeFaceMap<Dim> is supported.");

   return std::make_tuple(
      NonconformingMappedPointSet1D<
         PointSets,
         Face,
         static_cast<Integer>(Is)>{face}...);
}

template<class Face, class... PointSets>
GENDIL_HOST_DEVICE
auto MakeNonconformingMappedPointSetTuple(
   const Face& face,
   const std::tuple<PointSets...>& point_sets)
{
   return MakeNonconformingMappedPointSetTupleImpl(
      face,
      point_sets,
      std::make_index_sequence<sizeof...(PointSets)>{});
}

template<class Face, class LocalFaceQData>
GENDIL_HOST_DEVICE
auto MakeNonconformingFacetQuadData(
   const Face& face,
   const LocalFaceQData& local_face_qd)
{
   if constexpr (
      IsSupportedNonconformingFacetDofToQuadTuple_v<LocalFaceQData>)
   {
      return MakeNonconformingDofToQuadData(face, local_face_qd);
   }
   else if constexpr (
      IsStaticPointSetFacetQDataTuple_v<LocalFaceQData>)
   {
      return MakeNonconformingMappedPointSetTuple(face, local_face_qd);
   }
   else
   {
      static_assert(
         dependent_false_v<Face, LocalFaceQData>,
         "Nonconforming facet qdata is not implemented for this "
         "finite-element or mesh cell/qdata type.");
   }
}

template<class FaceQuadData, CellFaceView Face>
GENDIL_HOST_DEVICE
decltype(auto) GetFacetQuadData(
   const FaceQuadData& face_quad_data,
   const Face& face)
{
   constexpr Integer local_face_index = Face::local_face_index_type::value;
   auto&& local_face_qd = std::get<local_face_index>(face_quad_data);

   static_assert(
      !IsEmptyFacetQuadData_v<decltype(local_face_qd)>,
      "GetFacetQuadData selected an Empty global facet qdata slot. "
      "The face side local_face_index does not match the selected qdata tuple.");

   if constexpr (Face::is_conforming)
   {
      return (local_face_qd);
   }
   else
   {
      return MakeNonconformingFacetQuadData(face, local_face_qd);
   }
}

} // namespace gendil
