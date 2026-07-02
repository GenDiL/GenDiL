// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <tuple>
#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
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
   else
   {
      static_assert(
         dependent_false_v<Face, LocalFaceQData>,
         "Nonconforming facet qdata is not implemented for this finite-element qdata type.");
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
