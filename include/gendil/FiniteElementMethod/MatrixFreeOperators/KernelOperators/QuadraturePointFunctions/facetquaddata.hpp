// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/QuadraturePointFunctions/nonconformingfacetquaddata.hpp"

#include <type_traits>

namespace gendil
{

/**
 * @brief Recognizes the Empty sentinel used for unavailable local-face qdata.
 *
 * The trait remains beside GetFacetQuadData because it controls facade
 * selection and has no consumer outside this dispatch path.
 */
template<class T>
struct IsEmptyFacetQuadData : std::false_type {};

template<>
struct IsEmptyFacetQuadData<Empty> : std::true_type {};

template<class T>
inline constexpr bool IsEmptyFacetQuadData_v =
   IsEmptyFacetQuadData<std::remove_cvref_t<T>>::value;

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
