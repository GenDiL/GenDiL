// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"

namespace gendil
{

template<class T>
struct is_embedded_cell_reference_face_map : std::false_type {};

template<Integer Dim>
struct is_embedded_cell_reference_face_map<
   NonconformingHyperCubeFaceMap<Dim>> : std::true_type {};

template<class T>
inline constexpr bool is_embedded_cell_reference_face_map_v =
   is_embedded_cell_reference_face_map<std::remove_cvref_t<T>>::value;

template<class Points1D, class Face, Integer CellCoordIndex>
struct NonconformingMappedPointSet1D
{
   Face face;

   GENDIL_HOST_DEVICE
   Real coord(Integer q) const
   {
      const auto mapped =
         face.template MapReferenceToFaceCoordinates1d<CellCoordIndex>(
            Point<1>{Points1D::GetCoord(q)});
      return mapped[0];
   }

   GENDIL_HOST_DEVICE
   Real weights(Integer q) const
   {
      return Points1D::GetWeight(q);
   }

   GENDIL_HOST_DEVICE
   static Real GetWeight(Integer q)
   {
      return Points1D::GetWeight(q);
   }
};

} // namespace gendil
