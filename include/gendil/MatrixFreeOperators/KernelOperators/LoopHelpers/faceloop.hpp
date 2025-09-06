// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/Loop/loops.hpp"
#include "gendil/Meshes/mesh.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"

namespace gendil {

/**
 * @brief Iterate on each face and apply the provided lambda by giving it face information.
 * 
 * @tparam FiniteElementSpace The finite element space type.
 * @tparam InteriorLambda The function type for interior faces.
 * @tparam BoundaryLambda The function type for boundary faces.
 * @param interior_lambda The function to apply on each interior face.
 * @param boundary_lambda The function to apply on each boundary face.
 */
template <
   typename FiniteElementSpace,
   typename InteriorLambda,
   typename BoundaryLambda
>
GENDIL_HOST_DEVICE
void FaceLoop(
   const FiniteElementSpace & fe_space,
   const GlobalIndex & element_index,
   InteriorLambda && interior_lambda,
   BoundaryLambda && boundary_lambda )
{
   constexpr Integer num_faces = FiniteElementSpace::finite_element_type::geometry::num_faces;
   ConstexprLoop< num_faces >(
      [&](auto face_index)
      {
         auto face = mesh::GetLocalFaceInfo( fe_space, element_index, face_index );

         if ( IsBoundaryFace( face ) )
         {
            boundary_lambda( face );
         }
         else
         {
            interior_lambda( face );
         }
      }
   );
}

/**
 * @brief Iterate on each face and apply the provided lambda by giving it face information.
 * 
 * @tparam FiniteElementSpace The finite element space type.
 * @tparam Lambda The function type.
 * @param lambda The function to apply on each face.
 */
template <
   typename FiniteElementSpace,
   typename Lambda
>
GENDIL_HOST_DEVICE
void FaceLoop(
   const FiniteElementSpace & fe_space,
   const GlobalIndex & element_index,
   Lambda && lambda )
{
   FaceLoop( fe_space, element_index, std::forward< Lambda >( lambda ), [](const auto face_info){} );
}

}
