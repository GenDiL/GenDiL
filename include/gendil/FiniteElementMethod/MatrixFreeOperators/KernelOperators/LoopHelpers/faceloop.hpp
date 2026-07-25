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
 * @tparam Mesh The mesh type.
 * @tparam InteriorLambda The function type for interior faces.
 * @tparam BoundaryLambda The function type for boundary faces.
 * @param interior_lambda The function to apply on each interior face.
 * @param boundary_lambda The function to apply on each boundary face.
 */
template <
   typename Mesh,
   typename InteriorLambda,
   typename BoundaryLambda
>
GENDIL_HOST_DEVICE
void FaceLoop(
   const Mesh & mesh,
   const GlobalIndex & element_index,
   InteriorLambda && interior_lambda,
   BoundaryLambda && boundary_lambda )
{
   static_assert(
      mesh::LocalFaceConnectivity<Mesh>,
      "FaceLoop requires local face connectivity.");
   static_assert(
      !mesh::LocalFaceConnectivity<Mesh> ||
         mesh::LocalFaceGeometryCompatible<Mesh>,
      "FaceLoop: local face connectivity reference geometry must match "
      "Mesh::cell_type::geometry.");

   if constexpr (
      mesh::LocalFaceConnectivity<Mesh> &&
      mesh::LocalFaceGeometryCompatible<Mesh>)
   {
      using Geometry = mesh::mesh_geometry_t<Mesh>;
      ConstexprLoop< Geometry::num_faces >(
         [&](auto face_index)
         {
            auto face =
               mesh::GetLocalFaceInfo(mesh, element_index, face_index);

            if ( IsBoundaryFace( face ) )
            {
               boundary_lambda( face );
            }
            if ( !IsBoundaryFace( face ) )
            {
               interior_lambda( face );
            }
         }
      );
   }
}

/**
 * @brief Iterate on each face and apply the provided lambda by giving it face information.
 * 
 * @tparam Mesh The mesh type.
 * @tparam Lambda The function type.
 * @param lambda The function to apply on each face.
 */
template <
   typename Mesh,
   typename Lambda
>
GENDIL_HOST_DEVICE
void FaceLoop(
   const Mesh & mesh,
   const GlobalIndex & element_index,
   Lambda && lambda )
{
   FaceLoop(
      mesh,
      element_index,
      std::forward< Lambda >( lambda ),
      [](const auto face_info){});
}

/**
 * @brief Iterate on each interior face and apply the provided lambda by giving it face information.
 * 
 * @tparam Mesh The mesh type.
 * @tparam InteriorLambda The function type for interior faces.
 * @param interior_lambda The function to apply on each interior face.
 */
template <
   typename Mesh,
   typename InteriorLambda
>
GENDIL_HOST_DEVICE
void InteriorFaceLoop(
   const Mesh & mesh,
   const GlobalIndex & element_index,
   InteriorLambda && interior_lambda )
{
   static_assert(
      mesh::LocalFaceConnectivity<Mesh>,
      "InteriorFacets<Name> on MeshIntegrationDomain requires local face "
      "connectivity.");
   static_assert(
      !mesh::LocalFaceConnectivity<Mesh> ||
         mesh::LocalFaceGeometryCompatible<Mesh>,
      "InteriorFacets<Name> on MeshIntegrationDomain requires local face "
      "connectivity whose reference geometry matches "
      "Mesh::cell_type::geometry.");

   if constexpr (
      mesh::LocalFaceConnectivity<Mesh> &&
      mesh::LocalFaceGeometryCompatible<Mesh>)
   {
      using Geometry = mesh::mesh_geometry_t<Mesh>;
      ConstexprLoop< Geometry::num_faces >(
         [&](auto face_index)
         {
            auto face =
               mesh::GetLocalFaceInfo(mesh, element_index, face_index);

            if ( !IsBoundaryFace( face ) )
            {
               interior_lambda( face );
            }
         }
      );
   }
}

template <
   typename Mesh,
   typename BoundaryLambda
>
GENDIL_HOST_DEVICE
void BoundaryFaceLoop(
   const Mesh & mesh,
   const GlobalIndex & element_index,
   BoundaryLambda && boundary_lambda )
{
   static_assert(
      mesh::LocalFaceConnectivity<Mesh>,
      "BoundaryFacets<Name> on MeshIntegrationDomain requires local face "
      "connectivity.");
   static_assert(
      !mesh::LocalFaceConnectivity<Mesh> ||
         mesh::LocalFaceGeometryCompatible<Mesh>,
      "BoundaryFacets<Name> on MeshIntegrationDomain requires local face "
      "connectivity whose reference geometry matches "
      "Mesh::cell_type::geometry.");

   if constexpr (
      mesh::LocalFaceConnectivity<Mesh> &&
      mesh::LocalFaceGeometryCompatible<Mesh>)
   {
      using Geometry = mesh::mesh_geometry_t<Mesh>;
      ConstexprLoop< Geometry::num_faces >(
         [&](auto face_index)
         {
            auto face =
               mesh::GetLocalFaceInfo(mesh, element_index, face_index);

            if ( IsBoundaryFace( face ) )
            {
               boundary_lambda( face );
            }
         }
      );
   }
}

}
