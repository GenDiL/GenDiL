// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/segmentcell.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianlocalfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/periodiccartesianconnectivity.hpp"

namespace gendil {

// TODO: Generalize into CartesianMesh< Dim >
/**
 * @brief A 1D structured cartesian mesh.
 * 
 */
template < typename Connectivity >
struct Cartesian1DMeshBase
{
   static constexpr Integer Dim = 1;
   using cell_type = SegmentCell;
   using connectivity_type = Connectivity;

   const Real h; // size of the elements
   connectivity_type connectivity; // number of elements per dimension
   const Real mesh_origin; // Origin of the mesh.

   Cartesian1DMeshBase( Real h, Integer n, Real origin = 0.0) :
      h( h ),
      connectivity( n ),
      mesh_origin( origin )
   { }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return connectivity.GetNumberOfCells();
   }

   GENDIL_HOST_DEVICE
   cell_type GetCell( GlobalIndex element_index ) const
   {
      const Real cell_origin = mesh_origin + h * element_index;
      return SegmentCell( cell_origin, h );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > face_index ) const
   {
      return connectivity.GetLocalFaceInfo( cell_index, face_index );
   }
};

using Cartesian1DMesh = Cartesian1DMeshBase< CartesianLocalFaceConnectivity< 1 > >;
using PeriodicCartesian1DMesh = Cartesian1DMeshBase< PeriodicCartesianConnectivity< 1 > >;

}
