// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/cubecell.hpp"
#include "gendil/Meshes/Connectivities/cartesianconnectivity.hpp"
#include "gendil/Meshes/Connectivities/periodiccartesianconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/emptyhalo.hpp"

namespace gendil {

/**
 * @brief A 3D structured cartesian mesh.
 * 
 */
template < typename Connectivity >
struct Cartesian3DMeshBase
{
   static constexpr Integer Dim = 3;
   using connectivity_type = Connectivity;
   using cell_type = CubeCell;
   using halo_type = EmptyHalo<Dim>;

   const Real h_x, h_y, h_z; // size of the elements
   connectivity_type connectivity; // number of elements per dimension
   const Point< Dim > mesh_origin;

   Cartesian3DMeshBase( Real h, Integer nx, Integer ny, Integer nz ) :
      h_x( h ), h_y( h ), h_z( h ),
      connectivity( nx, ny, nz )
   { }

   Cartesian3DMeshBase( Real h_x,
                        Real h_y,
                        Real h_z ,
                        Integer nx,
                        Integer ny,
                        Integer nz,
                        const Point< Dim > & origin ) :
      h_x( h_x ), h_y( h_y ), h_z( h_z ),
      connectivity( nx, ny, nz ),
      mesh_origin( origin )
   { }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return connectivity.GetNumberOfCells();
   }

   GENDIL_HOST_DEVICE
   CubeCell GetCell( GlobalIndex element_index ) const
   {
      const auto [i, j, k] = GetStructuredSubIndices( element_index, connectivity.sizes );

      const Point< Dim > cell_origin
      {
         mesh_origin[0] + h_x * i,
         mesh_origin[1] + h_y * j,
         mesh_origin[2] + h_z * k
      };
      return CubeCell( cell_origin, h_x, h_y, h_z );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetFaceNeighborInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > face_index ) const
   {
      return connectivity( cell_index, face_index );
   }
};

using Cartesian3DMesh = Cartesian3DMeshBase< CartesianConnectivity< 3 > >;
using PeriodicCartesian3DMesh = Cartesian3DMeshBase< PeriodicCartesianConnectivity< 3 > >;

}
