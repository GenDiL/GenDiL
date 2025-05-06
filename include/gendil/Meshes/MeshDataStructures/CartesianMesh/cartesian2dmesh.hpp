// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/tensorindex.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/getstructuredsubindex.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/squarecell.hpp"
#include "gendil/Meshes/Connectivities/cartesianconnectivity.hpp"
#include "gendil/Meshes/Connectivities/periodiccartesianconnectivity.hpp"

namespace gendil {

/**
 * @brief A 2D structured cartesian mesh.
 * 
 */
template < typename Connectivity >
struct Cartesian2DMeshBase
{
   static constexpr Integer Dim = 2;
   using connectivity_type = Connectivity;
   using cell_type = SquareCell;

   const Real h_x, h_y; // size of the elements
   connectivity_type connectivity; // number of elements per dimension
   const Point< Dim > mesh_origin;

   Cartesian2DMeshBase( Real h,
                        Integer nx,
                        Integer ny ) :
      h_x( h ), h_y( h ),
      connectivity( nx, ny )
   { }

   Cartesian2DMeshBase( Real h,
                        Integer nx,
                        Integer ny,
                        const Point< Dim > & origin ) :
      h_x( h ), h_y( h ),
      connectivity( nx, ny ),
      mesh_origin( origin )
   { }

   Cartesian2DMeshBase( Real h_x,
                        Real h_y,
                        Integer nx,
                        Integer ny,
                        const Point< Dim > & origin ) :
      h_x( h_x ), h_y( h_y ),
      connectivity( nx, ny ),
      mesh_origin( origin )
   { }

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return connectivity.GetNumberOfCells();
   }

   GENDIL_HOST_DEVICE
   SquareCell GetCell( GlobalIndex element_index ) const
   {
      const auto [i, j] = GetStructuredSubIndices( element_index, connectivity.sizes );
      const Point< Dim > cell_origin
      {
         mesh_origin[0] + h_x * i,
         mesh_origin[1] + h_y * j
      };

      return SquareCell( cell_origin, h_x, h_y );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetFaceNeighborInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > face_index ) const
   {
      return connectivity( cell_index, face_index );
   }
};

using Cartesian2DMesh = Cartesian2DMeshBase< CartesianConnectivity< 2 > >;
using PeriodicCartesian2DMesh = Cartesian2DMeshBase< PeriodicCartesianConnectivity< 2 > >;

}
