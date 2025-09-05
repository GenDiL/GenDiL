// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <vector>
#include <mfem.hpp>
#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/unstructuredconformingconnectivity.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "orientation.hpp"

namespace gendil {

/**
 * @brief Get the gendil local face index based on the MFEM local face index.
 * 
 * @note Only valid for HyperCube geometries.
 * 
 * @tparam Dim The dimension of the space.
 * @param mfem_local_face_index The MFEM local face index.
 * @return size_t The corresponding gendil local face index.
 */
template < Integer Dim >
size_t GetLocalFaceIndex( int mfem_local_face_index )
{
   size_t local_face_index = std::numeric_limits< size_t >::max();
   if constexpr ( Dim == 1 )
   {
      local_face_index = (size_t) mfem_local_face_index;
   }
   else if constexpr ( Dim == 2 )
   {
      switch ( mfem_local_face_index )
      {
      case 0: // BOTTOM
         local_face_index = 1;
         break;
      
      case 1: // RIGHT
         local_face_index = 2;
         break;

      case 2: // TOP
         local_face_index = 3;
         break;

      case 3: // LEFT
         local_face_index = 0;
         break;

      default:
         break;
      }
   }
   else if constexpr ( Dim == 3 )
   {
      switch ( mfem_local_face_index )
      {
      case 0: // BOTTOM
         local_face_index = 2;
         break;
      
      case 1: // FRONT
         local_face_index = 1;
         break;

      case 2: // RIGHT
         local_face_index = 3;
         break;

      case 3: // BACK
         local_face_index = 4;
         break;

      case 4: // LEFT
         local_face_index = 0;
         break;
      
      case 5: // TOP
         local_face_index = 5;
         break;
      
      default:
         break;
      }
   }
   return local_face_index;
}

// !FIXME inv_orientation is not correct for 1D and 2D
// TODO: Provide Geometry?
template < Integer Dim >
UnstructuredConformingConnectivity< HyperCube< Dim > > MakeMeshConnectivity( const mfem::Mesh & mesh )
{
   const Integer NE = mesh.GetNE();
   UnstructuredConformingConnectivity< HyperCube< Dim > > element_connectivities( NE );
   // !FIXME: Only valid for hex
   constexpr int inv_orientation[8] = { 0, 1, 6, 3, 4, 5, 2, 7 };
   const int num_faces = mesh.GetNumFaces();
   for (size_t face = 0; face < num_faces; face++)
   {
      const auto face_info = mesh.GetFaceInformation( face );
      if ( face_info.IsBoundary() )
      {
         const int elem_index = face_info.element[0].index;
         const int mfem_local_face_id = face_info.element[0].local_face_id;
         const size_t local_face_id = GetLocalFaceIndex< Dim >( mfem_local_face_id );
         const int orientation = face_info.element[0].orientation;
         element_connectivities[ elem_index ].faces[ local_face_id ].neighbor_index = std::numeric_limits< GlobalIndex >::quiet_NaN();
         element_connectivities[ elem_index ].faces[ local_face_id ].orientation =
            TranslateMFEMOrientation( mfem_orientation< Dim >{ mfem_local_face_id, -1, orientation } );
         element_connectivities[ elem_index ].faces[ local_face_id ].boundary = true;
      }
      else
      {
         const int elem_index_0 = face_info.element[0].index;
         const int mfem_local_face_id_0 = face_info.element[0].local_face_id;
         const size_t local_face_id_0 = GetLocalFaceIndex< Dim >( mfem_local_face_id_0 );
         const int elem_index_1 = face_info.element[1].index;
         const int mfem_local_face_id_1 = face_info.element[1].local_face_id;
         const size_t local_face_id_1 = GetLocalFaceIndex< Dim >( mfem_local_face_id_1 );
         const int orientation_1 = face_info.element[1].orientation;
         element_connectivities[ elem_index_0 ].faces[ local_face_id_0 ].neighbor_index = elem_index_1;
         element_connectivities[ elem_index_0 ].faces[ local_face_id_0 ].orientation =
            TranslateMFEMOrientation( mfem_orientation< Dim >{ mfem_local_face_id_0, mfem_local_face_id_1, orientation_1 } );
         element_connectivities[ elem_index_0 ].faces[ local_face_id_0 ].boundary = false;
         element_connectivities[ elem_index_1 ].faces[ local_face_id_1 ].neighbor_index = elem_index_0;
         element_connectivities[ elem_index_1 ].faces[ local_face_id_1 ].orientation =
            TranslateMFEMOrientation( mfem_orientation< Dim >{ mfem_local_face_id_1, mfem_local_face_id_0, inv_orientation[ orientation_1 ] } );
         element_connectivities[ elem_index_1 ].faces[ local_face_id_1 ].boundary = false;
      }
   }

   ToDevice( NE, element_connectivities.element_connectivities );

   return element_connectivities;
}

}

#endif
