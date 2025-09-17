// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Meshes/Geometries/canonicalvector.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/linecell.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/unstructuredconformingconnectivity.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"

namespace gendil {

/**
 * @brief An unstructured segment mesh using continuous isoparametric representation.
 * 
 * @tparam order The order of the isoparametric transformation mapping reference elements to physical elements.
 */
template < Integer order >
struct LineMesh
{
   StridedView<1, const Real> nodes; // Should we just use an array?
   HostDeviceStridedView<2, const int> restriction;
   UnstructuredConformingConnectivity< HyperCube< 1 > > connectivity;
   GlobalIndex num_elems; // Total number of elements

   static constexpr Integer Dim = 1;
   static constexpr Integer D1D = order + 1;
   using cell_type = LineCell< D1D >;
   using cell_index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return num_elems;
   }

   GENDIL_HOST_DEVICE
   auto GetCell( GlobalIndex element_index ) const
   {
      return LineCell< D1D >( nodes, restriction, element_index );
   }

   template < Integer FaceIndex >
   GENDIL_HOST_DEVICE
   auto GetLocalFaceInfo( GlobalIndex cell_index, std::integral_constant< Integer, FaceIndex > face_index ) const
   {
      return connectivity.GetLocalFaceInfo( cell_index, face_index );
   }
};

}
