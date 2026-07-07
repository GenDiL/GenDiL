// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/hexcell.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/linecell.hpp"
#include "gendil/Meshes/Cells/ReferenceCells/quadcell.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/types.hpp"

namespace gendil {

/**
 * @brief Cell-only unstructured line mesh.
 *
 * This mesh is intended for global-facet workflows where facet topology is
 * supplied by global face families. It intentionally does not expose local
 * facet connectivity.
 */
template < Integer order >
struct LineCellMesh
{
   StridedView< 1, const Real > nodes;
   HostDeviceStridedView< 2, const int > restriction;
   GlobalIndex num_elems;

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
};

/**
 * @brief Cell-only unstructured quad mesh.
 *
 * This mesh is intended for global-facet workflows where facet topology is
 * supplied by global face families. It intentionally does not expose local
 * facet connectivity.
 */
template < Integer order >
struct QuadCellMesh
{
   StridedView< 2, const Real > nodes;
   HostDeviceStridedView< 3, const int > restriction;
   GlobalIndex num_elems;

   static constexpr Integer Dim = 2;
   static constexpr Integer D1D = order + 1;
   using cell_type = QuadCell< D1D >;
   using cell_index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return num_elems;
   }

   GENDIL_HOST_DEVICE
   auto GetCell( GlobalIndex element_index ) const
   {
      return QuadCell< D1D >( nodes, restriction, element_index );
   }
};

/**
 * @brief Cell-only unstructured hex mesh.
 *
 * This mesh is intended for global-facet workflows where facet topology is
 * supplied by global face families. It intentionally does not expose local
 * facet connectivity.
 */
template < Integer order >
struct HexCellMesh
{
   StridedView< 2, const Real > nodes;
   HostDeviceStridedView< 4, const int > restriction;
   GlobalIndex num_elems;

   static constexpr Integer Dim = 3;
   static constexpr Integer D1D = order + 1;
   using cell_type = HexCell< D1D >;
   using cell_index_type = GlobalIndex;

   GENDIL_HOST_DEVICE
   Integer GetNumberOfCells() const
   {
      return num_elems;
   }

   GENDIL_HOST_DEVICE
   auto GetCell( GlobalIndex element_index ) const
   {
      return HexCell< D1D >( nodes, restriction, element_index );
   }
};

} // namespace gendil
