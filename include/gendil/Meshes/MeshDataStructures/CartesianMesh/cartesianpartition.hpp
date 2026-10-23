// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Reusable conforming partitions for structured Cartesian meshes.
 */

#include "gendil/Meshes/partition.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesian1dmesh.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianndmesh.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianglobalfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianglobalboundaryfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/CartesianMesh/cartesianintermeshfaceconnectivity.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace gendil {

/**
 * @brief Describe a nonperiodic Cartesian mesh as a one-cell-part partition.
 *
 * Interior and boundary families are explicit, which makes the result suitable
 * for both matrix-free partition traversal and sparse assembly.
 */
template<Integer Dim>
auto MakeCartesianPartition(const CartesianMesh<Dim>& mesh)
{
   for (Integer axis = 0; axis < Dim; ++axis)
   {
      GENDIL_VERIFY(
         mesh.sizes[axis] > 0,
         "MakeCartesianPartition requires positive mesh extents." );
      GENDIL_VERIFY(
         !mesh.is_periodic_boundary[axis],
         "MakeCartesianPartition currently requires nonperiodic axes." );
   }
   return MakePartition(
      MakeCellPart(mesh),
      MakeInteriorFacePart<0, 0>(
         MakeCartesianInteriorFaceConnectivity<Dim>(mesh.sizes)),
      MakeBoundaryFacePart<0>(
         MakeCartesianBoundaryFaceConnectivity<Dim>(mesh.sizes)));
}

namespace cartesian_partition_detail {

template<size_t Part, size_t NumParts>
auto Make1DCellPart(
   const Real spacing,
   const std::array<GlobalIndex, NumParts>& elements_per_part,
   const Real origin)
{
   GlobalIndex preceding_elements = 0;
   for (size_t i = 0; i < Part; ++i)
   {
      preceding_elements = CheckedAdd(
         preceding_elements,
         elements_per_part[i],
         "Cartesian 1D partition element offset overflow." );
   }
   return MakeCellPart(Cartesian1DMesh{
      spacing,
      static_cast<Integer>(elements_per_part[Part]),
      origin + spacing * static_cast<Real>(preceding_elements)});
}

template<size_t NumParts, size_t... Part>
auto Make1DCellParts(
   const Real spacing,
   const std::array<GlobalIndex, NumParts>& elements_per_part,
   const Real origin,
   std::index_sequence<Part...>)
{
   return std::tuple{
      Make1DCellPart<Part>(spacing, elements_per_part, origin)...};
}

template<size_t NumParts, size_t... Part>
auto Make1DInteriorParts(
   const std::array<GlobalIndex, NumParts>& elements_per_part,
   std::index_sequence<Part...>)
{
   return std::tuple_cat(
      MakeInteriorFacePart<Part, Part>(
         MakeCartesianInteriorFaceConnectivity<1>(
            std::array<GlobalIndex, 1>{elements_per_part[Part]}))...);
}

template<size_t NumParts, size_t... Interface>
auto Make1DInterfaceParts(
   const std::array<GlobalIndex, NumParts>& elements_per_part,
   std::index_sequence<Interface...>)
{
   return std::tuple{
      MakeInteriorFacePart<Interface, Interface + 1>(
         CartesianIntermeshFaceConnectivity<1, 1>{
            std::array<GlobalIndex, 1>{elements_per_part[Interface]},
            std::array<GlobalIndex, 1>{elements_per_part[Interface + 1]}})...};
}

} // namespace cartesian_partition_detail

/**
 * @brief Split one nonperiodic Cartesian interval into contiguous cell parts.
 *
 * Part indices increase from the physical origin. Every part contributes one
 * same-part interior family, adjacent parts contribute one cross-part family,
 * and only the first and last physical endpoints are boundary families.
 */
template<size_t NumParts>
auto MakeCartesian1DPartition(
   const Real spacing,
   const std::array<GlobalIndex, NumParts>& elements_per_part,
   const Real origin = 0.0)
{
   static_assert(
      NumParts > 0,
      "MakeCartesian1DPartition requires at least one part." );
   GENDIL_VERIFY(
      spacing > 0.0,
      "MakeCartesian1DPartition requires positive spacing." );
   GlobalIndex total_elements = 0;
   for (const GlobalIndex size : elements_per_part)
   {
      GENDIL_VERIFY(
         size > 0,
         "MakeCartesian1DPartition requires every part to be nonempty." );
      total_elements = CheckedAdd(
         total_elements,
         size,
         "Cartesian 1D partition total extent overflow." );
   }
   (void)total_elements;

   auto cell_parts = cartesian_partition_detail::Make1DCellParts(
      spacing,
      elements_per_part,
      origin,
      std::make_index_sequence<NumParts>{});
   auto local_interiors = cartesian_partition_detail::Make1DInteriorParts(
      elements_per_part,
      std::make_index_sequence<NumParts>{});
   auto interfaces = cartesian_partition_detail::Make1DInterfaceParts(
      elements_per_part,
      std::make_index_sequence<NumParts - 1>{});
   return MakePartition(
      std::move(cell_parts),
      std::tuple_cat(std::move(local_interiors), std::move(interfaces)),
      std::tuple{
         MakeBoundaryFacePart<0>(
            CartesianBoundaryFaceConnectivity<1, 0>{
               std::array<GlobalIndex, 1>{elements_per_part.front()}}),
         MakeBoundaryFacePart<NumParts - 1>(
            CartesianBoundaryFaceConnectivity<1, 1>{
               std::array<GlobalIndex, 1>{elements_per_part.back()}})});
}

} // namespace gendil
