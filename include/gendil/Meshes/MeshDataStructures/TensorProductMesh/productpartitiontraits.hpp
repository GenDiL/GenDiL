// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Cell-mesh recognition and uniform-dimension traits for Cartesian
 * product partitions.
 *
 * These traits validate the cell parts of one partition factor before the
 * product factory instantiates cell meshes or face-connectivity adapters.
 * They do not validate face relation indices, face geometry, conformity maps,
 * or connectivity; those checks remain with the construction that consumes
 * that information.
 */

#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/partition.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

namespace cartesian_product_partition_detail {

/**
 * @brief Classifies a cell mesh accepted by a Cartesian-product partition.
 *
 * A supported mesh provides a static `Dim`, a `cell_type`, and a
 * `cell_type::geometry`. The geometry must be a full-dimensional HyperCube
 * whose geometry dimension equals `Mesh::Dim`.
 *
 * The primary template deliberately reports `value == false` and `dim == 0`
 * when any required member is absent. This permits the factory to issue its
 * focused cell-part diagnostic instead of failing during member lookup.
 */
template<class Mesh, class = void>
struct IsSupportedProductPartitionCellMesh : std::false_type
{
   static constexpr Integer dim = 0;
};

/**
 * @brief Classification for meshes exposing the required static cell
 * metadata.
 *
 * Merely exposing the members is insufficient: the cell geometry must be a
 * HyperCube and its full dimension must agree with the mesh dimension.
 */
template<class Mesh>
struct IsSupportedProductPartitionCellMesh<
   Mesh,
   std::void_t<
      decltype(std::remove_cvref_t<Mesh>::Dim),
      typename std::remove_cvref_t<Mesh>::cell_type,
      typename std::remove_cvref_t<Mesh>::cell_type::geometry>>
   : std::bool_constant<
        is_hypercube_geometry<
           typename std::remove_cvref_t<Mesh>::cell_type::geometry>::value &&
        std::remove_cvref_t<Mesh>::Dim ==
           std::remove_cvref_t<Mesh>::cell_type::geometry::geometry_dim>
{
   static constexpr Integer dim = std::remove_cvref_t<Mesh>::Dim;
};

/**
 * @brief Check every cell part and require one uniform factor dimension.
 *
 * The first cell part establishes the expected dimension. Every cell mesh
 * must satisfy IsSupportedProductPartitionCellMesh and report that same
 * dimension. A valid Partition always contains at least one CellPart, as
 * enforced by MakePartition; this helper therefore intentionally indexes the
 * first tuple entry.
 */
template<class Partition, size_t... I>
consteval bool ProductPartitionCellsSupported(std::index_sequence<I...>)
{
   using Parts = typename Partition::cell_parts_type;
   using FirstMesh = typename std::tuple_element_t<0, Parts>::mesh_type;
   constexpr Integer Dim =
      IsSupportedProductPartitionCellMesh<FirstMesh>::dim;
   return (
      IsSupportedProductPartitionCellMesh<
         typename std::tuple_element_t<I, Parts>::mesh_type>::value &&
      ...) && (
      (IsSupportedProductPartitionCellMesh<
         typename std::tuple_element_t<I, Parts>::mesh_type>::dim == Dim) &&
      ...);
}

/**
 * @brief Whether all cell parts of one partition factor are supported and
 * share one reference dimension.
 *
 * This predicate covers factor cell meshes only. Face relations, source face
 * geometry, conformity maps, and concrete connectivity are validated by the
 * product-partition and product-face-connectivity implementations.
 */
template<class Partition>
inline constexpr bool product_partition_cells_supported_v =
   ProductPartitionCellsSupported<Partition>(
      std::make_index_sequence<Partition::num_cell_parts>{});

/**
 * @brief Uniform cell dimension of a supported product-partition factor.
 *
 * This value is meaningful only after
 * `product_partition_cells_supported_v<Partition>` is known to be true. The
 * factory establishes that precondition before using the dimension to form
 * product cell and face types.
 */
template<class Partition>
inline constexpr Integer product_partition_dim_v =
   IsSupportedProductPartitionCellMesh<
      typename std::tuple_element_t<
         0,
         typename Partition::cell_parts_type>::mesh_type>::dim;

} // namespace cartesian_product_partition_detail

} // namespace gendil
