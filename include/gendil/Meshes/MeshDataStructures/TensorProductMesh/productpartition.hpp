// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Construction of Cartesian products of explicitly described
 * partitions.
 *
 * This header owns factor validation orchestration, stable cell- and
 * face-part tuple construction, relation remapping, and the public factory.
 * Cell-mesh recognition lives in productpartitiontraits.hpp, while lazy face
 * extrusion and side lifting live in productfaceconnectivity.hpp.
 */

#include "gendil/prelude.hpp"
#include "gendil/Meshes/facemesh.hpp"
#include "gendil/Meshes/mesh.hpp"
#include "gendil/Meshes/partition.hpp"
#include "gendil/Meshes/MeshDataStructures/TensorProductMesh/productfaceconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/TensorProductMesh/productmesh.hpp"
#include "gendil/Meshes/MeshDataStructures/TensorProductMesh/productpartitiontraits.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

namespace cartesian_product_partition_detail {

/**
 * @brief Construct one product CellPart from a flattened pair of factor parts.
 *
 * `FlatI` decodes as
 * `FirstCellPartI = FlatI % NumFirstCellParts` and
 * `SecondCellPartI = FlatI / NumFirstCellParts`. Consequently the first
 * factor varies fastest and the product CellPart index is
 * `FirstCellPartI + NumFirstCellParts * SecondCellPartI`. The returned
 * CellPart owns the CartesianProductMesh produced from the two selected
 * factor meshes.
 */
template<size_t FlatI, class FirstPartition, class SecondPartition>
constexpr auto MakeProductCellPart(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr size_t NumFirstCellParts = FirstPartition::num_cell_parts;
   constexpr size_t FirstCellPartI = FlatI % NumFirstCellParts;
   constexpr size_t SecondCellPartI = FlatI / NumFirstCellParts;
   return MakeCellPart(
      MakeCartesianProductMesh(
         std::get<FirstCellPartI>(first.CellParts()).mesh,
         std::get<SecondCellPartI>(second.CellParts()).mesh));
}

/**
 * @brief Expand every flattened factor pair into the product cell-part tuple.
 *
 * The ascending index pack preserves the first-factor-fastest numbering used
 * by face relation remapping and by nested product cell indexing.
 */
template<class FirstPartition, class SecondPartition, size_t... I>
constexpr auto MakeProductCellParts(
   const FirstPartition& first,
   const SecondPartition& second,
   std::index_sequence<I...>)
{
   return std::tuple{MakeProductCellPart<I>(first, second)...};
}

/**
 * @brief Validate both sides of an interior source face against its referenced
 * cell meshes.
 *
 * The check compares the concrete source FaceView geometries with the indexed
 * cell geometries of the declared minus and plus CellParts. It is kept beside
 * relation construction because the referenced meshes are selected there.
 */
template<class FacePart, class MinusMesh, class PlusMesh>
consteval void ValidateInteriorSourceGeometry()
{
   using FaceMesh = typename FacePart::face_mesh_type;
   using Info = std::remove_cvref_t<
      decltype(
         std::declval<const FaceMesh&>().GetGlobalFaceInfo(GlobalIndex{}))>;
   static_assert(
      std::is_same_v<
         typename Info::minus_side_type::geometry,
         mesh::mesh_geometry_t<MinusMesh>> &&
      std::is_same_v<
         typename Info::plus_side_type::geometry,
         mesh::mesh_geometry_t<PlusMesh>>,
      "MakeCartesianProductPartition: interior face geometry is incompatible "
      "with its referenced cell parts.");
}

/**
 * @brief Validate the real side of a boundary source face against its
 * referenced cell mesh.
 *
 * Boundary dummy-side cell indices are connectivity data rather than
 * partition relations, so only the declared real minus side participates in
 * this geometry check.
 */
template<class FacePart, class CellMesh>
consteval void ValidateBoundarySourceGeometry()
{
   using FaceMesh = typename FacePart::face_mesh_type;
   using Info = std::remove_cvref_t<
      decltype(
         std::declval<const FaceMesh&>().GetGlobalFaceInfo(GlobalIndex{}))>;
   static_assert(
      std::is_same_v<
         typename Info::minus_side_type::geometry,
         mesh::mesh_geometry_t<CellMesh>>,
      "MakeCartesianProductPartition: boundary face geometry is incompatible "
      "with its referenced cell part.");
}

/**
 * @brief Construct an interior face part derived from the first factor.
 *
 * The flat family index is
 * `SourceFacePartI + NumFirstInteriorFaceParts * SecondCellPartI`:
 * second-factor CellParts are outer and first-factor face parts are inner.
 * Source relations `(MinusCellPartI, PlusCellPartI)` become
 * `(MinusCellPartI + NumFirstCellParts * SecondCellPartI,
 *   PlusCellPartI + NumFirstCellParts * SecondCellPartI)`.
 *
 * The lazy adapter owns the source connectivity by value and stores distinct
 * minus and plus source-cell counts. These separate strides are required when
 * the referenced minus and plus CellParts have different cell counts.
 */
template<size_t FlatI, class FirstPartition, class SecondPartition>
constexpr auto MakeFirstFactorInteriorFacePart(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr size_t NumFirstInteriorFaceParts =
      FirstPartition::num_interior_face_parts;
   constexpr size_t SourceFacePartI =
      FlatI % NumFirstInteriorFaceParts;
   constexpr size_t SecondCellPartI =
      FlatI / NumFirstInteriorFaceParts;
   const auto& face_part =
      std::get<SourceFacePartI>(first.InteriorFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t MinusCellPartI = FacePart::minus_cell_index;
   constexpr size_t PlusCellPartI = FacePart::plus_cell_index;
   const auto& minus_mesh =
      std::get<MinusCellPartI>(first.CellParts()).mesh;
   const auto& plus_mesh =
      std::get<PlusCellPartI>(first.CellParts()).mesh;
   const auto& second_mesh =
      std::get<SecondCellPartI>(second.CellParts()).mesh;
   using MinusMesh = std::remove_cvref_t<decltype(minus_mesh)>;
   using PlusMesh = std::remove_cvref_t<decltype(plus_mesh)>;
   ValidateInteriorSourceGeometry<FacePart, MinusMesh, PlusMesh>();

   using Connectivity = FirstFactorInteriorFaceConnectivity<
      product_partition_dim_v<FirstPartition>,
      product_partition_dim_v<SecondPartition>,
      typename FacePart::face_mesh_type>;
   constexpr size_t ProductMinusCellPartI =
      MinusCellPartI +
      FirstPartition::num_cell_parts * SecondCellPartI;
   constexpr size_t ProductPlusCellPartI =
      PlusCellPartI +
      FirstPartition::num_cell_parts * SecondCellPartI;
   const GlobalIndex minus_side_stride = minus_mesh.GetNumberOfCells();
   const GlobalIndex plus_side_stride = plus_mesh.GetNumberOfCells();
   const GlobalIndex num_extruded_cells = second_mesh.GetNumberOfCells();
   return MakeInteriorFacePart<
      ProductMinusCellPartI,
      ProductPlusCellPartI>(
      Connectivity{
         face_part.face_mesh,
         minus_side_stride,
         plus_side_stride,
         num_extruded_cells});
}

/**
 * @brief Construct an interior face part derived from the second factor.
 *
 * The flat family index is
 * `FirstCellPartI + NumFirstCellParts * SourceFacePartI`: second-factor face
 * parts are outer and first-factor CellParts are inner. Source relations
 * `(MinusCellPartI, PlusCellPartI)` become
 * `(FirstCellPartI + NumFirstCellParts * MinusCellPartI,
 *   FirstCellPartI + NumFirstCellParts * PlusCellPartI)`.
 *
 * Runtime product-cell indices use the selected first-factor mesh count as
 * the stride for each independently read source-side cell index.
 */
template<size_t FlatI, class FirstPartition, class SecondPartition>
constexpr auto MakeSecondFactorInteriorFacePart(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr size_t NumFirstCellParts = FirstPartition::num_cell_parts;
   constexpr size_t FirstCellPartI = FlatI % NumFirstCellParts;
   constexpr size_t SourceFacePartI = FlatI / NumFirstCellParts;
   const auto& face_part =
      std::get<SourceFacePartI>(second.InteriorFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t MinusCellPartI = FacePart::minus_cell_index;
   constexpr size_t PlusCellPartI = FacePart::plus_cell_index;
   const auto& first_mesh =
      std::get<FirstCellPartI>(first.CellParts()).mesh;
   const auto& minus_mesh =
      std::get<MinusCellPartI>(second.CellParts()).mesh;
   const auto& plus_mesh =
      std::get<PlusCellPartI>(second.CellParts()).mesh;
   using MinusMesh = std::remove_cvref_t<decltype(minus_mesh)>;
   using PlusMesh = std::remove_cvref_t<decltype(plus_mesh)>;
   ValidateInteriorSourceGeometry<FacePart, MinusMesh, PlusMesh>();

   using Connectivity = SecondFactorInteriorFaceConnectivity<
      product_partition_dim_v<FirstPartition>,
      product_partition_dim_v<SecondPartition>,
      typename FacePart::face_mesh_type>;
   constexpr size_t ProductMinusCellPartI =
      FirstCellPartI + NumFirstCellParts * MinusCellPartI;
   constexpr size_t ProductPlusCellPartI =
      FirstCellPartI + NumFirstCellParts * PlusCellPartI;
   const GlobalIndex num_first_cells = first_mesh.GetNumberOfCells();
   const GlobalIndex minus_side_stride = num_first_cells;
   const GlobalIndex plus_side_stride = num_first_cells;
   const GlobalIndex num_extruded_cells = num_first_cells;
   return MakeInteriorFacePart<
      ProductMinusCellPartI,
      ProductPlusCellPartI>(
      Connectivity{
         face_part.face_mesh,
         minus_side_stride,
         plus_side_stride,
         num_extruded_cells});
}

/**
 * @brief Build all first-factor-derived interior families in stable order.
 *
 * An explicitly declared source family participates even when its runtime face
 * count is zero, preserving zero-sized topology in the product Partition.
 */
template<class FirstPartition, class SecondPartition, size_t... I>
constexpr auto MakeFirstFactorInteriorFaceParts(
   const FirstPartition& first,
   const SecondPartition& second,
   std::index_sequence<I...>)
{
   return std::tuple{
      MakeFirstFactorInteriorFacePart<I>(first, second)...};
}

/**
 * @brief Build all second-factor-derived interior families in stable order.
 *
 * Only declared source families are expanded; local mesh connectivity is never
 * inspected to infer missing global topology.
 */
template<class FirstPartition, class SecondPartition, size_t... I>
constexpr auto MakeSecondFactorInteriorFaceParts(
   const FirstPartition& first,
   const SecondPartition& second,
   std::index_sequence<I...>)
{
   return std::tuple{
      MakeSecondFactorInteriorFacePart<I>(first, second)...};
}

/**
 * @brief Construct a boundary face part derived from the first factor.
 *
 * Families use the same
 * `SourceFacePartI + NumFirstBoundaryFaceParts * SecondCellPartI` ordering as
 * first-factor-derived interior faces. The declared real CellPart maps to
 * `SourceCellPartI + NumFirstCellParts * SecondCellPartI`. The boundary
 * adapter tensorizes only that real minus side and preserves the source dummy
 * plus cell index.
 */
template<size_t FlatI, class FirstPartition, class SecondPartition>
constexpr auto MakeFirstFactorBoundaryFacePart(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr size_t NumFirstBoundaryFaceParts =
      FirstPartition::num_boundary_face_parts;
   constexpr size_t SourceFacePartI =
      FlatI % NumFirstBoundaryFaceParts;
   constexpr size_t SecondCellPartI =
      FlatI / NumFirstBoundaryFaceParts;
   const auto& face_part =
      std::get<SourceFacePartI>(first.BoundaryFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t SourceCellPartI = FacePart::cell_index;
   const auto& cell_mesh =
      std::get<SourceCellPartI>(first.CellParts()).mesh;
   const auto& second_mesh =
      std::get<SecondCellPartI>(second.CellParts()).mesh;
   using CellMesh = std::remove_cvref_t<decltype(cell_mesh)>;
   ValidateBoundarySourceGeometry<FacePart, CellMesh>();

   using Connectivity = FirstFactorBoundaryFaceConnectivity<
      product_partition_dim_v<FirstPartition>,
      product_partition_dim_v<SecondPartition>,
      typename FacePart::face_mesh_type>;
   constexpr size_t ProductCellPartI =
      SourceCellPartI +
      FirstPartition::num_cell_parts * SecondCellPartI;
   const GlobalIndex real_side_stride = cell_mesh.GetNumberOfCells();
   const GlobalIndex num_extruded_cells = second_mesh.GetNumberOfCells();
   return MakeBoundaryFacePart<ProductCellPartI>(
      Connectivity{
         face_part.face_mesh,
         real_side_stride,
         num_extruded_cells});
}

/**
 * @brief Construct a boundary face part derived from the second factor.
 *
 * Families use
 * `FirstCellPartI + NumFirstCellParts * SourceFacePartI`. The declared real
 * CellPart maps to
 * `FirstCellPartI + NumFirstCellParts * SourceCellPartI`; dummy-side
 * preservation remains the responsibility of the boundary connectivity.
 */
template<size_t FlatI, class FirstPartition, class SecondPartition>
constexpr auto MakeSecondFactorBoundaryFacePart(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr size_t NumFirstCellParts = FirstPartition::num_cell_parts;
   constexpr size_t FirstCellPartI = FlatI % NumFirstCellParts;
   constexpr size_t SourceFacePartI = FlatI / NumFirstCellParts;
   const auto& face_part =
      std::get<SourceFacePartI>(second.BoundaryFaceParts());
   using FacePart = std::remove_cvref_t<decltype(face_part)>;
   constexpr size_t SourceCellPartI = FacePart::cell_index;
   const auto& first_mesh =
      std::get<FirstCellPartI>(first.CellParts()).mesh;
   const auto& cell_mesh =
      std::get<SourceCellPartI>(second.CellParts()).mesh;
   using CellMesh = std::remove_cvref_t<decltype(cell_mesh)>;
   ValidateBoundarySourceGeometry<FacePart, CellMesh>();

   using Connectivity = SecondFactorBoundaryFaceConnectivity<
      product_partition_dim_v<FirstPartition>,
      product_partition_dim_v<SecondPartition>,
      typename FacePart::face_mesh_type>;
   constexpr size_t ProductCellPartI =
      FirstCellPartI + NumFirstCellParts * SourceCellPartI;
   const GlobalIndex num_first_cells = first_mesh.GetNumberOfCells();
   const GlobalIndex real_side_stride = num_first_cells;
   const GlobalIndex num_extruded_cells = num_first_cells;
   return MakeBoundaryFacePart<ProductCellPartI>(
      Connectivity{
         face_part.face_mesh,
         real_side_stride,
         num_extruded_cells});
}

/**
 * @brief Build all first-factor-derived boundary families in stable order.
 *
 * Declared zero-face boundary families remain represented by zero-sized lazy
 * adapters.
 */
template<class FirstPartition, class SecondPartition, size_t... I>
constexpr auto MakeFirstFactorBoundaryFaceParts(
   const FirstPartition& first,
   const SecondPartition& second,
   std::index_sequence<I...>)
{
   return std::tuple{
      MakeFirstFactorBoundaryFacePart<I>(first, second)...};
}

/**
 * @brief Build all second-factor-derived boundary families in stable order.
 *
 * No boundary family is synthesized from local mesh connectivity.
 */
template<class FirstPartition, class SecondPartition, size_t... I>
constexpr auto MakeSecondFactorBoundaryFaceParts(
   const FirstPartition& first,
   const SecondPartition& second,
   std::index_sequence<I...>)
{
   return std::tuple{
      MakeSecondFactorBoundaryFacePart<I>(first, second)...};
}

/**
 * @brief Validate two partition factors and assemble their concrete product
 * Partition.
 *
 * Cell validation establishes one supported HyperCube dimension per factor.
 * Relation validation ensures every declared interior and boundary face part
 * references existing CellParts before any relation-dependent template is
 * instantiated. Source face geometry is then checked while each family is
 * built. Concrete source records remain trusted: the factory does not scan
 * runtime connectivity for duplicate records or out-of-range cell indices.
 *
 * The result owns its product cell meshes and owns each lazy adapter, which in
 * turn owns its source connectivity. This value ownership makes direct nested
 * calls safe when an inner product Partition is a temporary. Cell numbering
 * flattens factors associatively with the earliest factor varying fastest;
 * face-family tuple positions may differ between `(P x Q) x R` and
 * `P x (Q x R)` because families are grouped by the immediate binary factors.
 */
template<class FirstPartition, class SecondPartition>
constexpr auto MakeCartesianProductPartitionImpl(
   const FirstPartition& first,
   const SecondPartition& second)
{
   constexpr bool CellsSupported =
      product_partition_cells_supported_v<FirstPartition> &&
      product_partition_cells_supported_v<SecondPartition>;
   static_assert(
      CellsSupported,
      "MakeCartesianProductPartition: every cell part in one factor must "
      "have the same dimension and use a full-dimensional HyperCube cell "
      "geometry.");
   constexpr bool RelationsValid =
      partition_detail::all_partition_relation_indices_in_range_v<
         FirstPartition::num_cell_parts,
         typename FirstPartition::interior_face_parts_type> &&
      partition_detail::all_partition_relation_indices_in_range_v<
         FirstPartition::num_cell_parts,
         typename FirstPartition::boundary_face_parts_type> &&
      partition_detail::all_partition_relation_indices_in_range_v<
         SecondPartition::num_cell_parts,
         typename SecondPartition::interior_face_parts_type> &&
      partition_detail::all_partition_relation_indices_in_range_v<
         SecondPartition::num_cell_parts,
         typename SecondPartition::boundary_face_parts_type>;
   static_assert(
      RelationsValid,
      "MakeCartesianProductPartition: input face relation indices must refer "
      "to valid cell parts.");

   if constexpr (CellsSupported && RelationsValid)
   {
      constexpr size_t NumProductCellParts =
         FirstPartition::num_cell_parts *
         SecondPartition::num_cell_parts;
      constexpr size_t NumFirstFactorInteriorFaceParts =
         FirstPartition::num_interior_face_parts *
         SecondPartition::num_cell_parts;
      constexpr size_t NumSecondFactorInteriorFaceParts =
         FirstPartition::num_cell_parts *
         SecondPartition::num_interior_face_parts;
      constexpr size_t NumFirstFactorBoundaryFaceParts =
         FirstPartition::num_boundary_face_parts *
         SecondPartition::num_cell_parts;
      constexpr size_t NumSecondFactorBoundaryFaceParts =
         FirstPartition::num_cell_parts *
         SecondPartition::num_boundary_face_parts;

      auto cell_parts = MakeProductCellParts(
         first,
         second,
         std::make_index_sequence<NumProductCellParts>{});
      auto first_factor_interior_parts =
         MakeFirstFactorInteriorFaceParts(
            first,
            second,
            std::make_index_sequence<NumFirstFactorInteriorFaceParts>{});
      auto second_factor_interior_parts =
         MakeSecondFactorInteriorFaceParts(
            first,
            second,
            std::make_index_sequence<NumSecondFactorInteriorFaceParts>{});
      auto first_factor_boundary_parts =
         MakeFirstFactorBoundaryFaceParts(
            first,
            second,
            std::make_index_sequence<NumFirstFactorBoundaryFaceParts>{});
      auto second_factor_boundary_parts =
         MakeSecondFactorBoundaryFaceParts(
            first,
            second,
            std::make_index_sequence<NumSecondFactorBoundaryFaceParts>{});

      return MakePartition(
         std::move(cell_parts),
         std::tuple_cat(
            std::move(first_factor_interior_parts),
            std::move(second_factor_interior_parts)),
         std::tuple_cat(
            std::move(first_factor_boundary_parts),
            std::move(second_factor_boundary_parts)));
   }
   else
   {
      return Empty{};
   }
}

} // namespace cartesian_product_partition_detail

/**
 * @brief Form the Cartesian product of two explicitly described partitions.
 *
 * @tparam FirstPartition First Partition type. Every CellPart must use the same
 * full-dimensional HyperCube cell dimension within this factor.
 * @tparam SecondPartition Second Partition type, subject to the same cell-part
 * requirements independently of the first factor's dimension.
 * @param first First factor retained through product cell meshes and by-value
 * lazy face adapters.
 * @param second Second factor retained in the same manner.
 * @return A regular Partition whose CellParts contain CartesianProductMesh
 * values and whose declared face parts contain lazy Cartesian-product
 * connectivity adapters.
 *
 * Product CellPart `(i,j)` has index
 * `i + FirstPartition::num_cell_parts * j`, so the first factor varies fastest.
 * Interior and boundary families derived from `first` precede those derived
 * from `second`. Within those groups, first-factor-derived families use the
 * second-factor CellPart as the outer index, while second-factor-derived
 * families use the second-factor face part as the outer index.
 *
 * Every explicitly declared face family is extruded, including zero-sized
 * families. Undeclared topology is never inferred from local mesh
 * connectivity. Boundary adapters tensorize only their real side and retain
 * source dummy-side indices. Interior adapters preserve distinct minus and
 * plus cell-count strides. Input partitions remain responsible for valid,
 * nonduplicated runtime connectivity records and in-range record cell indices;
 * the factory validates compile-time relations and geometry types but does not
 * scan the source records.
 *
 * Product parts are reconstructed from the mesh or connectivity values and
 * compile-time relation indices represented by the current part types. If
 * part abstractions later acquire labels, markers, or other metadata, their
 * Cartesian-product combination rules must be defined explicitly.
 *
 * The factory builds topology only. It does not construct product finite
 * elements, restrictions, mixed spaces, or sparse-assembly support. Nested
 * binary products are supported; flattened cell numbering is associative,
 * although face-family tuple positions may reflect the association tree.
 *
 * Compilation fails with focused diagnostics when either argument is not a
 * Partition, factor cell meshes are unsupported or dimensionally nonuniform,
 * relation indices are invalid, source face geometry is incompatible, or the
 * face adapters reject unsupported static topology, maps, or orientations.
 */
template<class FirstPartition, class SecondPartition>
constexpr auto MakeCartesianProductPartition(
   const FirstPartition& first,
   const SecondPartition& second)
{
   static_assert(
      is_partition_v<FirstPartition> && is_partition_v<SecondPartition>,
      "MakeCartesianProductPartition requires two Partition arguments.");

   if constexpr (
      is_partition_v<FirstPartition> && is_partition_v<SecondPartition>)
   {
      return cartesian_product_partition_detail::
         MakeCartesianProductPartitionImpl(first, second);
   }
   else
   {
      return Empty{};
   }
}

} // namespace gendil
