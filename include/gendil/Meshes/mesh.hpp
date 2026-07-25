// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Core mesh concepts, reference-geometry traits, and cell traversal.
 */

#include "gendil/Utilities/types.hpp"

#include <concepts>
#include <type_traits>
#include <utility>

namespace gendil {

namespace mesh {

/**
 * @brief Minimal interface for an indexable collection of cells.
 *
 * This concept intentionally requires only a cell count and indexed cell
 * access. It does not require a `cell_type`, reference geometry, local face
 * connectivity, or any particular cell return type. Algorithms needing those
 * capabilities should use one of the stronger concepts below.
 *
 * @tparam T Candidate mesh type.
 */
template <typename T>
concept Mesh =
   requires (const T& m, GlobalIndex ci)
   {
      // Number of cells:
      { m.GetNumberOfCells() } -> std::convertible_to<GlobalIndex>;

      // Cell accessor:
      { m.GetCell(ci) };
   };

/**
 * @brief Get the cell object corresponding to the cell index.
 * 
 * @tparam Mesh The mesh type.
 * @param mesh The mesh.
 * @param cell_index The linear index of the cell.
 * @return auto The cell object associated to the cell index.
 * 
 * @note See cell.hpp to see requirements on the Cell interface.
 */
template < Mesh Mesh >
GENDIL_HOST_DEVICE
auto GetCell( const Mesh & mesh, GlobalIndex cell_index )
{
   return mesh.GetCell( cell_index );
}

/**
 * @brief Return local connectivity information for one cell face.
 * 
 * @tparam Mesh The mesh type.
 * @tparam FaceID Type identifying a local face of a cell.
 * @param mesh The mesh.
 * @param cell_index The current cell index.
 * @param face_id The local face identifier.
 * @return The mesh-defined local face information.
 */
template < typename Mesh, typename FaceID >
GENDIL_HOST_DEVICE
auto GetLocalFaceInfo( const Mesh & mesh, GlobalIndex cell_index, const FaceID & face_id )
{
   return mesh.GetLocalFaceInfo( cell_index, face_id );
}

/**
 * @brief Applies @a body to all cells using @a KernelConfiguration.
 *
 * @details CellIterator is a mesh-level forwarding convenience. Production
 * BlockLoop owns device launch details and inactive final-batch filtering, and
 * invokes @a body only as body(GlobalIndex work_item_index).
 *
 * @tparam KernelConfiguration Kernel policy providing `BlockLoop`.
 * @tparam Mesh Mesh type satisfying the minimal `Mesh` concept.
 * @tparam Lambda Callable type.
 * @param mesh Mesh whose cells are traversed.
 * @param body Callable invoked with each valid global cell index.
 */
template < typename KernelConfiguration, Mesh Mesh, typename Lambda >
void CellIterator( const Mesh & mesh, Lambda && body )
{
   KernelConfiguration::BlockLoop(
      mesh.GetNumberOfCells(),
      std::forward< Lambda >( body ) );
}

/**
 * @brief Require a cell type to declare its reference geometry.
 *
 * The associated `Cell::geometry` type describes reference topology and has
 * no runtime storage requirement. This concept does not constrain the cell's
 * physical mapping representation.
 *
 * @tparam Cell Candidate cell type.
 */
template<class Cell>
concept CellWithGeometry =
   requires
   {
      typename std::remove_cvref_t<Cell>::geometry;
   };

/**
 * @brief Require a mesh to expose a cell type with reference geometry.
 *
 * This refines the minimal `Mesh` concept by requiring
 * `MeshType::cell_type::geometry`. It does not by itself require local face
 * connectivity or prove that two mesh objects have identical topology,
 * ordering, or physical geometry.
 *
 * @tparam MeshType Candidate mesh type.
 */
template<class MeshType>
concept MeshWithCellGeometry =
   Mesh<MeshType> &&
   requires
   {
      typename std::remove_cvref_t<MeshType>::cell_type;
      typename std::remove_cvref_t<
         typename std::remove_cvref_t<MeshType>::cell_type>::geometry;
   };

/**
 * @brief Extract a cell's reference-geometry type.
 * @tparam Cell Cell type satisfying `CellWithGeometry`.
 */
template<class Cell>
using cell_geometry_t =
   typename std::remove_cvref_t<Cell>::geometry;

/**
 * @brief Extract the cell type declared by a mesh.
 * @tparam MeshType Mesh type exposing `cell_type`.
 */
template<class MeshType>
using mesh_cell_t =
   typename std::remove_cvref_t<MeshType>::cell_type;

/**
 * @brief Extract the reference geometry of a mesh's declared cell type.
 * @tparam MeshType Mesh type satisfying `MeshWithCellGeometry`.
 */
template<class MeshType>
using mesh_geometry_t =
   cell_geometry_t<mesh_cell_t<MeshType>>;

namespace detail {

/** @internal */
template<class MeshType, size_t... FaceI>
consteval bool HasLocalFaceConnectivity(std::index_sequence<FaceI...>)
{
   return (
      requires (const MeshType& mesh, GlobalIndex cell_index)
      {
         mesh.GetLocalFaceInfo(
            cell_index,
            std::integral_constant<Integer, FaceI>{});
      } &&
      ...);
}

/** @internal */
template<class MeshType, size_t FaceI>
consteval bool LocalFaceGeometryMatches()
{
   if constexpr (
      requires (const MeshType& mesh, GlobalIndex cell_index)
      {
         mesh.GetLocalFaceInfo(
            cell_index,
            std::integral_constant<Integer, FaceI>{});
      })
   {
      using FaceInfo = std::remove_cvref_t<
         decltype(
            std::declval<const MeshType&>().GetLocalFaceInfo(
               GlobalIndex{},
               std::integral_constant<Integer, FaceI>{}))>;

      if constexpr (
         requires
         {
            typename FaceInfo::minus_side_type::geometry;
            typename FaceInfo::plus_side_type::geometry;
         })
      {
         using Geometry = mesh_geometry_t<MeshType>;
         return
            std::is_same_v<
               Geometry,
               typename FaceInfo::minus_side_type::geometry> &&
            std::is_same_v<
               Geometry,
               typename FaceInfo::plus_side_type::geometry>;
      }
   }

   return false;
}

/** @internal */
template<class MeshType, size_t... FaceI>
consteval bool HasCompatibleLocalFaceGeometry(
   std::index_sequence<FaceI...>)
{
   return (LocalFaceGeometryMatches<MeshType, FaceI>() && ...);
}

} // namespace detail

/**
 * @brief Require cell-owned connectivity for every local reference face.
 *
 * A mesh satisfies this concept when it has cell geometry and provides
 * `GetLocalFaceInfo(cell_index, face_id)` for every compile-time face index
 * from zero through `mesh_geometry_t<MeshType>::num_faces - 1`.
 *
 * This concept checks availability only. Use `LocalFaceGeometryCompatible`
 * when the returned face views must also advertise matching reference
 * geometry.
 *
 * @tparam MeshType Candidate mesh type.
 */
template<class MeshType>
concept LocalFaceConnectivity =
   MeshWithCellGeometry<MeshType> &&
   detail::HasLocalFaceConnectivity<MeshType>(
      std::make_index_sequence<
         mesh_geometry_t<MeshType>::num_faces>{});

/**
 * @brief Require local face views to match the mesh reference geometry.
 *
 * This refines `LocalFaceConnectivity`. For every local face, both
 * `minus_side_type::geometry` and `plus_side_type::geometry` in the returned
 * face information must exactly equal `mesh_geometry_t<MeshType>`.
 *
 * @tparam MeshType Candidate mesh type.
 */
template<class MeshType>
concept LocalFaceGeometryCompatible =
   LocalFaceConnectivity<MeshType> &&
   detail::HasCompatibleLocalFaceGeometry<MeshType>(
      std::make_index_sequence<
         mesh_geometry_t<MeshType>::num_faces>{});

} // namespace mesh

} // namespace gendil
