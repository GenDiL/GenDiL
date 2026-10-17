// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/Utilities/staticstring.hpp"

#include <type_traits>

namespace gendil {

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned = false>
struct SelectedCellExecutionDomain
{
   static_assert(
      mesh::MeshWithCellGeometry<CellMesh>,
      "SelectedCellExecutionDomain requires a cell mesh with "
      "cell_type::geometry.");

   static constexpr auto domain_name = Name;
   static constexpr size_t cell_batch_index = CellI;

   CellMesh cell_mesh;

   GENDIL_HOST_DEVICE
   constexpr const CellMesh& GetCellMesh() const
   {
      return cell_mesh;
   }
};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class CellMesh>
struct SelectedBoundaryFaceExecutionDomain
{
   using face_part_type = FacePart;
   using face_mesh_type = typename FacePart::face_mesh_type;
   using cell_mesh_type = CellMesh;
   using face_info_type = std::remove_cvref_t<
      decltype(
         std::declval<const face_mesh_type&>().GetGlobalFaceInfo(
            GlobalIndex{}))>;

   static_assert(
      mesh::MeshWithCellGeometry<CellMesh>,
      "SelectedBoundaryFaceExecutionDomain requires a cell mesh with "
      "cell_type::geometry.");
   static_assert(
      std::is_same_v<
         typename face_info_type::minus_side_type::geometry,
         mesh::mesh_geometry_t<CellMesh>>,
      "SelectedBoundaryFaceExecutionDomain: boundary face reference "
      "geometry must match the referenced cell mesh geometry.");

   static constexpr auto domain_name = Name;
   static constexpr size_t face_batch_index = FaceI;
   static constexpr size_t cell_part_index = FacePart::cell_index;

   FacePart face_part;
   CellMesh cell_mesh;

   GENDIL_HOST_DEVICE
   constexpr const FacePart& GetBoundaryFacePart() const
   {
      return face_part;
   }

   GENDIL_HOST_DEVICE
   constexpr const face_mesh_type& GetFaceMesh() const
   {
      return face_part.face_mesh;
   }

   GENDIL_HOST_DEVICE
   constexpr const CellMesh& GetCellMesh() const
   {
      return cell_mesh;
   }
};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellMesh,
   class PlusCellMesh>
struct SelectedInteriorFaceExecutionDomain
{
   using face_part_type = FacePart;
   using face_mesh_type = typename FacePart::face_mesh_type;
   using minus_cell_mesh_type = MinusCellMesh;
   using plus_cell_mesh_type = PlusCellMesh;
   using face_info_type = std::remove_cvref_t<
      decltype(
         std::declval<const face_mesh_type&>().GetGlobalFaceInfo(
            GlobalIndex{}))>;

   static_assert(
      mesh::MeshWithCellGeometry<MinusCellMesh> &&
         mesh::MeshWithCellGeometry<PlusCellMesh>,
      "SelectedInteriorFaceExecutionDomain requires cell meshes with "
      "cell_type::geometry.");
   static_assert(
      std::is_same_v<
         typename face_info_type::minus_side_type::geometry,
         mesh::mesh_geometry_t<MinusCellMesh>>,
      "SelectedInteriorFaceExecutionDomain: minus face reference geometry "
      "must match the referenced minus cell mesh geometry.");
   static_assert(
      std::is_same_v<
         typename face_info_type::plus_side_type::geometry,
         mesh::mesh_geometry_t<PlusCellMesh>>,
      "SelectedInteriorFaceExecutionDomain: plus face reference geometry "
      "must match the referenced plus cell mesh geometry.");

   static constexpr auto domain_name = Name;
   static constexpr size_t face_batch_index = FaceI;
   static constexpr size_t minus_cell_part_index = FacePart::minus_cell_index;
   static constexpr size_t plus_cell_part_index = FacePart::plus_cell_index;

   FacePart face_part;
   MinusCellMesh minus_cell_mesh;
   PlusCellMesh plus_cell_mesh;

   GENDIL_HOST_DEVICE
   constexpr const FacePart& GetInteriorFacePart() const
   {
      return face_part;
   }

   GENDIL_HOST_DEVICE
   constexpr const face_mesh_type& GetFaceMesh() const
   {
      return face_part.face_mesh;
   }

   GENDIL_HOST_DEVICE
   constexpr const MinusCellMesh& GetMinusCellMesh() const
   {
      return minus_cell_mesh;
   }

   GENDIL_HOST_DEVICE
   constexpr const PlusCellMesh& GetPlusCellMesh() const
   {
      return plus_cell_mesh;
   }
};

template<class T>
struct is_partitioned_selected_execution_domain : std::false_type {};

template<
   StaticString Name,
   size_t CellI,
   class CellMesh,
   bool Partitioned>
struct is_partitioned_selected_execution_domain<
   SelectedCellExecutionDomain<Name, CellI, CellMesh, Partitioned>>
   : std::bool_constant<Partitioned> {};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class CellMesh>
struct is_partitioned_selected_execution_domain<
   SelectedBoundaryFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      CellMesh>> : std::true_type {};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellMesh,
   class PlusCellMesh>
struct is_partitioned_selected_execution_domain<
   SelectedInteriorFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      MinusCellMesh,
      PlusCellMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_partitioned_selected_execution_domain_v =
   is_partitioned_selected_execution_domain<
      std::remove_cvref_t<T>>::value;

template<class T>
struct is_selected_cell_execution_domain : std::false_type {};

template<StaticString Name, size_t CellI, class CellMesh, bool Partitioned>
struct is_selected_cell_execution_domain<
   SelectedCellExecutionDomain<Name, CellI, CellMesh, Partitioned>>
   : std::true_type {};

template<class T>
inline constexpr bool is_selected_cell_execution_domain_v =
   is_selected_cell_execution_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_selected_boundary_face_execution_domain : std::false_type {};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class CellMesh>
struct is_selected_boundary_face_execution_domain<
   SelectedBoundaryFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      CellMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_selected_boundary_face_execution_domain_v =
   is_selected_boundary_face_execution_domain<
      std::remove_cvref_t<T>>::value;

template<class T>
struct is_selected_interior_face_execution_domain : std::false_type {};

template<
   StaticString Name,
   size_t FaceI,
   class FacePart,
   class MinusCellMesh,
   class PlusCellMesh>
struct is_selected_interior_face_execution_domain<
   SelectedInteriorFaceExecutionDomain<
      Name,
      FaceI,
      FacePart,
      MinusCellMesh,
      PlusCellMesh>> : std::true_type {};

template<class T>
inline constexpr bool is_selected_interior_face_execution_domain_v =
   is_selected_interior_face_execution_domain<
      std::remove_cvref_t<T>>::value;

} // namespace gendil
