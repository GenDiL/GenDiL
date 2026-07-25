// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrate.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomain.hpp"
#include "gendil/FiniteElementMethod/WeakForm/weakformcontext.hpp"
#include "gendil/Utilities/dependentfalse.hpp"
#include "gendil/Utilities/Loop/constexprloop.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

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

template<StaticString Name, class Domain, class Fn>
void ForEachCellExecutionDomainInDomain(
   const CellIntegrationDomain<Domain>& normalized_domain,
   Fn&& fn)
{
   using Integration = std::remove_cvref_t<Domain>;

   if constexpr (is_mesh_integration_domain_v<Integration>)
   {
      using Mesh = typename Integration::mesh_type;
      fn(SelectedCellExecutionDomain<Name, 0, Mesh, false>{
         Mesh{ normalized_domain.domain.mesh } });
   }
   else if constexpr (is_partition_integration_domain_v<Integration>)
   {
      const auto& parts = normalized_domain.domain.partition.CellParts();
      using Parts = std::remove_cvref_t<decltype(parts)>;
      constexpr size_t NumParts = std::tuple_size_v<Parts>;
      ConstexprLoop<NumParts>(
         [&] (auto index)
         {
            constexpr size_t I = decltype(index)::value;
            const auto& mesh = std::get<I>(parts).mesh;
            using Mesh = std::remove_cvref_t<decltype(mesh)>;
            fn(SelectedCellExecutionDomain<Name, I, Mesh, true>{
               Mesh{ mesh } });
         });
   }
   else
   {
      static_assert(
         dependent_false_v<Integration>,
         "Cells<Name> requires a normalized mesh or partition integration "
         "domain.");
   }
}

template<class WFContext, StaticString Name, class Fn>
void ForEachCellExecutionDomain(
   const WFContext& wf_ctx,
   Cells<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;
   static_assert(
      Context::template has_domain<Name>(),
      "Cells<Name> requires an integration domain registered under Name.");

   if constexpr (Context::template has_domain<Name>())
   {
      ForEachCellExecutionDomainInDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachGlobalInteriorFaceExecutionDomain(
   const InteriorFaceIntegrationDomain<Domain>& normalized_domain,
   Fn&& fn)
{
   using Integration = std::remove_cvref_t<Domain>;
   static_assert(
      is_partition_integration_domain_v<Integration>,
      "InteriorFacets<Name> global traversal requires a "
      "PartitionIntegrationDomain.");

   if constexpr (is_partition_integration_domain_v<Integration>)
   {
      const auto& partition = normalized_domain.domain.partition;
      const auto& face_parts = partition.InteriorFaceParts();
      using FaceParts = std::remove_cvref_t<decltype(face_parts)>;
      constexpr size_t NumParts = std::tuple_size_v<FaceParts>;
      static_assert(
         NumParts > 0,
         "InteriorFacets<Name>: PartitionIntegrationDomain has no interior "
         "face parts; partition domains do not fall back to local-facet "
         "traversal.");

      ConstexprLoop<NumParts>(
         [&] (auto index)
         {
            constexpr size_t FaceI = decltype(index)::value;
            const auto& face_part = std::get<FaceI>(face_parts);
            using FacePart = std::remove_cvref_t<decltype(face_part)>;
            constexpr size_t MinusI = FacePart::minus_cell_index;
            constexpr size_t PlusI = FacePart::plus_cell_index;
            const auto& minus_mesh =
               std::get<MinusI>(partition.CellParts()).mesh;
            const auto& plus_mesh =
               std::get<PlusI>(partition.CellParts()).mesh;
            using MinusMesh = std::remove_cvref_t<decltype(minus_mesh)>;
            using PlusMesh = std::remove_cvref_t<decltype(plus_mesh)>;

            fn(SelectedInteriorFaceExecutionDomain<
               Name,
               FaceI,
               FacePart,
               MinusMesh,
               PlusMesh>{
                  FacePart{ face_part },
                  MinusMesh{ minus_mesh },
                  PlusMesh{ plus_mesh } });
         });
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachLocalInteriorFaceExecutionDomain(
   const CellIntegrationDomain<Domain>& normalized_domain,
   Fn&& fn)
{
   using Integration = std::remove_cvref_t<Domain>;
   if constexpr (is_mesh_integration_domain_v<Integration>)
   {
      using Mesh = typename Integration::mesh_type;
      fn(SelectedCellExecutionDomain<Name, 0, Mesh, false>{
         Mesh{ normalized_domain.domain.mesh } });
   }
   else
   {
      static_assert(
         dependent_false_v<Integration>,
         "InteriorFacets<Name>: PartitionIntegrationDomain has no interior "
         "face parts; partition domains do not fall back to local-facet "
         "traversal.");
   }
}

template<class WFContext, StaticString Name, class Fn>
void ForEachInteriorFaceExecutionDomain(
   const WFContext& wf_ctx,
   InteriorFacets<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;

   if constexpr (Context::template has_interior_face_domain<Name>())
   {
      ForEachGlobalInteriorFaceExecutionDomain<Name>(
         wf_ctx.template interior_face_domain<Name>(),
         std::forward<Fn>(fn));
   }
   else if constexpr (Context::template has_domain<Name>())
   {
      ForEachLocalInteriorFaceExecutionDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
   else
   {
      static_assert(
         dependent_false_v<Context>,
         "InteriorFacets<Name> requires an integration domain registered "
         "under Name.");
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachGlobalBoundaryFaceExecutionDomain(
   const BoundaryFaceIntegrationDomain<Domain>& normalized_domain,
   Fn&& fn)
{
   using Integration = std::remove_cvref_t<Domain>;
   static_assert(
      is_partition_integration_domain_v<Integration>,
      "BoundaryFacets<Name> global traversal requires a "
      "PartitionIntegrationDomain.");

   if constexpr (is_partition_integration_domain_v<Integration>)
   {
      const auto& partition = normalized_domain.domain.partition;
      const auto& face_parts = partition.BoundaryFaceParts();
      using FaceParts = std::remove_cvref_t<decltype(face_parts)>;
      constexpr size_t NumParts = std::tuple_size_v<FaceParts>;
      static_assert(
         NumParts > 0,
         "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary "
         "face parts; partition domains do not fall back to local-facet "
         "traversal.");

      ConstexprLoop<NumParts>(
         [&] (auto index)
         {
            constexpr size_t FaceI = decltype(index)::value;
            const auto& face_part = std::get<FaceI>(face_parts);
            using FacePart = std::remove_cvref_t<decltype(face_part)>;
            constexpr size_t CellI = FacePart::cell_index;
            const auto& cell_mesh =
               std::get<CellI>(partition.CellParts()).mesh;
            using CellMesh = std::remove_cvref_t<decltype(cell_mesh)>;

            fn(SelectedBoundaryFaceExecutionDomain<
               Name,
               FaceI,
               FacePart,
               CellMesh>{
                  FacePart{ face_part },
                  CellMesh{ cell_mesh } });
         });
   }
}

template<StaticString Name, class Domain, class Fn>
void ForEachLocalBoundaryFaceExecutionDomain(
   const CellIntegrationDomain<Domain>& normalized_domain,
   Fn&& fn)
{
   using Integration = std::remove_cvref_t<Domain>;
   if constexpr (is_mesh_integration_domain_v<Integration>)
   {
      using Mesh = typename Integration::mesh_type;
      fn(SelectedCellExecutionDomain<Name, 0, Mesh, false>{
         Mesh{ normalized_domain.domain.mesh } });
   }
   else
   {
      static_assert(
         dependent_false_v<Integration>,
         "BoundaryFacets<Name>: PartitionIntegrationDomain has no boundary "
         "face parts; partition domains do not fall back to local-facet "
         "traversal.");
   }
}

template<class WFContext, StaticString Name, class Fn>
void ForEachBoundaryFaceExecutionDomain(
   const WFContext& wf_ctx,
   BoundaryFacets<Name>,
   Fn&& fn)
{
   using Context = std::remove_cvref_t<WFContext>;

   if constexpr (Context::template has_boundary_face_domain<Name>())
   {
      ForEachGlobalBoundaryFaceExecutionDomain<Name>(
         wf_ctx.template boundary_face_domain<Name>(),
         std::forward<Fn>(fn));
   }
   else if constexpr (Context::template has_domain<Name>())
   {
      ForEachLocalBoundaryFaceExecutionDomain<Name>(
         wf_ctx.template domain<Name>(),
         std::forward<Fn>(fn));
   }
   else
   {
      static_assert(
         dependent_false_v<Context>,
         "BoundaryFacets<Name> requires an integration domain registered "
         "under Name.");
   }
}

} // namespace gendil
