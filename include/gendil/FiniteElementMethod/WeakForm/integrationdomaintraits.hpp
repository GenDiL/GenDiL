// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file
 * @brief Indexed mesh identity and integration-domain compatibility traits.
 *
 * These traits prove that domain and field meshes use the same cell indexing,
 * topology, and geometry. They are kept separate from integration-domain
 * wrapper and dispatch types so custom meshes can provide identity without
 * depending on the weak-form domain implementation.
 */

#include "gendil/Meshes/mesh.hpp"
#include "gendil/Meshes/partition.hpp"

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

/**
 * @brief Indexed-topology identity provider for a mesh type.
 *
 * The primary template deliberately provides no identity. A custom mesh may
 * specialize this trait by setting `available` to true and returning an
 * equality-comparable key from `Get(mesh)`. That key must contain every
 * property needed to prove that a cell index selects the same topology and
 * geometry in two mesh objects. Object address alone and cell count alone are
 * not sufficient.
 */
template<class Mesh>
struct MeshIdentityTraits
{
   static constexpr bool available = false;
};

/**
 * @brief Identity for periodic structured Cartesian meshes.
 *
 * Structured sizes, spacing, origin, and periodic-boundary state jointly
 * determine the indexed cells.
 */
template<class Mesh>
   requires requires (const Mesh& mesh)
   {
      mesh.sizes;
      mesh.h;
      mesh.mesh_origin;
      mesh.is_periodic_boundary;
   }
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.sizes,
         mesh.h,
         mesh.mesh_origin,
         mesh.is_periodic_boundary};
   }
};

/**
 * @brief Identity for one-dimensional Cartesian meshes.
 *
 * Spacing, origin, and the connectivity size determine the indexed cells.
 */
template<class Mesh>
   requires (
      requires (const Mesh& mesh)
      {
         mesh.h;
         mesh.mesh_origin;
         mesh.connectivity.size;
      })
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.h,
         mesh.mesh_origin,
         mesh.connectivity.size};
   }
};

/**
 * @brief Identity for structured meshes with one spacing value.
 *
 * Spacing, origin, and multidimensional connectivity sizes determine the
 * indexed cells.
 */
template<class Mesh>
   requires (
      requires (const Mesh& mesh)
      {
         mesh.h;
         mesh.mesh_origin;
         mesh.connectivity.sizes;
      } &&
      !requires (const Mesh& mesh)
      {
         mesh.sizes;
         mesh.is_periodic_boundary;
      })
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.h,
         mesh.mesh_origin,
         mesh.connectivity.sizes};
   }
};

/**
 * @brief Identity for two-dimensional Cartesian meshes.
 *
 * Both directional spacings, origin, and connectivity sizes are required.
 */
template<class Mesh>
   requires (
      requires (const Mesh& mesh)
      {
         mesh.h_x;
         mesh.h_y;
         mesh.mesh_origin;
         mesh.connectivity.sizes;
      } &&
      !requires (const Mesh& mesh)
      {
         mesh.h_z;
      })
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.h_x,
         mesh.h_y,
         mesh.mesh_origin,
         mesh.connectivity.sizes};
   }
};

/**
 * @brief Identity for three-dimensional Cartesian meshes.
 *
 * All directional spacings, origin, and connectivity sizes are required.
 */
template<class Mesh>
   requires requires (const Mesh& mesh)
   {
      mesh.h_x;
      mesh.h_y;
      mesh.h_z;
      mesh.mesh_origin;
      mesh.connectivity.sizes;
   }
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.h_x,
         mesh.h_y,
         mesh.h_z,
         mesh.mesh_origin,
         mesh.connectivity.sizes};
   }
};

/**
 * @brief Identity for connectivity-backed unstructured meshes.
 *
 * The identity records the node and restriction storage, their layouts, the
 * element-connectivity storage, and the element count. Shallow copies retain
 * identity; independently allocated connectivity does not.
 */
template<class Mesh>
   requires requires (const Mesh& mesh)
   {
      mesh.nodes.data.data;
      mesh.nodes.layout.strides;
      mesh.restriction.data.host_pointer;
      mesh.restriction.layout.strides;
      mesh.connectivity.element_connectivities.host_pointer;
      mesh.num_elems;
   }
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   template<class View, size_t... I>
   static auto GetStrides(
      const View& view,
      std::index_sequence<I...>)
   {
      return std::array<GlobalIndex, sizeof...(I)>{
         view.layout.strides[I]...};
   }

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.nodes.data.data,
         GetStrides(
            mesh.nodes,
            std::make_index_sequence<
               std::remove_cvref_t<decltype(mesh.nodes)>::rank>{}),
         mesh.restriction.data.host_pointer,
         GetStrides(
            mesh.restriction,
            std::make_index_sequence<
               std::remove_cvref_t<decltype(mesh.restriction)>::rank>{}),
         mesh.connectivity.element_connectivities.host_pointer,
         mesh.num_elems};
   }
};

/**
 * @brief Identity for cell-only unstructured meshes.
 *
 * The identity records node and restriction storage, their layouts, and the
 * element count when no separate element-connectivity storage is present.
 */
template<class Mesh>
   requires (
      requires (const Mesh& mesh)
      {
         mesh.nodes.data.data;
         mesh.nodes.layout.strides;
         mesh.restriction.data.host_pointer;
         mesh.restriction.layout.strides;
         mesh.num_elems;
      } &&
      !requires (const Mesh& mesh)
      {
         mesh.connectivity.element_connectivities.host_pointer;
      })
struct MeshIdentityTraits<Mesh>
{
   static constexpr bool available = true;

   template<class View, size_t... I>
   static auto GetStrides(
      const View& view,
      std::index_sequence<I...>)
   {
      return std::array<GlobalIndex, sizeof...(I)>{
         view.layout.strides[I]...};
   }

   static auto Get(const Mesh& mesh)
   {
      return std::tuple{
         mesh.nodes.data.data,
         GetStrides(
            mesh.nodes,
            std::make_index_sequence<
               std::remove_cvref_t<decltype(mesh.nodes)>::rank>{}),
         mesh.restriction.data.host_pointer,
         GetStrides(
            mesh.restriction,
            std::make_index_sequence<
               std::remove_cvref_t<decltype(mesh.restriction)>::rank>{}),
         mesh.num_elems};
   }
};

/**
 * @brief Recursive identity for Cartesian-product-like meshes.
 *
 * Identity is available only when every flattened component returned by
 * `GetSubMeshes()` provides MeshIdentityTraits. The ordered tuple of component
 * identities preserves factor ordering and nested-product topology.
 */
template<class Mesh>
   requires requires (const Mesh& mesh)
   {
      mesh.GetSubMeshes();
   }
struct MeshIdentityTraits<Mesh>
{
private:
   using SubMeshes =
      std::remove_cvref_t<decltype(std::declval<const Mesh&>().GetSubMeshes())>;

   template<size_t... I>
   static consteval bool SubMeshIdentitiesAvailable(
      std::index_sequence<I...>)
   {
      return (
         MeshIdentityTraits<
            std::remove_cvref_t<
               std::tuple_element_t<I, SubMeshes>>>::available &&
         ...);
   }

public:
   static constexpr bool available =
      SubMeshIdentitiesAvailable(
         std::make_index_sequence<std::tuple_size_v<SubMeshes>>{});

   static auto Get(const Mesh& mesh)
   {
      static_assert(
         available,
         "CartesianProductMesh identity requires every component mesh to "
         "provide MeshIdentityTraits.");
      return std::apply(
         [] (const auto&... sub_meshes)
         {
            return std::tuple{
               MeshIdentityTraits<
                  std::remove_cvref_t<decltype(sub_meshes)>>::Get(
                     sub_meshes)...};
         },
         mesh.GetSubMeshes());
   }
};

/**
 * @brief Compatibility between a domain mesh and a finite-element field mesh.
 *
 * `value` checks the mesh concepts and exact mesh type. The separate
 * `has_indexed_identity` member states whether compatibility can be proved,
 * and `Check` compares cell counts followed by the MeshIdentityTraits keys.
 */
template<class DomainMesh, class FieldMesh>
struct IsCompatibleMeshDomain
   : std::bool_constant<
        mesh::Mesh<std::remove_cvref_t<DomainMesh>> &&
        mesh::Mesh<std::remove_cvref_t<FieldMesh>> &&
        std::is_same_v<
           std::remove_cvref_t<DomainMesh>,
           std::remove_cvref_t<FieldMesh>>>
{
   using Domain = std::remove_cvref_t<DomainMesh>;
   using Field = std::remove_cvref_t<FieldMesh>;

   static constexpr bool has_indexed_identity =
      MeshIdentityTraits<Domain>::available &&
      MeshIdentityTraits<Field>::available;

   static bool Check(const DomainMesh& domain_mesh, const FieldMesh& field_mesh)
   {
      if (domain_mesh.GetNumberOfCells() != field_mesh.GetNumberOfCells())
      {
         return false;
      }

      if constexpr (has_indexed_identity)
      {
         return MeshIdentityTraits<Domain>::Get(domain_mesh) ==
                MeshIdentityTraits<Field>::Get(field_mesh);
      }
      else
      {
         // Equal cell counts cannot establish common indexed topology.
         return false;
      }
   }
};

namespace integration_domain_detail {

/*
 * Partition compatibility is intentionally based on cell-part layout,
 * relation-index validity, and indexed identity of corresponding cell meshes.
 * Global face-family counts and connectivity contents belong to the traversal
 * partition and are not compared with the field partition.
 */
template<class DomainPartition, class FieldPartition, size_t... I>
consteval bool CompatiblePartitionCellTypes(std::index_sequence<I...>)
{
   using DomainParts =
      typename std::remove_cvref_t<DomainPartition>::cell_parts_type;
   using FieldParts =
      typename std::remove_cvref_t<FieldPartition>::cell_parts_type;

   return (
      IsCompatibleMeshDomain<
         typename std::tuple_element_t<I, DomainParts>::mesh_type,
         typename std::tuple_element_t<I, FieldParts>::mesh_type>::value &&
      ...);
}

template<class DomainPartition, class FieldPartition, size_t... I>
consteval bool CompatiblePartitionCellIdentities(std::index_sequence<I...>)
{
   using DomainParts =
      typename std::remove_cvref_t<DomainPartition>::cell_parts_type;
   using FieldParts =
      typename std::remove_cvref_t<FieldPartition>::cell_parts_type;

   return (
      IsCompatibleMeshDomain<
         typename std::tuple_element_t<I, DomainParts>::mesh_type,
         typename std::tuple_element_t<I, FieldParts>::mesh_type>::
            has_indexed_identity &&
      ...);
}

template<class DomainPartition, class FieldPartition, size_t... I>
bool CheckPartitionCells(
   const DomainPartition& domain_partition,
   const FieldPartition& field_partition,
   std::index_sequence<I...>)
{
   return (
      IsCompatibleMeshDomain<
         typename std::tuple_element_t<
            I,
            typename DomainPartition::cell_parts_type>::mesh_type,
         typename std::tuple_element_t<
            I,
            typename FieldPartition::cell_parts_type>::mesh_type>::Check(
               std::get<I>(domain_partition.CellParts()).mesh,
               std::get<I>(field_partition.CellParts()).mesh) &&
      ...);
}

template<class DomainPartition, class FieldPartition, bool SameCellCount>
struct CompatiblePartitionTopologyImpl : std::false_type
{
   static constexpr bool has_indexed_identity = false;

   static bool Check(const DomainPartition&, const FieldPartition&)
   {
      return false;
   }
};

template<class DomainPartition, class FieldPartition>
struct CompatiblePartitionTopologyImpl<
   DomainPartition,
   FieldPartition,
   true>
   : std::bool_constant<
        CompatiblePartitionCellTypes<
           DomainPartition,
           FieldPartition>(
              std::make_index_sequence<DomainPartition::num_cell_parts>{}) &&
        partition_detail::all_partition_relation_indices_in_range_v<
           FieldPartition::num_cell_parts,
           typename DomainPartition::interior_face_parts_type> &&
        partition_detail::all_partition_relation_indices_in_range_v<
           FieldPartition::num_cell_parts,
           typename DomainPartition::boundary_face_parts_type>>
{
   static constexpr bool has_indexed_identity =
      CompatiblePartitionCellIdentities<
         DomainPartition,
         FieldPartition>(
            std::make_index_sequence<DomainPartition::num_cell_parts>{});

   static bool Check(
      const DomainPartition& domain_partition,
      const FieldPartition& field_partition)
   {
      return CheckPartitionCells(
         domain_partition,
         field_partition,
         std::make_index_sequence<DomainPartition::num_cell_parts>{});
   }
};

} // namespace integration_domain_detail

namespace integration_domain_detail {

template<
   class DomainPartition,
   class FieldPartition,
   bool BothPartitions =
      is_partition_v<DomainPartition> &&
      is_partition_v<FieldPartition>>
struct CompatiblePartitionDispatch : std::false_type
{
   static constexpr bool has_indexed_identity = false;

   static bool Check(const DomainPartition&, const FieldPartition&)
   {
      return false;
   }
};

template<class DomainPartition, class FieldPartition>
struct CompatiblePartitionDispatch<
   DomainPartition,
   FieldPartition,
   true>
   : CompatiblePartitionTopologyImpl<
        std::remove_cvref_t<DomainPartition>,
        std::remove_cvref_t<FieldPartition>,
        (
           std::remove_cvref_t<DomainPartition>::num_cell_parts ==
           std::remove_cvref_t<FieldPartition>::num_cell_parts)>
{};

} // namespace integration_domain_detail

/**
 * @brief Compatibility between traversal and field partition cell topology.
 *
 * The compile-time value requires matching cell-part counts and cell-mesh
 * types, plus valid traversal face relations into the field partition.
 * `has_indexed_identity` and `Check` then establish identity for every
 * corresponding cell-part mesh. Declared global-face connectivity itself is
 * deliberately not compared.
 */
template<class DomainPartition, class FieldPartition>
struct IsCompatiblePartitionTopology
   : integration_domain_detail::CompatiblePartitionDispatch<
        std::remove_cvref_t<DomainPartition>,
        std::remove_cvref_t<FieldPartition>>
{};

} // namespace gendil

