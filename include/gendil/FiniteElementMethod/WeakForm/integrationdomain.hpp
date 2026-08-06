// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/Meshes/mesh.hpp"
#include "gendil/Meshes/partition.hpp"

#include <tuple>
#include <type_traits>
#include <utility>

namespace gendil {

template<class Key, class T>
struct Entry;

template<class... Entries>
struct StaticMap;

template<class Map>
struct SumFormExpr;

/**
 * @brief Integration domain backed by one homogeneous mesh.
 *
 * The mesh supplies integration cells, geometry, and local face connectivity.
 * Participating finite-element spaces are bound independently through fields
 * and must be compatible with this mesh.
 */
template<class Mesh>
struct MeshIntegrationDomain
{
   using mesh_type = Mesh;

   Mesh mesh;
};

/**
 * @brief Integration domain backed by partitioned cell and face topology.
 *
 * The partition supplies cell-part meshes and global interior/boundary face
 * connectivity. Participating mixed finite-element spaces are bound
 * independently through fields and select compatible cell-part spaces.
 */
template<class Partition>
struct PartitionIntegrationDomain
{
   using partition_type = Partition;

   Partition partition;
};

template<class T>
struct is_mesh_integration_domain : std::false_type {};

template<class Mesh>
struct is_mesh_integration_domain<MeshIntegrationDomain<Mesh>>
   : std::true_type {};

template<class T>
inline constexpr bool is_mesh_integration_domain_v =
   is_mesh_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_partition_integration_domain : std::false_type {};

template<class Partition>
struct is_partition_integration_domain<PartitionIntegrationDomain<Partition>>
   : std::true_type {};

template<class T>
inline constexpr bool is_partition_integration_domain_v =
   is_partition_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
inline constexpr bool is_integration_domain_v =
   is_mesh_integration_domain_v<T> ||
   is_partition_integration_domain_v<T>;

// Internal normalized context entries. Domain stores topology and geometry;
// field spaces are selected independently when an integrand is prepared.
template<class Domain>
struct CellIntegrationDomain
{
   Domain domain;
};

template<class Domain>
struct InteriorFaceIntegrationDomain
{
   Domain domain;
};

template<class Domain>
struct BoundaryFaceIntegrationDomain
{
   Domain domain;
};

template<class T>
struct is_cell_integration_domain : std::false_type {};

template<class Domain>
struct is_cell_integration_domain<CellIntegrationDomain<Domain>>
   : std::true_type {};

template<class T>
inline constexpr bool is_cell_integration_domain_v =
   is_cell_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_interior_face_integration_domain : std::false_type {};

template<class Domain>
struct is_interior_face_integration_domain<
   InteriorFaceIntegrationDomain<Domain>> : std::true_type {};

template<class T>
inline constexpr bool is_interior_face_integration_domain_v =
   is_interior_face_integration_domain<std::remove_cvref_t<T>>::value;

template<class T>
struct is_boundary_face_integration_domain : std::false_type {};

template<class Domain>
struct is_boundary_face_integration_domain<
   BoundaryFaceIntegrationDomain<Domain>> : std::true_type {};

template<class T>
inline constexpr bool is_boundary_face_integration_domain_v =
   is_boundary_face_integration_domain<std::remove_cvref_t<T>>::value;

template<class Domain>
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetCellIntegrationDomainMesh(
   const CellIntegrationDomain<Domain>& domain)
{
   using DomainType = std::remove_cvref_t<Domain>;
   if constexpr (is_mesh_integration_domain_v<DomainType>)
   {
      return (domain.domain.mesh);
   }
   else if constexpr (
      requires (const DomainType& selected)
      {
         selected.GetCellMesh();
      })
   {
      return domain.domain.GetCellMesh();
   }
   else
   {
      static_assert(
         dependent_false_v<DomainType>,
         "Cell integration qdata and geometry require a selected cell domain. "
         "PartitionIntegrationDomain must be traversed one CellPart at a time.");
   }
}

template<class Integrand, class WeakFormContext>
   requires requires
   {
      typename std::remove_cvref_t<Integrand>::domain_type;
      std::remove_cvref_t<Integrand>::domain_type::name;
   }
GENDIL_HOST_DEVICE
constexpr decltype(auto) GetCellIntegrationDomainMesh(
   const Integrand&,
   const WeakFormContext& wf_ctx)
{
   using I = std::remove_cvref_t<Integrand>;
   constexpr auto DomainName = I::domain_type::name;
   return GetCellIntegrationDomainMesh(
      wf_ctx.template domain<DomainName>());
}

template<class WeakForm>
struct LocalFacetAssemblyDomain;

template<class WeakForm>
   requires requires
   {
      typename std::remove_cvref_t<WeakForm>::domain_type;
      std::remove_cvref_t<WeakForm>::domain_type::name;
   }
struct LocalFacetAssemblyDomain<WeakForm>
{
   using Form = std::remove_cvref_t<WeakForm>;
   static constexpr auto name = Form::domain_type::name;
};

template<class Key, class Form>
struct LocalFacetAssemblyDomain<Entry<Key, Form>>
   : LocalFacetAssemblyDomain<Form> {};

template<class FirstEntry, class... RestEntries>
struct LocalFacetAssemblyDomain<
   StaticMap<FirstEntry, RestEntries...>>
{
   static constexpr auto name =
      LocalFacetAssemblyDomain<FirstEntry>::name;
   static constexpr bool all_names_match =
      (
         (name == LocalFacetAssemblyDomain<RestEntries>::name) &&
         ...);

   static_assert(
      all_names_match,
      "GenericLocalFacetAssembly requires all integrands to use the same "
      "integration domain name.");
};

template<class Map>
struct LocalFacetAssemblyDomain<SumFormExpr<Map>>
   : LocalFacetAssemblyDomain<Map> {};

template<class WeakForm>
inline constexpr auto local_facet_assembly_domain_name_v =
   LocalFacetAssemblyDomain<
      std::remove_cvref_t<WeakForm>>::name;

template<class WeakForm, class WeakFormContext>
   requires (!requires
   {
      typename std::remove_cvref_t<WeakForm>::domain_type;
   })
constexpr decltype(auto) GetCellIntegrationDomainMesh(
   const WeakForm&,
   const WeakFormContext& wf_ctx)
{
   constexpr auto DomainName =
      local_facet_assembly_domain_name_v<WeakForm>;
   const auto& traversal_domain =
      wf_ctx.template domain<DomainName>();
   using IntegrationDomain =
      std::remove_cvref_t<decltype(traversal_domain.domain)>;
   static_assert(
      is_mesh_integration_domain_v<IntegrationDomain>,
      "GetCellIntegrationDomainMesh(weak_form, context) requires a "
      "MeshIntegrationDomain.");
   return GetCellIntegrationDomainMesh(traversal_domain);
}

template<class Mesh>
struct MeshIdentityTraits
{
   static constexpr bool available = false;
};

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

template<class DomainPartition, class FieldPartition>
struct IsCompatiblePartitionTopology
   : integration_domain_detail::CompatiblePartitionDispatch<
        std::remove_cvref_t<DomainPartition>,
        std::remove_cvref_t<FieldPartition>>
{};

} // namespace gendil
