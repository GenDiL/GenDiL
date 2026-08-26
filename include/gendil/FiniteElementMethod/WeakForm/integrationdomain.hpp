// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/mixedfiniteelementspace.hpp"
#include "gendil/FiniteElementMethod/WeakForm/integrationdomaintraits.hpp"
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

} // namespace gendil
