// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>

#include <array>

#include "gendil/Interfaces/MFEM/GlobalFaceConnectivity/globalfaceconnectivity.hpp"
#include "gendil/Interfaces/MFEM/restriction.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/cellmesh.hpp"
#include "gendil/Meshes/Geometries/hypercube.hpp"
#include "gendil/Meshes/partition.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/view.hpp"

namespace gendil {

namespace mfem_interface {
namespace detail {

template < typename Geometry >
constexpr mfem::Geometry::Type MFEMGlobalPartitionElementGeometry()
{
   static_assert(
      is_hypercube_geometry< Geometry >::value,
      "MakeGlobalPartition(mfem::Mesh&): Geometry must be a HyperCube." );
   static_assert(
      Geometry::geometry_dim == Geometry::space_dim,
      "MakeGlobalPartition(mfem::Mesh&): embedded meshes are not supported." );
   static_assert(
      Geometry::geometry_dim >= 1 && Geometry::geometry_dim <= 3,
      "MakeGlobalPartition(mfem::Mesh&): v1 supports line, quad, and hex meshes only." );

   if constexpr ( Geometry::geometry_dim == 1 )
   {
      return mfem::Geometry::SEGMENT;
   }
   else if constexpr ( Geometry::geometry_dim == 2 )
   {
      return mfem::Geometry::SQUARE;
   }
   else
   {
      return mfem::Geometry::CUBE;
   }
}

template < typename Geometry, Integer MeshOrder >
void VerifyMFEMGlobalPartitionMesh( mfem::Mesh & mesh )
{
   constexpr Integer dim = Geometry::geometry_dim;
   constexpr auto element_geometry =
      MFEMGlobalPartitionElementGeometry< Geometry >();

   mesh.EnsureNodes();
   MFEM_VERIFY(
      mesh.Dimension() == dim,
      "MakeGlobalPartition(mfem::Mesh&): mesh dimension does not match Geometry." );
   MFEM_VERIFY(
      mesh.SpaceDimension() == dim,
      "MakeGlobalPartition(mfem::Mesh&): embedded meshes are not supported." );
   MFEM_VERIFY(
      mesh.GetNodalFESpace()->GetMaxElementOrder() == MeshOrder,
      "MakeGlobalPartition(mfem::Mesh&): dynamic and static curvature order do not match." );
   MFEM_VERIFY(
      mesh.HasGeometry( element_geometry ),
      "MakeGlobalPartition(mfem::Mesh&): mesh does not contain the requested element geometry." );
   MFEM_VERIFY(
      mesh.GetNumGeometries( dim ) == 1,
      "MakeGlobalPartition(mfem::Mesh&): mixed element geometries are not supported." );
}

} // namespace detail
} // namespace mfem_interface

/**
 * @brief Build a cell-only GenDiL mesh from a serial MFEM line, quad, or hex
 * mesh.
 *
 * This import path intentionally does not construct local face connectivity.
 * It is intended for global-facet partition workflows where interior and
 * boundary facets are supplied by materialized global face families.
 */
template < typename Geometry, Integer MeshOrder >
auto MakeMFEMCellMesh( mfem::Mesh & mesh )
{
   mfem_interface::detail::VerifyMFEMGlobalPartitionMesh<
      Geometry,
      MeshOrder >( mesh );

   constexpr Integer dim = Geometry::geometry_dim;
   constexpr GlobalIndex d1d = MeshOrder + 1;

   if constexpr ( dim == 1 )
   {
      return LineCellMesh< MeshOrder >{
         MakeFIFOView(
            mesh.GetNodes()->Read(),
            static_cast< GlobalIndex >( mesh.GetNodalFESpace()->GetNDofs() ) ),
         MakeFIFOView(
            GetRestrictionIndices( mesh ),
            d1d,
            static_cast< GlobalIndex >( mesh.GetNE() ) ),
         static_cast< GlobalIndex >( mesh.GetNE() )
      };
   }
   else if constexpr ( dim == 2 )
   {
      std::array< GlobalIndex, 2 > sizes{
         dim,
         static_cast< GlobalIndex >( mesh.GetNodalFESpace()->GetNDofs() )
      };
      const Permutation< 2 > permutation =
         mesh.GetNodalFESpace()->GetOrdering() == mfem::Ordering::Type::byVDIM ?
         Permutation< 2 >{ 1, 2 } : Permutation< 2 >{ 2, 1 };

      return QuadCellMesh< MeshOrder >{
         MakeStridedView( mesh.GetNodes()->Read(), sizes, permutation ),
         MakeFIFOView(
            GetRestrictionIndices( mesh ),
            d1d,
            d1d,
            static_cast< GlobalIndex >( mesh.GetNE() ) ),
         static_cast< GlobalIndex >( mesh.GetNE() )
      };
   }
   else
   {
      std::array< GlobalIndex, 2 > sizes{
         dim,
         static_cast< GlobalIndex >( mesh.GetNodalFESpace()->GetNDofs() )
      };
      const Permutation< 2 > permutation =
         mesh.GetNodalFESpace()->GetOrdering() == mfem::Ordering::Type::byVDIM ?
         Permutation< 2 >{ 1, 2 } : Permutation< 2 >{ 2, 1 };

      return HexCellMesh< MeshOrder >{
         MakeStridedView( mesh.GetNodes()->Read(), sizes, permutation ),
         MakeFIFOView(
            GetRestrictionIndices( mesh ),
            d1d,
            d1d,
            d1d,
            static_cast< GlobalIndex >( mesh.GetNE() ) ),
         static_cast< GlobalIndex >( mesh.GetNE() )
      };
   }
}

/**
 * @brief Build a global-facet partition from a serial MFEM line, quad, or hex
 * mesh.
 *
 * The returned partition contains a cell-only mesh, conforming global
 * interior face families, nonconforming global interior leaf-face families,
 * and boundary face families. Generic nonconforming operator execution remains
 * out of scope; consumers must dispatch only compatible face parts.
 */
template < typename Geometry, Integer MeshOrder >
auto MakeGlobalPartition( mfem::Mesh & mesh )
{
   auto cell_mesh = MakeMFEMCellMesh< Geometry, MeshOrder >( mesh );
   auto interior = MakeMFEMGlobalInteriorFaceConnectivity< Geometry >( mesh );
   auto boundary = MakeMFEMGlobalBoundaryFaceConnectivity< Geometry >( mesh );

   return MakePartition(
      MakeCellPart( cell_mesh ),
      MakeInteriorFacePart< 0, 0 >( interior.conforming.connectivity ),
      MakeInteriorFacePart< 0, 0 >( interior.nonconforming.connectivity ),
      MakeBoundaryFacePart< 0 >( boundary.connectivity ) );
}

} // namespace gendil

#endif // GENDIL_USE_MFEM
