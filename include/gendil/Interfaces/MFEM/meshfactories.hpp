// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include "restriction.hpp"
#include "meshconnectivity.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/linemesh.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/quadmesh.hpp"
#include "gendil/Meshes/MeshDataStructures/UnstructuredMesh/hexmesh.hpp"

namespace gendil {

template < Integer MeshOrder >
LineMesh< MeshOrder > MakeLineMesh( mfem::Mesh & mesh )
{
   constexpr Integer Dim = 1;
   constexpr GlobalIndex D1D = MeshOrder + 1;
   mesh.EnsureNodes();
   MFEM_VERIFY( mesh.GetNodalFESpace()->GetMaxElementOrder() == MeshOrder, "Dynamic and static curvature order do not match." );
   MFEM_VERIFY( mesh.HasGeometry( mfem::Geometry::SEGMENT ), "The mesh must be a segment mesh." );
   MFEM_VERIFY( mesh.GetNumGeometries( Dim ) == 1, "The provided mesh must only contain segment elements." );

   return LineMesh< MeshOrder >{
      MakeFIFOView( mesh.GetNodes()->Read(), (GlobalIndex)mesh.GetNodalFESpace()->GetNDofs() ),
      MakeFIFOView( GetRestrictionIndices( mesh ), D1D, (GlobalIndex)mesh.GetNE() ),
      MakeMeshConnectivity< Dim >( mesh ),
      (GlobalIndex)mesh.GetNE()
   };
}

template < Integer MeshOrder >
QuadMesh< MeshOrder > MakeQuadMesh( mfem::Mesh & mesh )
{
   constexpr Integer Dim = 2;
   constexpr GlobalIndex D1D = MeshOrder + 1;
   mesh.EnsureNodes();
   MFEM_VERIFY( mesh.GetNodalFESpace()->GetMaxElementOrder() == MeshOrder, "Dynamic and static curvature order do not match." );
   MFEM_VERIFY( mesh.HasGeometry( mfem::Geometry::SQUARE ), "The mesh must be a quad mesh." );
   MFEM_VERIFY( mesh.GetNumGeometries( Dim ) == 1, "The provided mesh must only contain quad elements." );
   std::array< GlobalIndex, 2 > sizes{ Dim, (GlobalIndex)mesh.GetNodalFESpace()->GetNDofs() };
   Permutation< 2 > permutation = 
      mesh.GetNodalFESpace()->GetOrdering() == mfem::Ordering::Type::byVDIM ?
      Permutation< 2 >{ 1, 2 } : Permutation< 2 >{ 2, 1 };

   return QuadMesh< MeshOrder >{
      MakeStridedView( mesh.GetNodes()->Read(), sizes, permutation ),
      MakeFIFOView( GetRestrictionIndices( mesh ), D1D, D1D, (GlobalIndex)mesh.GetNE() ),
      MakeMeshConnectivity< Dim >( mesh ),
      (GlobalIndex)mesh.GetNE()
   };
}

template < Integer MeshOrder >
HexMesh< MeshOrder > MakeHexMesh( mfem::Mesh & mesh )
{
   constexpr Integer Dim = 3;
   constexpr GlobalIndex D1D = MeshOrder + 1;
   mesh.EnsureNodes();
   MFEM_VERIFY( mesh.GetNodalFESpace()->GetMaxElementOrder() == MeshOrder, "Dynamic and static curvature order do not match." );
   MFEM_VERIFY( mesh.HasGeometry( mfem::Geometry::CUBE ), "The mesh must be an hex mesh." );
   MFEM_VERIFY( mesh.GetNumGeometries( Dim ) == 1, "The provided mesh must only contain hex elements." );
   std::array< GlobalIndex, 2 > sizes{ Dim, (GlobalIndex)mesh.GetNodalFESpace()->GetNDofs() };
   Permutation< 2 > permutation = 
      mesh.GetNodalFESpace()->GetOrdering() == mfem::Ordering::Type::byVDIM ?
      Permutation< 2 >{ 1, 2 } : Permutation< 2 >{ 2, 1 };

   return HexMesh< MeshOrder >{
      MakeStridedView( mesh.GetNodes()->Read(), sizes, permutation ),
      MakeFIFOView( GetRestrictionIndices( mesh ), D1D, D1D, D1D, (GlobalIndex)mesh.GetNE() ),
      MakeMeshConnectivity< Dim >( mesh ),
      (GlobalIndex)mesh.GetNE()
   };
}

}

#endif
