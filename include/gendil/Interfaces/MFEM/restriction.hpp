// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>
#include "gendil/FiniteElementMethod/restriction.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

/**
 * @brief Get the restriction indices from an mfem::Mesh describing the isoparametric transformation.
 * 
 * @param mesh The mfem::Mesh from which to extract the restriction indices.
 * @return const int* The C-array containing the restriction indices.
 */
auto GetRestrictionIndices( const mfem::FiniteElementSpace & finite_element_space )
{
   HostDevicePointer< const int > indices;
   const mfem::ElementRestriction* restr =
      dynamic_cast< const mfem::ElementRestriction * >( 
         finite_element_space.GetElementRestriction( mfem::ElementDofOrdering::LEXICOGRAPHIC )
      );
   if ( restr != nullptr ){
      indices.host_pointer = restr->GatherMap().HostRead();
      #ifdef GENDIL_USE_DEVICE
      indices.device_pointer = restr->GatherMap().Read();
      #endif
   }
   else
   {
      std::cout << "WARNING: Failed to cast restriction to mfem::ElementRestriction, creating an identity restriction." << std::endl;
      const int num_mesh_dofs = finite_element_space.GetNDofs();
      int * const ind = new int[ num_mesh_dofs ];
      indices.host_pointer = ind;
      AllocateDevicePointer( num_mesh_dofs, indices );
      for( int i = 0; i < num_mesh_dofs; i++ )
      {
         ind[i] = i;
      }
      ToDevice( num_mesh_dofs, indices );
      GarbageCollector::Instance().RegisterHostDevicePtr( indices );
   }
   return indices;
}

/**
 * @brief Get the restriction indices from an mfem::Mesh describing the isoparametric transformation.
 * 
 * @param mesh The mfem::Mesh from which to extract the restriction indices.
 * @return const int* The C-array containing the restriction indices.
 */
auto GetRestrictionIndices( mfem::Mesh & mesh )
{
   mesh.EnsureNodes();
   return GetRestrictionIndices( *(mesh.GetNodalFESpace()) );
}

/**
 * @brief Get an H1Restriction from an mfem::FiniteElementSpace
 * 
 * @param finite_element_space The MFEM finite element space.
 * @return const H1Restriction The H1 restriction.alignas
 * 
 * @note This will return an H1Restriction even if the MFEM finite element space is a DG space.
 */
const H1Restriction GetH1Restriction( const mfem::FiniteElementSpace & finite_element_space )
{
   return H1Restriction{ GetRestrictionIndices( finite_element_space ), (Integer)finite_element_space.GetNDofs() };
}

}

#endif
