// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef GENDIL_USE_MFEM

#include <mfem.hpp>
#include "gendil/FiniteElementMethod/Restrictions/RestrictionTypes/indirecth1restriction.hpp"
#include "gendil/Utilities/MemoryManagement/garbagecollector.hpp"

namespace gendil {

/**
 * @brief Get the restriction indices from an mfem::Mesh describing the isoparametric transformation.
 * 
 * @param mesh The mfem::Mesh from which to extract the restriction indices.
 * @return const int* The C-array containing the restriction indices.
 */
inline auto GetRestrictionIndices( const mfem::FiniteElementSpace & finite_element_space )
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
inline auto GetRestrictionIndices( mfem::Mesh & mesh )
{
   mesh.EnsureNodes();
   return GetRestrictionIndices( *(mesh.GetNodalFESpace()) );
}

/**
 * @brief Get an IndirectH1RestrictionSpecification from an mfem::FiniteElementSpace
 * 
 * @param finite_element_space The MFEM finite element space.
 * @return const IndirectH1RestrictionSpecification The H1 restriction.alignas
 * 
 * @note This will return an IndirectH1RestrictionSpecification even if the MFEM finite element space is a DG space.
 */
inline IndirectH1RestrictionSpecification
GetIndirectH1RestrictionSpecification(
   const mfem::FiniteElementSpace & finite_element_space )
{
   return IndirectH1RestrictionSpecification{ GetRestrictionIndices( finite_element_space ), (Integer)finite_element_space.GetNDofs() };
}

}

#endif
