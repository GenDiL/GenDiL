// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

#ifdef GENDIL_USE_MFEM
#include <general/forall.hpp>
#endif

namespace gendil {

namespace mesh {

/**
 * @brief Get the cell object corresponding to the cell index.
 * 
 * @tparam Mesh The mesh type.
 * @param mesh The mesh.
 * @param linear_index The linear index of the cell.
 * @return auto The cell object associated to the cell index.
 * 
 * @note See cell.hpp to see requirements on the Cell interface.
 */
template < typename Mesh >
GENDIL_HOST_DEVICE
auto GetCell( const Mesh & mesh, GlobalIndex cell_index )
{
   return mesh.GetCell( cell_index );
}

/**
 * @brief Get the cell index of a neighhboring cell.
 * 
 * @tparam Mesh The mesh type.
 * @tparam FaceID The type used to identified a face on a cell.
 * @param mesh The mesh.
 * @param cell_index The current cell index.
 * @param face_id The face identifiant.
 * @return auto The neighboring cell.
 */
template < typename Mesh, typename FaceID >
GENDIL_HOST_DEVICE
auto GetFaceNeighborInfo( const Mesh & mesh, GlobalIndex cell_index, const FaceID & face_id )
{
   return mesh.GetFaceNeighborInfo( cell_index, face_id );
}

/**
 * @brief Applies @a body to all cells in the @a mesh
 * 
 * @tparam Mesh The mesh type.
 * @tparam Lambda Invocable like void (*) ( GlobalIndex )
 * @param mesh The mesh.
 * @param body The function to invoke for each cell in mesh.
*/
template < typename Mesh, typename Lambda >
void CellIterator( Mesh const & mesh, Lambda && body )
{
   const GlobalIndex num_cells = mesh.GetNumberOfCells();

#if defined( GENDIL_USE_MFEM )
   mfem::forall( num_cells, body );
#elif defined( GENDIL_USE_RAJA )
   // TODO:
#else
   #pragma omp parallel for
   for (GlobalIndex i = 0; i < num_cells; i++)
   {
      body( i );
   }
#endif
}

template < typename KernelContext, typename Mesh, typename Lambda >
void CellIterator( const Mesh & mesh, Lambda && body )
{
   const GlobalIndex num_cells = mesh.GetNumberOfCells();

   KernelContext::BlockLoop( num_cells, std::forward< Lambda >( body ) );
}

} // namespace mesh

} // namespace gendil
