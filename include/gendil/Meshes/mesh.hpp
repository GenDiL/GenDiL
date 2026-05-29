// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil {

namespace mesh {

template <typename T>
concept Mesh =
   requires (const T& m, GlobalIndex ci)
   {
      // Number of cells:
      { m.GetNumberOfCells() } -> std::convertible_to<GlobalIndex>;

      // Cell accessor:
      { m.GetCell(ci) };
   };

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
template < Mesh Mesh >
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
auto GetLocalFaceInfo( const Mesh & mesh, GlobalIndex cell_index, const FaceID & face_id )
{
   return mesh.GetLocalFaceInfo( cell_index, face_id );
}

/**
 * @brief Applies @a body to all cells using @a KernelConfiguration.
 *
 * @details For config-aware bodies, this overload invokes the body with the
 * kernel configuration object:
 *
 * @code
 * CellIterator< Config >( mesh, [] GENDIL_DEVICE ( auto kernel ) {
 *    auto cell_index = kernel.WorkItemIndex();
 *    ...
 * });
 * @endcode
 *
 * Batched device configurations currently use an experimental filtered model:
 * config-aware bodies are invoked only for active lanes in the final partial
 * batch. kernel.WorkItemIndex() is the active cell index. While SyncWorkItem()
 * remains block-wide, bodies that synchronize after CellIterator filtering are
 * intentionally experimental and must be validated per operator.
 *
 * One-index bodies remain the legacy path and are only valid for
 * KernelConfiguration::batch_size == 1.
 */
template < typename KernelConfiguration, Mesh Mesh, typename Lambda >
void CellIterator( const Mesh & mesh, Lambda && body )
{
   const GlobalIndex num_cells = mesh.GetNumberOfCells();

   if constexpr ( KernelConfiguration::batch_size > 1 )
   {
      auto config_body =
         [body = std::forward< Lambda >( body ), num_cells]
         GENDIL_HOST_DEVICE ( const KernelConfiguration & kernel ) mutable
         {
         #ifdef GENDIL_DEVICE_CODE
            if ( kernel.IsActive( num_cells ) )
            {
               body( kernel );
            }
         #else
            (void) kernel;
         #endif
         };

      KernelConfiguration::BlockLoop( num_cells, config_body );
   }
   else
   {
      KernelConfiguration::BlockLoop(
         num_cells,
         std::forward< Lambda >( body ) );
   }
}

} // namespace mesh

} // namespace gendil
