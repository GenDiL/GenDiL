// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

// TODO: Do this at the halo level?
template < typename FiniteElementSpace, typename Dofs, typename HaloDofsIn, typename HaloDofsOut >
void SynchronizeHalo(
   const FiniteElementSpace & fe_space,
   Dofs & dofs,
   HaloDofsIn & interior_halo_dofs,
   HaloDofsOut & exterior_halo_dofs )
{
   if constexpr ( !std::is_same_v< typename FiniteElementSpace::halo_type, EmptyHalo<FiniteElementSpace::Dim> > )
   {
#ifdef GENDIL_USE_MPI
      const Integer num_neighbors = fe_space.halo.num_neighbors; // TODO: Make this a function
      MPI_Request * requests = new MPI_Request[ 2 * num_neighbors ]; // TODO: should we avoid the new?
      MPI_Request * send_requests = requests;
      MPI_Request * recv_requests = requests + num_neighbors;
      // TODO Synchronize host with device
      // Pack interior halo data
      ForEachInteriorHaloCell( fe_space, [&]( GlobalIndex halo_index, GlobalIndex halo_cell_index, GlobalIndex cell_index )
      {
         // TODO: Only this block seems specific to the FiniteElementSpace
         auto & halo_dofs = interior_halo_dofs.halos[ halo_index ];
         DofLoop< FiniteElementSpace >( [&]( auto... indices )
         {
            halo_dofs( indices..., halo_cell_index ) = dofs( indices..., cell_index );
         });
      });
      // Send and receive halo data
      for (size_t halo_index = 0; halo_index < num_neighbors; halo_index++)
      {
         constexpr Integer num_dofs = FiniteElementSpace::finite_element_type::GetNumDofs();
         const Integer halo_size = fe_space.halo.halo_sizes[halo_index] * num_dofs;
         const int neighbor = fe_space.halo.neighbors[halo_index];
         const int tag = 0;
         MPI_Isend( interior_halo_dofs[halo_index], halo_size, MPI_DOUBLE, neighbor, tag, fe_space.halo.communicator, &send_requests[halo_index] );
         MPI_Irecv( exterior_halo_dofs[halo_index], halo_size, MPI_DOUBLE, neighbor, tag, fe_space.halo.communicator, &recv_requests[halo_index] );
      }
      MPI_Waitall( 2*num_neighbors, requests, MPI_STATUSES_IGNORE);

      delete [] requests;
#endif
   }
}

}