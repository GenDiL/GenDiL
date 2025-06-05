// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"

namespace gendil {

/**
 * @brief Write element local degrees-of-freedom to a global tensor of degrees-of-freedom.
 * 
 * @tparam FiniteElementSpace The type of finite element space.
 * @tparam Dim The dimension of the finite element space.
 * @param element_index The index of the finite element.
 * @param local_dofs The container containing element local degrees-of-freedom.
 * @param global_dofs The tensor containing all the degrees-of-freedom.
 */
template < typename FiniteElementSpace,
           typename View >
GENDIL_HOST_DEVICE
void WriteDofs( const GlobalIndex element_index,
                const ElementDoF< FiniteElementSpace > & local_dofs,
                View & global_dofs )
{
   static_assert(
      View::rank == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in WriteDofs."
   );
   

   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            global_dofs( indices..., element_index ) = local_dofs( indices... );
         }
      );
   }
   else // Gathering of H1 dofs.
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            AtomicAdd( global_dofs( indices..., element_index ), local_dofs( indices... ) );
         }
      );
   }
}

/**
 * @brief Writes element DOFs to global memory. Assumes that `x` are only the
 * DOFs belonging to the thread accroding to the threading strategy.
*/
template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void SerialWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   static_assert(
      get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );

   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {      
         if constexpr ( Add )
            AtomicAdd( global_dofs( indices..., element_index ), x( indices... ) );
            // TODO: Should we assume no aliasing in the general case?
            // global_dofs( indices..., element_index ) += x( indices... );
         else
            global_dofs( indices..., element_index ) = x( indices... );
      });
   }
   else // Gathering of H1 dofs.
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            AtomicAdd( global_dofs( indices..., element_index ), x( indices... ) );
         }
      );
   }
}

/**
 * @brief Writes element DOFs to global memory. Assumes that `x` are only the
 * DOFs belonging to the thread accroding to the threading strategy.
*/
template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void ThreadedWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... k )
         {
            if constexpr ( Add )
               AtomicAdd( global_dofs( t..., k..., element_index ), x( k... ) );
               // TODO: Should we assume no aliasing in the general case?
               // global_dofs( t..., k..., element_index ) += x( k... );
            else
               global_dofs( t..., k..., element_index ) = x( k... );
         });
      });
   }
   else
   {
      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... k )
         {
            AtomicAdd( global_dofs( t..., k..., element_index ), x( k... ) );
         });
      });
   }
}

/**
 * @brief Writes element DOFs to global memory. Assumes that `x` are only the
 * DOFs belonging to the thread accroding to the threading strategy.
*/
template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void WriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return SerialWriteDofs<false>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return ThreadedWriteDofs<false>( thread, fe_space, element_index, x, global_dofs );
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE
void WriteAddDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return SerialWriteDofs<true>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return ThreadedWriteDofs<true>( thread, fe_space, element_index, x, global_dofs );
   }
}

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void SerialWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   static_assert(
      get_rank_v< GlobalDofsType > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in WriteDofs."
   );

   using rshape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   Real data[ data_size ];

   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );

   // Copy from local_dofs to reference_view
   auto reference_view = MakeFIFOView( data, dofs_sizes );
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         reference_view( indices... ) = local_dofs( indices... );
      }
   );

   // Apply orientation
   Permutation< FiniteElementSpace::Dim > orientation = face_info.orientation;
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );

   const GlobalIndex element_index = face_info.neighbor_index;
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         if constexpr ( Add )
            AtomicAdd( global_dofs( indices..., element_index ), oriented_view( indices... ) );
         else
            global_dofs( indices..., element_index ) = oriented_view( indices... );
      }
   );
}

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void ThreadedWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   static_assert(
      get_rank_v< GlobalDofsType > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in WriteDofs."
   );

   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   Real * data = thread.SharedAllocator.allocate( data_size );

   // Copy local dofs into reference view
   auto reference_view = MakeFixedFIFOView( data, DofShape{} );
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         reference_view( t..., k... ) = local_dofs( k... );
      });
   });

   // Apply orientation
   Permutation< FiniteElementSpace::Dim > orientation = face_info.orientation;
   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );

   const GlobalIndex element_index = face_info.neighbor_index;
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         global_dofs( t..., k..., element_index ) = oriented_view( t..., k... );
         if constexpr ( Add )
            AtomicAdd( global_dofs( t..., k..., element_index ), oriented_view( t..., k... ) );
         else
            global_dofs( t..., k..., element_index ) = oriented_view( t..., k... );;
      });
   });

   thread.Synchronize();
   thread.SharedAllocator.reset();
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      SerialWriteDofs<false>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      ThreadedWriteDofs<false>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteAddDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      SerialWriteDofs<true>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      ThreadedWriteDofs<true>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

}
