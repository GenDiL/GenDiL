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

}
