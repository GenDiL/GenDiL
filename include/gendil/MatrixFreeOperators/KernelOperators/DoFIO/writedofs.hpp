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

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void WriteScalarDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return SerialWriteDofs<Add>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return ThreadedWriteDofs<Add>( thread, fe_space, element_index, x, global_dofs );
   }
}

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void WriteVectorDofsSerial(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   if constexpr ( std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
   {
      ConstexprLoop< v_dim >( [&]( auto i )
      {
         UnitLoop< std::tuple_element_t< i, dof_shape > >( [&]( auto... indices )
         {
            if constexpr ( Add )
               AtomicAdd( std::get< i >( global_dofs )( indices..., element_index ), std::get< i >( x )( indices... ) );
               // TODO: Should we assume no aliasing in the general case?
               // global_dofs( indices..., element_index ) += x( indices... );
            else
               std::get< i >( global_dofs )( indices..., element_index ) = std::get< i >( x )( indices... );
         });
      });
   }
   else // Gathering of H1 dofs.
   {
      ConstexprLoop< v_dim >( [&]( auto i )
      {
         UnitLoop< std::tuple_element_t< i, dof_shape > >( [&]( auto... indices )
         {
            AtomicAdd( std::get< i >( global_dofs )( indices..., element_index ), std::get< i >( x )( indices... ) );
         });
      });
   }
}

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void WriteVectorDofsThreaded(
   const KernelContext & ctx,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & local_dofs,
   GlobalTensor & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;
   using t_shapes = typename VectorDofShapes< KernelContext, dof_shape >::t_shapes;
   using r_shapes = typename VectorDofShapes< KernelContext, dof_shape >::r_shapes;

   ConstexprLoop< v_dim >( [&]( auto i_ )
   {
      constexpr Integer i = i_;
      ThreadLoop< std::tuple_element_t< i, t_shapes > >( ctx, [&] ( auto... t )
      {
         UnitLoop< std::tuple_element_t< i, r_shapes > >( [&] ( auto... k )
         {
            if constexpr ( !Add && std::is_same_v< typename FiniteElementSpace::restriction_type, L2Restriction > )
               std::get<i>( global_dofs )( t..., k..., element_index ) = std::get<i>( local_dofs )( k... );
            else
               AtomicAdd( std::get<i>( global_dofs )( t..., k..., element_index ), std::get<i>( local_dofs )( k... ) );
         });
      });
   });
}

template <
   bool Add,
   typename KernelContext,
   typename FiniteElementSpace,
   typename LocalTensor,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
void WriteVectorDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const LocalTensor & x,
   GlobalTensor & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return WriteVectorDofsSerial<Add>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return WriteVectorDofsThreaded<Add>( thread, fe_space, element_index, x, global_dofs );
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
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return WriteVectorDofs<false>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return WriteScalarDofs<false>( thread, fe_space, element_index, x, global_dofs );
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
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return WriteVectorDofs<true>( thread, fe_space, element_index, x, global_dofs );
   }
   else
   {
      return WriteScalarDofs<true>( thread, fe_space, element_index, x, global_dofs );
   }
}

}
