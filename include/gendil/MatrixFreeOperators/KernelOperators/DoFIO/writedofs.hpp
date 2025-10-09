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

enum WriteOp
{
   Write,
   WriteAdd,
   WriteSub
};

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void SerialWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
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
   Permutation< FiniteElementSpace::Dim > orientation = face_info.get_orientation();
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );

   const GlobalIndex element_index = face_info.get_cell_index();
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         if constexpr ( Op == WriteAdd )
            AtomicAdd( global_dofs( indices..., element_index ), oriented_view( indices... ) );
         else if constexpr ( Op == WriteSub )
         {
            const Real value = -oriented_view( indices... );
            AtomicAdd( global_dofs( indices..., element_index ), value );
         }
         else
            global_dofs( indices..., element_index ) = oriented_view( indices... );
      }
   );
}

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void ThreadedWriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
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
   Permutation< FiniteElementSpace::Dim > orientation = face_info.get_orientation();
   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );

   const GlobalIndex element_index = face_info.get_cell_index();
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         if constexpr ( Op == WriteAdd )
            AtomicAdd( global_dofs( t..., k..., element_index ), oriented_view( t..., k... ) );
         else if constexpr ( Op == WriteSub )
            AtomicAdd( global_dofs( t..., k..., element_index ), -oriented_view( t..., k... ) );
         else
            global_dofs( t..., k..., element_index ) = oriented_view( t..., k... );;
      });
   });

   thread.Synchronize();
   thread.SharedAllocator.reset();
}

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteScalarDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      SerialWriteDofs<Op>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      ThreadedWriteDofs<Op>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void SerialWriteVectorDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   // TODO: branch on identity orientation to skip copy
   ConstexprLoop< v_dim >( [&]( auto i )
   {
      using dof_scalar_shape = std::tuple_element_t< i, dof_shape >;
      constexpr size_t data_size = Product( dof_scalar_shape{} );
      Real data[ data_size ];

      // Copy from local_dofs to reference_view
      auto reference_view = MakeFixedFIFOView( data, dof_scalar_shape{} );
      UnitLoop< dof_scalar_shape >(
         [&]( auto... indices )
         {
            reference_view( indices... ) = std::get< i >( local_dofs )( indices... );
         }
      );

      // Apply orientation
      Permutation< FiniteElementSpace::Dim > orientation = face_info.get_orientation();
      auto oriented_view = MakeOrientedView( data, dof_scalar_shape{}, orientation );

      const GlobalIndex element_index = face_info.get_cell_index();
      UnitLoop< dof_scalar_shape >(
         [&]( auto... indices )
         {
            if constexpr ( Op == WriteAdd )
               AtomicAdd( std::get< i >( global_dofs )( indices..., element_index ), oriented_view( indices... ) );
            else if constexpr ( Op == WriteSub )
            {
               const Real value = -oriented_view( indices... );
               AtomicAdd( std::get< i >( global_dofs )( indices..., element_index ), value );
            }
            else
               std::get< i >( global_dofs )( indices..., element_index ) = oriented_view( indices... );
         }
      );
   } );
}

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void ThreadedWriteVectorDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
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
   Permutation< FiniteElementSpace::Dim > orientation = face_info.get_orientation();
   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );

   const GlobalIndex element_index = face_info.get_cell_index();
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         if constexpr ( Op == WriteAdd )
            AtomicAdd( global_dofs( t..., k..., element_index ), oriented_view( t..., k... ) );
         else if constexpr ( Op == WriteSub )
            AtomicAdd( global_dofs( t..., k..., element_index ), -oriented_view( t..., k... ) );
         else
            global_dofs( t..., k..., element_index ) = oriented_view( t..., k... );;
      });
   });

   thread.Synchronize();
   thread.SharedAllocator.reset();
}

template <
   WriteOp Op,
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteVectorDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      SerialWriteVectorDofs<Op>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      ThreadedWriteVectorDofs<Op>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      WriteVectorDofs<Write>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      WriteScalarDofs<Write>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteAddDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      WriteVectorDofs<WriteAdd>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      WriteScalarDofs<WriteAdd>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename LocalDofsType,
   typename GlobalDofsType >
GENDIL_HOST_DEVICE
void WriteSubDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const LocalDofsType & local_dofs,
   GlobalDofsType & global_dofs )
{
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      WriteVectorDofs<WriteSub>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
   else
   {
      WriteScalarDofs<WriteSub>( thread, fe_space, face_info, local_dofs, global_dofs );
   }
}

}
