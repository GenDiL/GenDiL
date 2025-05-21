// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"
#include "gendil/Utilities/getrank.hpp"

namespace gendil {

/**
 * @brief Reads element local degrees-of-freedom from a global tensor of degrees-of-freedom.
 * 
 * @tparam FiniteElementSpace The type of finite element space.
 * @tparam Dim The dimension of the finite element space.
 * @tparam T The type of the values.
 * @param element_index The index of the finite element.
 * @param global_dofs The tensor containing all the degrees-of-freedom.
 * @param local_dofs The container containing element local degrees-of-freedom.
 */
template < typename FiniteElementSpace,
           typename View >
GENDIL_HOST_DEVICE
void ReadDofs( const GlobalIndex element_index,
               const View & global_dofs,
               ElementDoF< FiniteElementSpace > & local_dofs )
{
   static_assert(
      View::rank == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         local_dofs( indices... ) = global_dofs( indices..., element_index );
      }
   );
}

template < typename FiniteElementSpace,
           Integer Dim,
           typename T,
           typename TensorType,
           typename KernelContext >
GENDIL_HOST_DEVICE
void ReadDofs( const KernelContext & ctx,
               const GlobalIndex element_index,
               const StridedView< Dim, T > & global_dofs,
               TensorType & local_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );

   DofLoop< FiniteElementSpace >(
      ctx,
      [&]( auto ... indices )
      {
         local_dofs( indices ... ) = global_dofs( indices ..., element_index );
      }
   );
}

template < typename FiniteElementSpace,
           Integer Dim,
           typename T,
           typename KernelContext,
           typename SharedTensor,
           size_t... Is >
GENDIL_HOST_DEVICE GENDIL_INLINE
void ReadDofsToShared( const KernelContext & ctx,
                       GlobalIndex element_index,
                       const StridedView< Dim, T > & global_dofs,
                       SharedTensor & local_dofs,
                       std::index_sequence<Is...> )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );

   using Orders = typename FiniteElementSpace::finite_element_type::shape_functions::orders;
   using dof_shape = std::index_sequence< GetNumDofs<Is, Orders>::value... >;

   DofLoop< FiniteElementSpace >(
      ctx,
      [&]( auto ... indices )
      {
         local_dofs( indices ... ) = global_dofs( indices ..., element_index );
      }
   );
}

template < typename FiniteElementSpace,
           Integer Dim,
           typename T,
           typename SharedTensor,
           typename KernelContext >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto ReadDofsToShared( const KernelContext & ctx,
                       GlobalIndex element_index,
                       const StridedView< Dim, T > & global_dofs,
                       SharedTensor & local_dofs )
{
   return ReadDofsToShared< FiniteElementSpace >( ctx, element_index, global_dofs, local_dofs, std::make_index_sequence<FiniteElementSpace::Dim>{} );
}


/**
 * @brief Read the degrees-of-freedom of a neighboring element according to @a face_info.
 * The degrees-of-freedom are reordered to be in a reference configuration.
 * 
 * @tparam FaceInfo 
 * @tparam FiniteElementSpace 
 * @tparam Dim 
 * @tparam T The type of the values.
 * @param face_info 
 * @param global_dofs 
 * @param local_dofs  
 */
template < 
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   Integer Dim,
   typename T,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void ReadDofs(
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const StridedView< Dim, T > & global_dofs,
   ElementDoF< FiniteElementSpace > & local_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   const GlobalIndex element_index = face_info.neighbor_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;

   Permutation< space_dim > orientation = face_info.orientation;

   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   Real data[ data_size ];

   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );
   auto reference_view = MakeFIFOView( data, dofs_sizes );
   
   // TODO: Study if oriented_view first or second is better ( invert orientation )
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         oriented_view( indices... ) = global_dofs( indices..., element_index );
      }
   );
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         local_dofs( indices... ) = reference_view( indices... );
      }
   );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto SerialReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   static_assert(
      get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );

   using rshape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         local_dofs( indices... ) = global_dofs( indices..., element_index );
      }
   );

   return local_dofs;
}

/**
 * @brief read element DOFs from global memory. Reads only the memory needed by
 * the thread as determined by the threading strategy.
*/
template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto ThreadedReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   auto x = MakeSerialRecursiveArray< Real >( rshape{} );

   // !FIXME this assumes the first dimensions are threaded but gives the threaded indices as last indices
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         x( k... ) = global_dofs( t..., k..., element_index );
      });
   });

   return x;
}

/**
 * @brief read element DOFs from global memory. Reads only the memory needed by
 * the thread as determined by the threading strategy.
*/
template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto ReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return SerialReadDofs( thread, fe_space, element_index, global_dofs );
   }
   else
   {
      return ThreadedReadDofs( thread, fe_space, element_index, global_dofs );
   }
}

/**
 * @brief Read the degrees-of-freedom of a neighboring element according to @a face_info.
 * The degrees-of-freedom are reordered to be in a reference configuration.
 * 
 * @tparam FaceInfo 
 * @tparam FiniteElementSpace 
 * @tparam Dim 
 * @tparam T The type of the values.
 * @param face_info 
 * @param global_dofs 
 * @param local_dofs  
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename GlobalDofs >
GENDIL_HOST_DEVICE
auto SerialReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const GlobalDofs & global_dofs )
{
   static_assert(
      get_rank_v< GlobalDofs > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using rshape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   const GlobalIndex element_index = face_info.neighbor_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;

   Permutation< space_dim > orientation = face_info.orientation;

   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   Real data[ data_size ];

   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );
   auto reference_view = MakeFIFOView( data, dofs_sizes );
   
   // TODO: Study if oriented_view first or second is better ( invert orientation )
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         oriented_view( indices... ) = global_dofs( indices..., element_index );
      }
   );
   DofLoop< FiniteElementSpace >(
      [&]( auto... indices )
      {
         local_dofs( indices... ) = reference_view( indices... );
      }
   );

   return local_dofs;
}


/**
 * @brief Read the degrees-of-freedom of a neighboring element according to @a face_info.
 * The degrees-of-freedom are reordered to be in a reference configuration.
 * 
 * @tparam FaceInfo 
 * @tparam FiniteElementSpace 
 * @tparam Dim 
 * @tparam T The type of the values.
 * @param face_info 
 * @param global_dofs 
 * @param local_dofs  
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename GlobalDofs >
GENDIL_HOST_DEVICE
auto ThreadedReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const GlobalDofs & global_dofs )
{
   static_assert(
      get_rank_v< GlobalDofs > == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   const GlobalIndex element_index = face_info.neighbor_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;

   Permutation< space_dim > orientation = face_info.orientation;

   // TODO: Use fixed FIFO view and dynamic shared allocation through kernel_conf
   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   Real * data = thread.SharedAllocator.allocate( data_size );

   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );
   auto reference_view = MakeFixedFIFOView( data, DofShape{} );
   
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   // TODO: Use a shared memory slice.
   // TODO: Study if oriented_view first or second is better ( invert orientation )
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         oriented_view( t..., k... ) = global_dofs( t..., k..., element_index ); // Assumes threads < registers
      });
   });
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         local_dofs( k... ) = reference_view( t..., k... );
      });
   });
   thread.Synchronize();

   thread.SharedAllocator.reset();

   return local_dofs;
}

/**
 * @brief Read the degrees-of-freedom of a neighboring element according to @a face_info.
 * The degrees-of-freedom are reordered to be in a reference configuration.
 * 
 * @tparam FaceInfo 
 * @tparam FiniteElementSpace 
 * @tparam Dim 
 * @tparam T The type of the values.
 * @param face_info 
 * @param global_dofs 
 * @param local_dofs  
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename GlobalDofs >
GENDIL_HOST_DEVICE
auto ReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const GlobalDofs & global_dofs )
{
   if constexpr ( is_serial_v< KernelContext > )
   {
      return SerialReadDofs( thread, fe_space, face_info, global_dofs );
   }
   else
   {
      return ThreadedReadDofs( thread, fe_space, face_info, global_dofs );
   }
}
/**
 * @brief Read the degrees-of-freedom of a neighboring element according to @a face_info.
 * This parallel version uses halo dofs to read the data from neiighboring ranks.
 * 
 * @tparam FaceInfo 
 * @tparam FiniteElementSpace 
 * @tparam Dim 
 * @tparam T The type of the values.
 * @param face_info 
 * @param global_dofs 
 * @param halo_dofs
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   Integer FaceIndex,
   typename Geometry,
   typename OrientationType,
   typename BoundaryType,
   typename NormalType,
   typename GlobalDofs,
   typename HaloDofs >
GENDIL_HOST_DEVICE
auto ReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const FaceConnectivity< FaceIndex, Geometry, OrientationType, BoundaryType, NormalType > & face_info,
   const GlobalDofs & global_dofs,
   const HaloDofs & halos )
{
   if ( IsDistributedFace( face_info ) )
   {
      auto halo_dofs = halos.GetView(); // TODO: Remove
      return ReadDofs( thread, fe_space, face_info, halo_dofs );
   }
   else
   {
      return ReadDofs( thread, fe_space, face_info, global_dofs );
   }
}

}
