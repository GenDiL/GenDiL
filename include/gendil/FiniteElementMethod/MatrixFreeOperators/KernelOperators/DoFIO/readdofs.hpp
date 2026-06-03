// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/toarray.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/elementdof.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/dofloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/facereaddofspolicy.hpp"
#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/KernelContext/isthreadeddim.hpp"
#include "gendil/Utilities/KernelContext/threadedshapecoverage.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Meshes/Connectivities/faceconnectivity.hpp"
#include "gendil/Utilities/TupleHelperFunctions/tuplehelperfunctions.hpp"
#include "gendil/Utilities/getrank.hpp"
#include "gendil/Utilities/MathHelperFunctions/product.hpp"

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
   CellFaceView Face,
   Integer Dim,
   typename T,
   typename FiniteElementSpace >
GENDIL_HOST_DEVICE
void ReadDofs(
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs,
   ElementDoF< FiniteElementSpace > & local_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   using DofShape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;

   Permutation< space_dim > orientation = face_info.orientation;
   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

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
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   static_assert(
      threaded_shape_covered_v< KernelContext, DofShape >,
      "Under-threaded strided coverage is not supported by this threaded helper yet." );
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

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto ReadScalarDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      return SerialReadDofs( thread, fe_space, element_index, global_dofs );
   }
   else
   {
      return ThreadedReadDofs( thread, fe_space, element_index, global_dofs );
   }
}

template < typename dof_shape, size_t ... I >
auto MakeVectorDofs( dof_shape, std::index_sequence< I... > )
{
   return std::make_tuple( MakeSerialRecursiveArray< Real >( std::tuple_element_t< I, dof_shape >{} )... );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto ReadVectorDofsSerial(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;
   auto local_dofs = MakeVectorDofs( dof_shape{}, std::make_index_sequence< v_dim >{} );

   ConstexprLoop< v_dim >( [&]( auto i )
   {
      UnitLoop< std::tuple_element_t< i, dof_shape > >( [&]( auto... indices )
      {
         std::get< i >( local_dofs )( indices... ) = std::get< i >( global_dofs )( indices..., element_index );
      });
   });

   return local_dofs;
}

template <
   typename KernelContext,
   typename DofShapes,
   typename Indices = std::make_index_sequence< std::tuple_size_v< DofShapes > > >
struct VectorDofShapes;

template < typename KernelContext, typename DofShapes, size_t ... I >
struct VectorDofShapes< KernelContext, DofShapes, std::index_sequence< I... > >
{
   using t_shapes =
      std::tuple<
         subsequence_t<
            std::tuple_element_t< I, DofShapes >,
            typename KernelContext::template threaded_dimensions< std::tuple_element_t< I, DofShapes >::size() >
         >...
      >;
   using r_shapes =
      std::tuple<
         subsequence_t<
            std::tuple_element_t< I, DofShapes >,
            typename KernelContext::template register_dimensions< std::tuple_element_t< I, DofShapes >::size() >
         >...
      >;
};

template < typename KernelContext, typename dof_shape, size_t ... I >
GENDIL_HOST_DEVICE
auto MakeVectorDofs( const KernelContext & ctx, dof_shape, std::index_sequence< I... > )
{
   using r_shapes = typename VectorDofShapes< KernelContext, dof_shape >::r_shapes;
   return std::make_tuple( MakeSerialRecursiveArray< Real >( std::tuple_element_t< I, r_shapes >{} )... );
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
auto ReadVectorDofsThreaded(
   const KernelContext & ctx,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;
   using t_shapes = typename VectorDofShapes< KernelContext, dof_shape >::t_shapes;
   using r_shapes = typename VectorDofShapes< KernelContext, dof_shape >::r_shapes;
   auto local_dofs = MakeVectorDofs( ctx, dof_shape{}, std::make_index_sequence< v_dim >{} );

   // !FIXME this assumes the first dimensions are threaded but gives the threaded indices as last indices
   ConstexprLoop< v_dim >( [&]( auto i_ )
   {
      constexpr Integer i = i_;
      ThreadLoop< std::tuple_element_t< i, t_shapes > >( ctx, [&] ( auto... t )
      {
         UnitLoop< std::tuple_element_t< i, r_shapes > >( [&] ( auto... k )
         {
            std::get<i>( local_dofs )( k... ) = std::get<i>( global_dofs )( t..., k..., element_index );
         });
      });
   });

   return local_dofs;
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   typename GlobalTensor >
GENDIL_HOST_DEVICE inline
auto ReadVectorDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   GlobalIndex element_index,
   const GlobalTensor & global_dofs )
{
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      return ReadVectorDofsSerial( thread, fe_space, element_index, global_dofs );
   }
   else
   {
      return ReadVectorDofsThreaded( thread, fe_space, element_index, global_dofs );
   }
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
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return ReadVectorDofs( thread, fe_space, element_index, global_dofs );
   }
   else
   {
      return ReadScalarDofs( thread, fe_space, element_index, global_dofs );
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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto SerialReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   // static_assert(
   //    get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
   //    "Mismatching dimensions in ReadDofs."
   // );
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using rshape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   const GlobalIndex element_index = face_info.GetCellIndex();
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   using DofShape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;

   Permutation< space_dim > orientation = face_info.GetOrientation();
   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ThreadedReadDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   // static_assert(
   //    get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
   //    "Mismatching dimensions in ReadDofs."
   // );
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   static_assert(
      threaded_shape_covered_v< KernelContext, DofShape >,
      "Under-threaded strided coverage is not supported by this threaded helper yet." );
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   const GlobalIndex element_index = face_info.GetCellIndex();
   constexpr Integer space_dim = FiniteElementSpace::Dim;

   Permutation< space_dim > orientation = face_info.GetOrientation();
   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

   // TODO: Use fixed FIFO view and dynamic shared allocation through kernel_conf
   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   GENDIL_CHECK_MEMORY_ARENA_REQUEST( thread.SharedAllocator, data_size );
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
   thread.Synchronize();
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

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto DirectGlobalSerialReadDofs(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using DofShape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;
   auto local_dofs = MakeSerialRecursiveArray< Real >( DofShape{} );

   const GlobalIndex element_index = face_info.GetCellIndex();
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation< space_dim > orientation = face_info.GetOrientation();
   const auto dof_sizes = to_array( DofShape{} );

   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

   if ( FaceReadDofsOrientationIsIdentity( orientation ) )
   {
      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            local_dofs( indices... ) =
               global_dofs( indices..., element_index );
         }
      );
   }
   else
   {
      const auto oriented_global_dofs =
         MakeOrientedGlobalDofView(
            global_dofs,
            element_index,
            dof_sizes,
            orientation );

      DofLoop< FiniteElementSpace >(
         [&]( auto... indices )
         {
            local_dofs( indices... ) =
               oriented_global_dofs( indices... );
         }
      );
   }

   return local_dofs;
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto DirectGlobalThreadedReadDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using DofShape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;
   static_assert(
      threaded_shape_covered_v< KernelContext, DofShape >,
      "Under-threaded strided coverage is not supported by this threaded helper yet." );
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   const GlobalIndex element_index = face_info.GetCellIndex();
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation< space_dim > orientation = face_info.GetOrientation();
   const auto dof_sizes = to_array( DofShape{} );

   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

   if ( FaceReadDofsOrientationIsIdentity( orientation ) )
   {
      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... k )
         {
            local_dofs( k... ) =
               global_dofs( t..., k..., element_index );
         });
      });
   }
   else
   {
      const auto oriented_global_dofs =
         MakeOrientedGlobalDofView(
            global_dofs,
            element_index,
            dof_sizes,
            orientation );

      ThreadLoop< tshape >( thread, [&] ( auto... t )
      {
         UnitLoop< rshape >( [&] ( auto... k )
         {
            local_dofs( k... ) =
               oriented_global_dofs( t..., k... );
         });
      });
   }

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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ReadVectorDofsSerial(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   // static_assert(
   //    get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
   //    "Mismatching dimensions in ReadDofs."
   // );
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using rshape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   using DofShape =
      orders_to_num_dofs<
         typename FiniteElementSpace::finite_element_type::
            shape_functions::orders >;

   Permutation< space_dim > orientation = face_info.orientation;
   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

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
 * @brief Read vector DOFs from tuple storage (one tuple element per component) for facet.
 *
 * For vector FE with tuple-per-component storage, reads DOFs componentwise.
 * Each component's global DOFs are in a separate tuple element.
 * Returns tuple of local DOF arrays, one per component.
 *
 * Mirrors the cell tuple read pattern (lines 269-288) but for facets.
 *
 * @param face_info Face information (element index, orientation)
 * @param global_dofs Tuple of per-component global DOF views
 * @return Tuple of per-component local DOF arrays
 */
template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalTensor >
GENDIL_HOST_DEVICE
auto ReadVectorDofsSerial(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim = FiniteElementSpace::finite_element_type::shape_functions::vector_dim;
   using dof_shape = typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   // Allocate local DOF storage for each component
   auto local_dofs = MakeVectorDofs( dof_shape{}, std::make_index_sequence< v_dim >{} );

   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation< space_dim > orientation = face_info.orientation;

   // Process each component separately
   ConstexprLoop< v_dim >( [&]( auto i )
   {
      using component_dof_shape = std::tuple_element_t< i, dof_shape >;
      constexpr size_t data_size = Product( component_dof_shape{} );
      Real data[ data_size ];

      auto dofs_sizes = to_array( component_dof_shape{} );
      VerifyOrientedTensorDofShapeCompatibility< component_dof_shape >(
         orientation );
      auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );
      auto reference_view = MakeFIFOView( data, dofs_sizes );

      // Read with orientation
      UnitLoop< component_dof_shape >( [&]( auto... indices )
      {
         oriented_view( indices... ) = std::get< i >( global_dofs )( indices..., element_index );
      });

      // Copy to local with reference orientation
      UnitLoop< component_dof_shape >( [&]( auto... indices )
      {
         std::get< i >( local_dofs )( indices... ) = reference_view( indices... );
      });
   });

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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ReadVectorDofsThreaded(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   // static_assert(
   //    get_rank_v< GlobalTensor > == FiniteElementSpace::Dim + 1,
   //    "Mismatching dimensions in ReadDofs."
   // );
   static_assert(
      Dim == FiniteElementSpace::Dim + 1,
      "Mismatching dimensions in ReadDofs."
   );
   using DofShape = orders_to_num_dofs< typename FiniteElementSpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;

   Permutation< space_dim > orientation = face_info.orientation;
   VerifyOrientedTensorDofShapeCompatibility< DofShape >( orientation );

   constexpr size_t data_size = FiniteElementSpace::finite_element_type::GetNumDofs();
   GENDIL_CHECK_MEMORY_ARENA_REQUEST( thread.SharedAllocator, data_size );
   Real * data = thread.SharedAllocator.allocate( data_size );

   auto dofs_sizes = GetDofsSizes( typename FiniteElementSpace::finite_element_type::shape_functions{} );
   
   auto oriented_view = MakeOrientedView( data, dofs_sizes, orientation );
   auto reference_view = MakeFixedFIFOView( data, DofShape{} );
   
   auto local_dofs = MakeSerialRecursiveArray< Real >( rshape{} );

   // TODO: Study if oriented_view first or second is better ( invert orientation )
   ThreadLoop< tshape >( thread, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         oriented_view( t..., k... ) = global_dofs( t..., k..., element_index ); // Assumes threads < registers
      });
   });
   thread.Synchronize();
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

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalTensor >
GENDIL_HOST_DEVICE
auto ReadVectorDofsThreaded(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim =
      FiniteElementSpace::finite_element_type::shape_functions::vector_dim;

   using dof_shape =
      typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   auto local_dofs =
      MakeVectorDofs( thread, dof_shape{}, std::make_index_sequence<v_dim>{});

   const GlobalIndex element_index = face_info.cell_index;

   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation<space_dim> orientation = face_info.orientation;

   ConstexprLoop<v_dim>([&](auto i)
   {
      using component_dof_shape = std::tuple_element_t<i, dof_shape>;

      using tshape = subsequence_t<
         component_dof_shape,
         typename KernelContext::template threaded_dimensions<component_dof_shape::size()>
      >;

      using rshape = subsequence_t<
         component_dof_shape,
         typename KernelContext::template register_dimensions<component_dof_shape::size()>
      >;

      constexpr size_t data_size = Product(component_dof_shape{});

      // TODO: This shared-memory staging was added as a presumed optimization.
      // Direct reads through oriented indices may be faster and would avoid
      // this hidden shared-memory requirement.
      VerifyOrientedTensorDofShapeCompatibility< component_dof_shape >(
         orientation );
      GENDIL_CHECK_MEMORY_ARENA_REQUEST( thread.SharedAllocator, data_size );
      Real * data = thread.SharedAllocator.allocate( data_size );

      auto dofs_sizes = to_array(component_dof_shape{});

      auto oriented_view = MakeOrientedView(data, dofs_sizes, orientation);
      auto reference_view = MakeFIFOView(data, dofs_sizes);

      ThreadLoop<tshape>(thread, [&](auto... t)
      {
         UnitLoop<rshape>([&](auto... k)
         {
            oriented_view(t..., k...) =
               std::get<i>(global_dofs)(t..., k..., element_index);
         });
      });

      thread.Synchronize();

      ThreadLoop<tshape>(thread, [&](auto... t)
      {
         UnitLoop<rshape>([&](auto... k)
         {
            std::get<i>(local_dofs)(k...) =
               reference_view(t..., k...);
         });
      });

      thread.Synchronize();
      thread.SharedAllocator.reset();
   });

   return local_dofs;
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalTensor >
GENDIL_HOST_DEVICE
auto DirectGlobalReadVectorDofsSerial(
   const KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim =
      FiniteElementSpace::finite_element_type::shape_functions::vector_dim;

   using dof_shape =
      typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   auto local_dofs =
      MakeVectorDofs( dof_shape{}, std::make_index_sequence<v_dim>{});

   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation<space_dim> orientation = face_info.orientation;

   ConstexprLoop<v_dim>([&](auto i)
   {
      using component_dof_shape = std::tuple_element_t<i, dof_shape>;
      const auto dof_sizes = to_array(component_dof_shape{});

      VerifyOrientedTensorDofShapeCompatibility< component_dof_shape >(
         orientation );

      if ( FaceReadDofsOrientationIsIdentity( orientation ) )
      {
         UnitLoop<component_dof_shape>([&](auto... indices)
         {
            std::get<i>(local_dofs)(indices...) =
               std::get<i>(global_dofs)(indices..., element_index);
         });
      }
      else
      {
         const auto oriented_global_dofs =
            MakeOrientedGlobalDofView(
               std::get<i>(global_dofs),
               element_index,
               dof_sizes,
               orientation );

         UnitLoop<component_dof_shape>([&](auto... indices)
         {
            std::get<i>(local_dofs)(indices...) =
               oriented_global_dofs(indices...);
         });
      }
   });

   return local_dofs;
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalTensor >
GENDIL_HOST_DEVICE
auto DirectGlobalReadVectorDofsThreaded(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalTensor & global_dofs )
{
   constexpr Integer v_dim =
      FiniteElementSpace::finite_element_type::shape_functions::vector_dim;

   using dof_shape =
      typename FiniteElementSpace::finite_element_type::shape_functions::dof_shape;

   auto local_dofs =
      MakeVectorDofs( thread, dof_shape{}, std::make_index_sequence<v_dim>{});

   const GlobalIndex element_index = face_info.cell_index;
   constexpr Integer space_dim = FiniteElementSpace::Dim;
   Permutation<space_dim> orientation = face_info.orientation;

   ConstexprLoop<v_dim>([&](auto i)
   {
      using component_dof_shape = std::tuple_element_t<i, dof_shape>;

      using tshape = subsequence_t<
         component_dof_shape,
         typename KernelContext::template threaded_dimensions<
            component_dof_shape::size() > >;

      using rshape = subsequence_t<
         component_dof_shape,
         typename KernelContext::template register_dimensions<
            component_dof_shape::size() > >;

      const auto dof_sizes = to_array(component_dof_shape{});

      VerifyOrientedTensorDofShapeCompatibility< component_dof_shape >(
         orientation );

      if ( FaceReadDofsOrientationIsIdentity( orientation ) )
      {
         ThreadLoop<tshape>(thread, [&](auto... t)
         {
            UnitLoop<rshape>([&](auto... k)
            {
               std::get<i>(local_dofs)(k...) =
                  std::get<i>(global_dofs)(t..., k..., element_index);
            });
         });
      }
      else
      {
         const auto oriented_global_dofs =
            MakeOrientedGlobalDofView(
               std::get<i>(global_dofs),
               element_index,
               dof_sizes,
               orientation );

         ThreadLoop<tshape>(thread, [&](auto... t)
         {
            UnitLoop<rshape>([&](auto... k)
            {
               std::get<i>(local_dofs)(k...) =
                  oriented_global_dofs(t..., k...);
            });
         });
      }
   });

   return local_dofs;
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ReadVectorDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   if constexpr ( !is_threaded_v< KernelContext > )
   {
      return ReadVectorDofsSerial( thread, fe_space, face_info, global_dofs );
   }
   else
   {
      return ReadVectorDofsThreaded( thread, fe_space, face_info, global_dofs );
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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ReadScalarDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   using FaceReadPolicy =
      face_read_dofs_policy_t<
         typename KernelContext::kernel_configuration_type >;

   if constexpr ( !is_threaded_v< KernelContext > )
   {
      if constexpr (
         std::is_same_v<
            FaceReadPolicy,
            DirectGlobalFaceReadDofsPolicy > )
      {
         return DirectGlobalSerialReadDofs(
            thread,
            fe_space,
            face_info,
            global_dofs );
      }
      else
      {
         return SerialReadDofs( thread, fe_space, face_info, global_dofs );
      }
   }
   else
   {
      if constexpr (
         std::is_same_v<
            FaceReadPolicy,
            DirectGlobalFaceReadDofsPolicy > )
      {
         return DirectGlobalThreadedReadDofs(
            thread,
            fe_space,
            face_info,
            global_dofs );
      }
      else
      {
         return ThreadedReadDofs( thread, fe_space, face_info, global_dofs );
      }
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
   CellFaceView Face,
   Integer Dim,
   typename T >
GENDIL_HOST_DEVICE
auto ReadDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const StridedView< Dim, T > & global_dofs )
{
   return ReadScalarDofs( thread, fe_space, face_info, global_dofs );
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalDofs >
GENDIL_HOST_DEVICE
auto ReadVectorDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalDofs & global_dofs )
{
   using FaceReadPolicy =
      face_read_dofs_policy_t<
         typename KernelContext::kernel_configuration_type >;

   if constexpr ( !is_threaded_v< KernelContext > )
   {
      if constexpr (
         std::is_same_v<
            FaceReadPolicy,
            DirectGlobalFaceReadDofsPolicy > )
      {
         return DirectGlobalReadVectorDofsSerial(
            thread,
            fe_space,
            face_info,
            global_dofs );
      }
      else
      {
         return ReadVectorDofsSerial( thread, fe_space, face_info, global_dofs );
      }
   }
   else
   {
      if constexpr (
         std::is_same_v<
            FaceReadPolicy,
            DirectGlobalFaceReadDofsPolicy > )
      {
         return DirectGlobalReadVectorDofsThreaded(
            thread,
            fe_space,
            face_info,
            global_dofs );
      }
      else
      {
         return ReadVectorDofsThreaded( thread, fe_space, face_info, global_dofs );
      }
   }
}

template <
   typename KernelContext,
   typename FiniteElementSpace,
   CellFaceView Face,
   typename GlobalDofs >
GENDIL_HOST_DEVICE
auto ReadDofs(
   KernelContext & thread,
   const FiniteElementSpace & fe_space,
   const Face & face_info,
   const GlobalDofs & global_dofs )
{
   if constexpr ( is_vector_shape_functions_v< typename FiniteElementSpace::finite_element_type::shape_functions > )
   {
      return ReadVectorDofs( thread, fe_space, face_info, global_dofs );
   }
   else
   {
      return ReadScalarDofs( thread, fe_space, face_info, global_dofs );
   }
}


}
