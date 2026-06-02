// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/multiindex.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"

namespace gendil {

struct FullSharedFaceReadDofsPolicy
{
};

struct DirectGlobalFaceReadDofsPolicy
{
};

template < typename KernelConfiguration, typename = void >
struct face_read_dofs_policy
{
   using type = FullSharedFaceReadDofsPolicy;
};

template < typename KernelConfiguration >
struct face_read_dofs_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_read_dofs_policy > >
{
   using type = typename KernelConfiguration::face_read_dofs_policy;
};

template < typename KernelConfiguration >
using face_read_dofs_policy_t =
   typename face_read_dofs_policy<
      std::remove_cvref_t< KernelConfiguration > >::type;

/**
 * @brief FIFO/reference offset used by MakeFIFOView/MakeFixedFIFOView.
 *
 * @details The mixed-radix order is
 *
 *   offset = i0 + sizes[0] * i1
 *              + sizes[0] * sizes[1] * i2 + ...
 */
template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr GlobalIndex FaceReadDofsFIFOOffset(
   const std::array< GlobalIndex, Dim > & reference_indices,
   const std::array< size_t, Dim > & sizes )
{
   GlobalIndex offset = 0;
   GlobalIndex stride = 1;
   for ( Integer i = 0; i < Dim; ++i )
   {
      offset += reference_indices[ i ] * stride;
      stride *= static_cast< GlobalIndex >( sizes[ i ] );
   }
   return offset;
}

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool FaceReadDofsOrientationIsIdentity(
   const Permutation< Dim > & orientation )
{
   for ( Integer i = 0; i < Dim; ++i )
   {
      if ( orientation( i ) != static_cast< LocalIndex >( i + 1 ) )
      {
         return false;
      }
   }
   return true;
}

/**
 * @brief Return whether an orientation can be represented as an affine view
 * over the original reference shape.
 *
 * @details For native dimension j, abs(orientation(j)) - 1 is the reference
 * axis populated by that native dimension.  The optimized DirectGlobal face
 * read path supports flips and swaps only when the native extent matches the
 * target reference-axis extent.  This keeps each access to a signed/permuted
 * affine stride instead of an inverse mixed-radix lookup.
 */
template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool FaceReadDofsOrientationIsShapeCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   for ( Integer native_dim = 0; native_dim < Dim; ++native_dim )
   {
      const LocalIndex o = orientation( native_dim );
      const Integer reference_axis =
         static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
      if ( sizes[ native_dim ] != sizes[ reference_axis ] )
      {
         return false;
      }
   }
   return true;
}

using FaceReadDofsSignedIndex = std::make_signed_t< GlobalIndex >;

template < typename GlobalDofsView, Integer Dim >
struct OrientedGlobalDofView
{
   GlobalDofsView global_dofs;
   GlobalIndex element_index;
   FaceReadDofsSignedIndex base_offset;
   std::array< FaceReadDofsSignedIndex, Dim > strides;

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices ) const
   {
      static_assert(
         sizeof...( Indices ) == Dim,
         "Wrong number of arguments." );

      FaceReadDofsSignedIndex offset = base_offset;
      Integer axis = 0;
      ( ( offset +=
             static_cast< FaceReadDofsSignedIndex >( indices ) *
             strides[ axis++ ] ), ... );

      return global_dofs.data[ static_cast< GlobalIndex >( offset ) ];
   }
};

template < typename GlobalDofsView, Integer Dim >
GENDIL_HOST_DEVICE
auto MakeOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   OrientedGlobalDofView< GlobalDofsView, Dim > view{
      global_dofs,
      element_index,
      static_cast< FaceReadDofsSignedIndex >( element_index ) *
         static_cast< FaceReadDofsSignedIndex >(
            global_dofs.layout.strides[ Dim ] ),
      {} };

   for ( Integer native_dim = 0; native_dim < Dim; ++native_dim )
   {
      const LocalIndex o = orientation( native_dim );
      const Integer reference_axis =
         static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
      const FaceReadDofsSignedIndex native_stride =
         static_cast< FaceReadDofsSignedIndex >(
            global_dofs.layout.strides[ native_dim ] );

      if ( o < 0 )
      {
         view.base_offset +=
            static_cast< FaceReadDofsSignedIndex >(
               sizes[ native_dim ] - 1 ) *
            native_stride;
         view.strides[ reference_axis ] = -native_stride;
      }
      else
      {
         view.strides[ reference_axis ] = native_stride;
      }
   }

   return view;
}

template < typename View, Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE
decltype(auto) FaceReadDofsGlobalValueAt(
   const View & global_dofs,
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index,
   std::index_sequence< Is... > )
{
   return global_dofs( native_indices[ Is ]..., element_index );
}

template < typename View, Integer Dim >
GENDIL_HOST_DEVICE
decltype(auto) FaceReadDofsGlobalValueAt(
   const View & global_dofs,
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index )
{
   return FaceReadDofsGlobalValueAt(
      global_dofs,
      native_indices,
      element_index,
      std::make_index_sequence< Dim >{} );
}

} // namespace gendil
