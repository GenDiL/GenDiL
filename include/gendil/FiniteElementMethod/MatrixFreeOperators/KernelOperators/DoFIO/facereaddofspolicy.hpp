// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Utilities/getrank.hpp"
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

struct FullSharedFaceWriteDofsPolicy
{
};

struct DirectGlobalFaceWriteDofsPolicy
{
};

template < typename KernelConfiguration, typename = void >
struct face_read_dofs_policy
{
   using type = DirectGlobalFaceReadDofsPolicy;
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

template < typename KernelConfiguration, typename = void >
struct face_write_dofs_policy
{
   using type = DirectGlobalFaceWriteDofsPolicy;
};

template < typename KernelConfiguration >
struct face_write_dofs_policy<
   KernelConfiguration,
   std::void_t< typename KernelConfiguration::face_write_dofs_policy > >
{
   using type = typename KernelConfiguration::face_write_dofs_policy;
};

template < typename KernelConfiguration >
using face_write_dofs_policy_t =
   typename face_write_dofs_policy<
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

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool FaceReadDofsOrientationIsValid(
   const Permutation< Dim > & orientation )
{
   std::array< bool, Dim > seen{};
   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex o = orientation( native_axis );
      if ( o == 0 )
      {
         return false;
      }

      const Integer reference_axis = static_cast< Integer >(
         o > 0 ? o - 1 : -( o + 1 ) );
      if ( reference_axis >= Dim || seen[ reference_axis ] )
      {
         return false;
      }
      seen[ reference_axis ] = true;
   }
   return true;
}

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool OrientedTensorDofExtentsAreCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   for ( Integer native_axis = 0; native_axis < Dim; ++native_axis )
   {
      const LocalIndex o = orientation( native_axis );
      const Integer reference_axis = static_cast< Integer >(
         o > 0 ? o - 1 : -( o + 1 ) );
      if ( reference_axis >= Dim ||
           sizes[ native_axis ] != sizes[ reference_axis ] )
      {
         return false;
      }
   }
   return true;
}

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool OrientedTensorDofShapeIsCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   return FaceReadDofsOrientationIsValid( orientation ) &&
      OrientedTensorDofExtentsAreCompatible( sizes, orientation );
}

template < typename DofShape, Integer Dim >
GENDIL_HOST_DEVICE
void VerifyOrientedTensorDofShapeCompatibility(
   const Permutation< Dim > & orientation )
{
   static_assert(
      DofShape::size() == Dim,
      "Mismatching oriented tensor-product DOF shape and orientation dimensions." );

   const auto sizes = to_array( DofShape{} );
   const bool valid = FaceReadDofsOrientationIsValid( orientation );
   GENDIL_VERIFY(
      valid,
      "DOF orientation must be a signed permutation." );
   if ( valid )
   {
      GENDIL_VERIFY(
         OrientedTensorDofExtentsAreCompatible( sizes, orientation ),
         "DOF orientation permutes local tensor axes with incompatible extents." );
   }
}

/**
 * @brief Return whether the temporary oriented tensor-product DOF rule accepts
 * the shape/orientation pair.
 *
 * @details A valid orientation is a signed permutation. Each native local-DOF
 * axis must have the same extent as the reference axis mapped onto it.
 */
template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool FaceReadDofsOrientationIsShapeCompatible(
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   return OrientedTensorDofShapeIsCompatible( sizes, orientation );
}

using FaceReadDofsSignedIndex = std::make_signed_t< GlobalIndex >;

namespace detail
{

template < typename ViewType >
struct is_explicitly_strided_element_view : std::false_type
{
};

template < typename Container, Integer Rank >
struct is_explicitly_strided_element_view<
   View< Container, StridedLayout< Rank > > > : std::true_type
{
};

template < typename ViewType >
inline constexpr bool is_explicitly_strided_element_view_v =
   is_explicitly_strided_element_view<
      std::remove_cvref_t< ViewType > >::value;

template < typename ViewType, Integer Dim, size_t... Is >
GENDIL_HOST_DEVICE
decltype(auto) OrientedGlobalDofValueAt(
   ViewType && global_dofs,
   const std::array< Integer, Dim > & native_indices,
   const GlobalIndex element_index,
   std::index_sequence< Is... > )
{
   return std::forward< ViewType >( global_dofs )(
      native_indices[ Is ]..., element_index );
}

template < typename ViewType, Integer Dim >
GENDIL_HOST_DEVICE
decltype(auto) OrientedGlobalDofValueAt(
   ViewType && global_dofs,
   const std::array< Integer, Dim > & native_indices,
   const GlobalIndex element_index )
{
   return OrientedGlobalDofValueAt(
      std::forward< ViewType >( global_dofs ),
      native_indices,
      element_index,
      std::make_index_sequence< Dim >{} );
}

} // namespace detail

template < typename GlobalDofsView, Integer Dim >
struct OrientedGlobalDofView
{
   GlobalDofsView global_dofs;
   GlobalIndex element_index;
   FaceReadDofsSignedIndex base_offset;
   std::array< FaceReadDofsSignedIndex, Dim > strides;

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   FaceReadDofsSignedIndex Offset( Indices... indices ) const
   {
      static_assert(
         sizeof...( Indices ) == Dim,
         "Wrong number of arguments." );

      FaceReadDofsSignedIndex offset = base_offset;
      Integer axis = 0;
      ( ( offset +=
             static_cast< FaceReadDofsSignedIndex >( indices ) *
             strides[ axis++ ] ), ... );

      return offset;
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices )
   {
      const FaceReadDofsSignedIndex offset = Offset( indices... );
      return global_dofs.data[ static_cast< GlobalIndex >( offset ) ];
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices ) const
   {
      const FaceReadDofsSignedIndex offset = Offset( indices... );
      return global_dofs.data[ static_cast< GlobalIndex >( offset ) ];
   }
};

namespace detail
{

/**
 * @brief Layout-independent canonical-to-native local DOF view.
 */
template < typename GlobalDofsView, Integer Dim >
struct GenericOrientedGlobalDofView
{
   GlobalDofsView global_dofs;
   GlobalIndex element_index;
   std::array< size_t, Dim > native_sizes;
   Permutation< Dim > orientation;

private:
   template < typename Self, typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   static decltype(auto) Access( Self && self, Indices... indices )
   {
      static_assert(
         sizeof...( Indices ) == Dim,
         "Wrong number of local DOF indices." );
      const std::array< Integer, Dim > reference_indices{
         static_cast< Integer >( indices )... };
      const auto native_indices = ReferenceToNativeIndex(
         reference_indices,
         self.native_sizes,
         self.orientation );
      return detail::OrientedGlobalDofValueAt(
         std::forward< Self >( self ).global_dofs,
         native_indices,
         self.element_index );
   }

public:
   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices )
   {
      return Access( *this, indices... );
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices ) const
   {
      return Access( *this, indices... );
   }
};

template < typename GlobalDofsView, Integer Dim >
GENDIL_HOST_DEVICE
auto MakeGenericOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   static_assert(
      get_rank_v< GlobalDofsView > == Dim + 1,
      "An oriented element-indexed view must have rank Dim + 1." );
   GENDIL_VERIFY(
      OrientedTensorDofShapeIsCompatible( sizes, orientation ),
      "Invalid or extent-incompatible local DOF orientation." );
   return GenericOrientedGlobalDofView< GlobalDofsView, Dim >{
      global_dofs, element_index, sizes, orientation };
}

template < typename GlobalDofsView, Integer Dim >
GENDIL_HOST_DEVICE
auto MakeStridedOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   static_assert(
      get_rank_v< GlobalDofsView > == Dim + 1,
      "An oriented element-indexed view must have rank Dim + 1." );
   GENDIL_VERIFY(
      OrientedTensorDofShapeIsCompatible( sizes, orientation ),
      "Invalid or extent-incompatible local DOF orientation." );

   OrientedGlobalDofView< GlobalDofsView, Dim > view{
      global_dofs,
      element_index,
      static_cast< FaceReadDofsSignedIndex >( element_index ) *
         static_cast< FaceReadDofsSignedIndex >(
            global_dofs.layout.strides[ Dim ] ),
      {} };

   if constexpr ( Dim == 1 )
   {
      const LocalIndex o = orientation( 0 );
      GENDIL_ASSERT(
         o == 1 || o == -1,
         "Invalid 1D face orientation." );

      const FaceReadDofsSignedIndex native_stride =
         static_cast< FaceReadDofsSignedIndex >(
            global_dofs.layout.strides[ 0 ] );
      if ( o < 0 )
      {
         view.base_offset +=
            static_cast< FaceReadDofsSignedIndex >(
               sizes[ 0 ] - 1 ) *
            native_stride;
      }
      view.strides[ 0 ] = o < 0 ? -native_stride : native_stride;
   }
   else
   {
      for ( Integer native_dim = 0; native_dim < Dim; ++native_dim )
      {
         const LocalIndex o = orientation( native_dim );
         const Integer reference_axis =
            static_cast< Integer >( o > 0 ? o - 1 : -o - 1 );
         const FaceReadDofsSignedIndex native_stride =
            static_cast< FaceReadDofsSignedIndex >(
               global_dofs.layout.strides[ native_dim ] );

         GENDIL_ASSERT(
            reference_axis >= 0 && reference_axis < static_cast< Integer >( Dim ),
            "Invalid face orientation axis." );

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
   }

   return view;
}

} // namespace detail

template < typename GlobalDofsView, Integer Dim >
GENDIL_HOST_DEVICE
auto MakeOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Dim > & orientation )
{
   if constexpr (
      detail::is_explicitly_strided_element_view_v< GlobalDofsView > )
   {
      return detail::MakeStridedOrientedGlobalDofView(
         global_dofs, element_index, sizes, orientation );
   }
   else
   {
      return detail::MakeGenericOrientedGlobalDofView(
         global_dofs, element_index, sizes, orientation );
   }
}

template < typename ViewType, Integer Dim >
GENDIL_HOST_DEVICE
decltype(auto) FaceReadDofsGlobalValueAt(
   ViewType && global_dofs,
   const std::array< GlobalIndex, Dim > & native_indices,
   const GlobalIndex element_index )
{
   return detail::OrientedGlobalDofValueAt(
      std::forward< ViewType >( global_dofs ),
      native_indices,
      element_index );
}

} // namespace gendil
