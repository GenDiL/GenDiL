// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <type_traits>

#include "gendil/Meshes/Connectivities/orientation.hpp"
#include "gendil/Utilities/debug.hpp"
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

template < typename DofShape >
struct is_isotropic_shape;

template <>
struct is_isotropic_shape< std::index_sequence<> >
{
   static constexpr bool value = true;
};

template < size_t First >
struct is_isotropic_shape< std::index_sequence< First > >
{
   static constexpr bool value = true;
};

template < size_t First, size_t Second, size_t... Rest >
struct is_isotropic_shape< std::index_sequence< First, Second, Rest... > >
{
   static constexpr bool value =
      ( Second == First ) && ( ( Rest == First ) && ... );
};

template < typename DofShape >
inline constexpr bool is_isotropic_shape_v =
   is_isotropic_shape< DofShape >::value;

template < Integer Dim >
GENDIL_HOST_DEVICE
constexpr bool OrientedTensorDofShapeIsIsotropic(
   const std::array< size_t, Dim > & sizes )
{
   for ( Integer i = 1; i < Dim; ++i )
   {
      if ( sizes[ i ] != sizes[ 0 ] )
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
   return OrientedTensorDofShapeIsIsotropic( sizes ) ||
      FaceReadDofsOrientationIsIdentity( orientation );
}

template < typename DofShape, Integer Dim >
GENDIL_HOST_DEVICE
void VerifyOrientedTensorDofShapeCompatibility(
   const Permutation< Dim > & orientation )
{
   static_assert(
      DofShape::size() == Dim,
      "Mismatching oriented tensor-product DOF shape and orientation dimensions." );

   if constexpr ( !is_isotropic_shape_v< DofShape > )
   {
      GENDIL_VERIFY(
         FaceReadDofsOrientationIsIdentity( orientation ),
         "anisotropic tensor-product DOF shapes currently support only identity face orientation; non-identity orientation requires future tensor-shape/basis-transform support." );
   }
}

/**
 * @brief Return whether the temporary oriented tensor-product DOF rule accepts
 * the shape/orientation pair.
 *
 * @details This conservative rule is policy-independent and applies to both
 * face ReadDofs and WriteDofs: isotropic tensor-product DOF shapes support all
 * orientations; anisotropic shapes currently support identity orientation only.
 * Non-identity anisotropic orientation requires a future tensor-shape/basis
 * transform abstraction.
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
