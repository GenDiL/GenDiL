// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file orientedglobaldofview.hpp
 * @brief Reference-oriented access to element-indexed global DoF views.
 *
 * Explicitly strided views use a signed-stride adapter whose orientation is
 * constructed once. Other views use a layout-independent adapter that maps
 * each reference index to its native index on access.
 */

#include <array>
#include <type_traits>
#include <utility>

#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/orienteddofvalidation.hpp"
#include "gendil/Meshes/Connectivities/Orientations/referencetonativeindex.hpp"
#include "gendil/Utilities/getrank.hpp"
#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"

namespace gendil {

namespace detail {

using OrientedGlobalDofSignedIndex = std::make_signed_t< GlobalIndex >;

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
GENDIL_HOST_DEVICE GENDIL_INLINE
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
GENDIL_HOST_DEVICE GENDIL_INLINE
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

/**
 * @brief Signed-stride reference-oriented view of one element's global DoFs.
 *
 * Construction incorporates the element offset, axis permutation, and axis
 * reversals into base_offset and strides. Access therefore requires only one
 * signed dot product and no orientation lookup.
 */
template < typename GlobalDofsView, Integer Dim >
struct OrientedGlobalDofView
{
   GlobalDofsView global_dofs;
   GlobalIndex element_index;
   detail::OrientedGlobalDofSignedIndex base_offset;
   std::array< detail::OrientedGlobalDofSignedIndex, Dim > strides;

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   detail::OrientedGlobalDofSignedIndex Offset( Indices... indices ) const
   {
      static_assert(
         sizeof...( Indices ) == Dim,
         "Wrong number of arguments." );

      detail::OrientedGlobalDofSignedIndex offset = base_offset;
      Integer axis = 0;
      ( ( offset +=
             static_cast< detail::OrientedGlobalDofSignedIndex >( indices ) *
             strides[ axis++ ] ), ... );
      return offset;
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices )
   {
      const auto offset = Offset( indices... );
      return global_dofs.data[ static_cast< GlobalIndex >( offset ) ];
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE GENDIL_INLINE
   decltype(auto) operator()( Indices... indices ) const
   {
      const auto offset = Offset( indices... );
      return global_dofs.data[ static_cast< GlobalIndex >( offset ) ];
   }
};

namespace detail {

/** @brief Layout-independent reference-to-native local DoF adapter. */
template <
   typename GlobalDofsView,
   Integer Dim,
   class Orientation = Permutation< Dim > >
struct GenericOrientedGlobalDofView
{
   GlobalDofsView global_dofs;
   GlobalIndex element_index;
   std::array< size_t, Dim > native_sizes;
   Orientation orientation;

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
      return OrientedGlobalDofValueAt(
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

template < typename GlobalDofsView, Integer Dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeGenericOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Orientation & orientation )
{
   static_assert(
      orientation_dimension_v< Orientation > ==
         static_cast< size_t >( Dim ) );
   static_assert(
      get_rank_v< GlobalDofsView > == Dim + 1,
      "An oriented element-indexed view must have rank Dim + 1." );
   GENDIL_ASSERT(
      OrientedTensorDofShapeIsCompatible( sizes, orientation ),
      "Invalid or extent-incompatible local DOF orientation." );
   return GenericOrientedGlobalDofView<
      GlobalDofsView,
      Dim,
      std::remove_cvref_t< Orientation > >{
      global_dofs, element_index, sizes, orientation };
}

template < size_t Offset, typename GlobalDofsView, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
void FillStridedOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const std::array< size_t, Dim > & sizes,
   const Permutation< Rank > & orientation,
   OrientedGlobalDofView< GlobalDofsView, Dim > & view )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   for ( size_t native_axis = 0; native_axis < Rank; ++native_axis )
   {
      const LocalIndex mapped_axis = orientation( native_axis );
      const size_t reference_axis = static_cast< size_t >(
         mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
      const size_t native_global_axis = Offset + native_axis;
      const auto native_stride =
         static_cast< OrientedGlobalDofSignedIndex >(
            global_dofs.layout.strides[ native_global_axis ] );
      if ( mapped_axis < 0 )
      {
         view.base_offset += static_cast< OrientedGlobalDofSignedIndex >(
            sizes[ native_global_axis ] - 1 ) * native_stride;
      }
      view.strides[ Offset + reference_axis ] =
         mapped_axis < 0 ? -native_stride : native_stride;
   }
}

template < size_t Offset, typename GlobalDofsView, Integer Dim, size_t Rank >
GENDIL_HOST_DEVICE GENDIL_INLINE
void FillStridedOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const std::array< size_t, Dim > &,
   const IdentityOrientation< Rank > &,
   OrientedGlobalDofView< GlobalDofsView, Dim > & view )
{
   static_assert( Offset + Rank <= static_cast< size_t >( Dim ) );
   gendil::ConstexprLoop< Rank >( [&] ( auto axis )
   {
      view.strides[ Offset + axis ] =
         static_cast< OrientedGlobalDofSignedIndex >(
            global_dofs.layout.strides[ Offset + axis ] );
   } );
}

template <
   size_t Offset,
   typename GlobalDofsView,
   Integer Dim,
   class... Orientations >
GENDIL_HOST_DEVICE GENDIL_INLINE
void FillStridedOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const std::array< size_t, Dim > & sizes,
   const TensorProductOrientation< Orientations... > & orientation,
   OrientedGlobalDofView< GlobalDofsView, Dim > & view )
{
   static_assert( Offset == 0 );
   static_assert(
      TensorProductOrientation< Orientations... >::Dim ==
         static_cast< size_t >( Dim ) );
   gendil::ConstexprLoop< Dim >( [&] ( auto native_axis )
   {
      const auto mapped_axis =
         detail::GetStructuredOrientationAxis< native_axis, 0 >(
            orientation );
      const auto native_stride =
         static_cast< OrientedGlobalDofSignedIndex >(
            global_dofs.layout.strides[ native_axis ] );
      if constexpr (
         detail::is_static_orientation_axis_v< decltype( mapped_axis ) > )
      {
         view.strides[ native_axis ] = native_stride;
      }
      else
      {
         const size_t reference_axis = static_cast< size_t >(
            mapped_axis > 0 ? mapped_axis - 1 : -mapped_axis - 1 );
         if ( mapped_axis < 0 )
         {
            view.base_offset += static_cast< OrientedGlobalDofSignedIndex >(
               sizes[ native_axis ] - 1 ) * native_stride;
         }
         view.strides[ reference_axis ] =
            mapped_axis < 0 ? -native_stride : native_stride;
      }
   } );
}

template < typename GlobalDofsView, Integer Dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeStridedOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Orientation & orientation )
{
   static_assert(
      orientation_dimension_v< Orientation > ==
         static_cast< size_t >( Dim ) );
   static_assert(
      get_rank_v< GlobalDofsView > == Dim + 1,
      "An oriented element-indexed view must have rank Dim + 1." );
   GENDIL_ASSERT(
      OrientedTensorDofShapeIsCompatible( sizes, orientation ),
      "Invalid or extent-incompatible local DOF orientation." );

   OrientedGlobalDofView< GlobalDofsView, Dim > view{
      global_dofs,
      element_index,
      static_cast< OrientedGlobalDofSignedIndex >( element_index ) *
         static_cast< OrientedGlobalDofSignedIndex >(
            global_dofs.layout.strides[ Dim ] ),
      {} };

   FillStridedOrientedGlobalDofView< 0 >(
      global_dofs,
      sizes,
      orientation,
      view );
   return view;
}

} // namespace detail

/**
 * @brief Create a reference-oriented view of one element's global DoFs.
 *
 * Explicit StridedLayout views select the signed-stride fast path. All other
 * views preserve their accessor semantics through the generic adapter.
 */
template < typename GlobalDofsView, Integer Dim, class Orientation >
GENDIL_HOST_DEVICE GENDIL_INLINE
auto MakeOrientedGlobalDofView(
   const GlobalDofsView & global_dofs,
   const GlobalIndex element_index,
   const std::array< size_t, Dim > & sizes,
   const Orientation & orientation )
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

} // namespace gendil
