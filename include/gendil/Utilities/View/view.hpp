// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/View/extents.hpp"
#include "Containers/containers.hpp"

#include <type_traits>
#include <utility>

namespace gendil {

namespace view_detail {

template < typename Container, typename Layout, typename... Indices >
concept RvalueContainerAccess = requires(
   Container && container,
   const Layout & layout,
   Indices... indices )
{
   std::move( container )[ layout.Offset( indices... ) ];
};

} // namespace view_detail

/**
 * @brief Lightweight logical view composed from access and indexing policies.
 *
 * `Layout` maps logical indices to a container key. `Container` interprets
 * that key and determines the returned access type, constness, and whether
 * access through a temporary view is safe. Both policies are stored by value;
 * ownership of element storage follows the container (borrowed for pointer
 * handles, inline-owned for StaticContainer).
 */
template < typename Container, typename Layout >
struct View
{
   using element_type = typename Container::element_type;

   static constexpr size_t rank = Layout::rank;

   Container data;
   Layout layout;

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()( Indices... idx ) const &
   {
      return data[ layout.Offset( idx... ) ];
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()( Indices... idx ) &
   {
      return data[ layout.Offset( idx... ) ];
   }

   template < typename... Indices >
      requires view_detail::RvalueContainerAccess<
         Container, Layout, Indices... >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()( Indices... idx ) &&
   {
      // The container decides whether access through a temporary is safe.
      return std::move( data )[ layout.Offset( idx... ) ];
   }

   // Prevent an unsafe temporary from falling back to the const-lvalue access.
   template < typename... Indices >
   GENDIL_HOST_DEVICE
   void operator()( Indices... ) && = delete;

   template < typename... Indices >
      requires view_detail::RvalueContainerAccess<
         const Container, Layout, Indices... >
   GENDIL_HOST_DEVICE
   decltype(auto) operator()( Indices... idx ) const &&
   {
      return std::move( data )[ layout.Offset( idx... ) ];
   }

   template < typename... Indices >
   GENDIL_HOST_DEVICE
   void operator()( Indices... ) const && = delete;

   template < typename Index >
      requires ( rank == 1 )
   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( Index index ) const &
   {
      return (*this)( index );
   }

   template < typename Index >
      requires ( rank == 1 )
   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( Index index ) &
   {
      return (*this)( index );
   }

   template < typename Index >
      requires (
         rank == 1 &&
         view_detail::RvalueContainerAccess<
            Container, Layout, Index > )
   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( Index index ) &&
   {
      return std::move( *this )( index );
   }

   template < typename Index >
   GENDIL_HOST_DEVICE
   void operator[]( Index ) && = delete;

   template < typename Index >
      requires (
         rank == 1 &&
         view_detail::RvalueContainerAccess<
            const Container, Layout, Index > )
   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( Index index ) const &&
   {
      return std::move( *this )( index );
   }

   template < typename Index >
   GENDIL_HOST_DEVICE
   void operator[]( Index ) const && = delete;

   template < size_t Axis >
      requires requires( const Layout & candidate )
      {
         candidate.Get( AxisTag< Axis >{} );
      }
   GENDIL_HOST_DEVICE
   constexpr decltype(auto) Get( AxisTag< Axis > axis ) const
   {
      return layout.Get( axis );
   }

   GENDIL_HOST_DEVICE
   constexpr GlobalIndex Size() const
      requires has_complete_extents_v< Layout >
   {
      return GetLogicalSize( layout );
   }
};

namespace view_detail {

template < typename T >
struct is_view_impl : std::false_type { };

template < typename Container, typename Layout >
struct is_view_impl< View< Container, Layout > > : std::true_type { };

} // namespace view_detail

/** Whether a possibly cv/ref-qualified type is a GenDiL `View`. */
template < typename T >
struct is_view : view_detail::is_view_impl< std::remove_cvref_t< T > > { };

template < typename T >
inline constexpr bool is_view_v = is_view< T >::value;

template < typename Container, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( const Container & data, const Layout & layout )
{
   return View< Container, Layout >{ data, layout };
}

template < typename T, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( T* const data, Layout layout )
{
   return MakeView( PointerContainer< T >{ data }, layout );
}

template < typename T, typename Layout >
GENDIL_HOST_DEVICE
auto MakeView( const HostDevicePointer< T > & data, const Layout & layout )
{
   return View< HostDevicePointer< T >, Layout >{ data, layout };
}

}
