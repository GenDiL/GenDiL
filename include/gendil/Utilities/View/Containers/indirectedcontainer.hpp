// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <utility>

namespace gendil {

namespace detail {

   template < typename T >
   struct get_element_type
   {
      using type = typename T::element_type;
   };

   template < typename T >
   struct get_element_type< T* >
   {
      using type = T;
   };

   template < typename T >
   using get_element_type_t = typename get_element_type< T >::type;

   template < typename Container, typename Index >
   concept IndirectedContainerRvalueAccess = requires(
      Container && container,
      const HostDevicePointer< const int > & indirections,
      const Index & index )
   {
      std::move( container )[ indirections[ index ] ];
   };

}

/**
 * @brief Remap indices before delegating access to a child container.
 *
 * The child determines the access result: an ordinary reference, a value, or
 * a proxy. Its constness and value category are preserved during delegation;
 * this adaptor does not impose deep constness on a shallow-const child.
 *
 * Rvalue access is available only when the child supports access in that
 * category. Deleted fallbacks prevent a temporary from using const-lvalue
 * access instead. References and proxies retain the child's lifetime rules.
 *
 * The child is stored by value, while the indirection map is borrowed. For
 * GPU execution, both the child storage and the map must be device-accessible.
 * Child copying follows its own ownership semantics; the adaptor performs no
 * memory transfer or synchronization.
 */
template < typename Container, typename IndexType = GlobalIndex >
struct IndirectedContainer
{
   using element_type = detail::get_element_type_t< Container >;
   using index_type = IndexType;

   Container data;
   HostDevicePointer< const int > indirections;

   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( index_type const & index ) &
   {
      return data[ indirections[ index ] ];
   }

   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( index_type const & index ) const &
   {
      return data[ indirections[ index ] ];
   }

   /** Propagate temporary-access safety from the underlying container. */
   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( index_type const & index ) &&
      requires detail::IndirectedContainerRvalueAccess<
         Container, index_type >
   {
      return std::move( data )[ indirections[ index ] ];
   }

   GENDIL_HOST_DEVICE
   void operator[]( index_type const & ) &&
      requires ( !detail::IndirectedContainerRvalueAccess<
         Container, index_type > ) = delete;

   GENDIL_HOST_DEVICE
   decltype(auto) operator[]( index_type const & index ) const &&
      requires detail::IndirectedContainerRvalueAccess<
         const Container, index_type >
   {
      return std::move( data )[ indirections[ index ] ];
   }

   GENDIL_HOST_DEVICE
   void operator[]( index_type const & ) const &&
      requires ( !detail::IndirectedContainerRvalueAccess<
         const Container, index_type > ) = delete;
};

}
