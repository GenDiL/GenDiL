// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/debug.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace gendil
{

namespace details
{

template < typename T >
struct IsIntegralConstant : std::false_type
{ };

template < typename T, T Value >
struct IsIntegralConstant< std::integral_constant< T, Value > >
   : std::true_type
{ };

template < typename T >
inline constexpr bool is_integral_constant_v =
   IsIntegralConstant< std::remove_cv_t< T > >::value;

} // namespace details

/**
 * Move-only owner of one logical array with host/device storage and validity.
 *
 * SizeType can be a runtime integral type or a std::integral_constant for a
 * compile-time extent. The fields intentionally remain public: this is a
 * low-level storage building block, while higher-level owners provide
 * cache-aware view acquisition.
 */
template < typename T, typename SizeType = GlobalIndex >
struct SyncHostDeviceArray
{
   static_assert(
      !std::is_const_v< T >,
      "SyncHostDeviceArray owns mutable storage and cannot have a const "
      "element type." );

   using value_type = T;
   using size_type = SizeType;

   HostDevicePointer< T > data{};
   [[no_unique_address]] SizeType size{};
   mutable bool host_valid = false;
   mutable bool device_valid = false;

   SyncHostDeviceArray() = default;
   SyncHostDeviceArray( const SyncHostDeviceArray & ) = delete;
   SyncHostDeviceArray & operator=( const SyncHostDeviceArray & ) = delete;

   SyncHostDeviceArray( SyncHostDeviceArray && other ) noexcept
   : data( std::move( other.data ) ),
     size( std::move( other.size ) ),
     host_valid( std::exchange( other.host_valid, false ) ),
     device_valid( std::exchange( other.device_valid, false ) )
   {
      if constexpr ( !details::is_integral_constant_v< SizeType > )
      {
         other.size = SizeType{};
      }
   }

   SyncHostDeviceArray & operator=( SyncHostDeviceArray && other ) noexcept
   {
      if ( this != &other )
      {
         Release();
         data = std::move( other.data );
         size = std::move( other.size );
         host_valid = std::exchange( other.host_valid, false );
         device_valid = std::exchange( other.device_valid, false );

         if constexpr ( !details::is_integral_constant_v< SizeType > )
         {
            other.size = SizeType{};
         }
      }
      return *this;
   }

   ~SyncHostDeviceArray() noexcept
   {
      Release();
   }

private:
   void Release() noexcept
   {
      FreeHostPointer( data );
      FreeDevicePointer( data );
      data.host_pointer = nullptr;
#if defined(GENDIL_USE_DEVICE)
      data.device_pointer = nullptr;
#endif
      if constexpr ( !details::is_integral_constant_v< SizeType > )
      {
         size = SizeType{};
      }
      host_valid = false;
      device_valid = false;
   }
};

template < typename T, typename SizeType >
size_t GetSize( const SyncHostDeviceArray< T, SizeType > & array )
{
   if constexpr ( details::is_integral_constant_v< SizeType > )
   {
      return static_cast< size_t >( SizeType::value );
   }
   else
   {
      if constexpr ( std::is_signed_v< SizeType > )
      {
         GENDIL_VERIFY(
            array.size >= SizeType( 0 ),
            "SyncHostDeviceArray requires a nonnegative runtime size." );
      }
      return static_cast< size_t >( array.size );
   }
}

template < typename T, typename SizeType >
auto MakeSyncHostDeviceArray( const SizeType size )
{
   SyncHostDeviceArray< T, SizeType > array{};
   array.size = size;
   const size_t count = GetSize( array );
   AllocateHostPointer( count, array.data );
   AllocateDevicePointer( count, array.data );
   return array;
}

template < typename T, typename SizeType >
bool IsHostValid( const SyncHostDeviceArray< T, SizeType > & array )
{
   return array.host_valid;
}

template < typename T, typename SizeType >
bool IsDeviceValid( const SyncHostDeviceArray< T, SizeType > & array )
{
   return array.device_valid;
}

template < typename T, typename SizeType >
const T * ReadHost( const SyncHostDeviceArray< T, SizeType > & array )
{
   const size_t count = GetSize( array );
   if ( count == 0 )
   {
      array.host_valid = true;
      return array.data.host_pointer;
   }

   GENDIL_VERIFY(
      array.host_valid || array.device_valid,
      "SyncHostDeviceArray data is not valid on either host or device." );

   if ( !array.host_valid && array.device_valid )
   {
      ToHost( static_cast< GlobalIndex >( count ), array.data );
      array.host_valid = true;
   }
   return array.data.host_pointer;
}

template < typename T, typename SizeType >
T * ReadWriteHost( SyncHostDeviceArray< T, SizeType > & array )
{
   ReadHost( array );
   array.device_valid = false;
   return array.data.host_pointer;
}

template < typename T, typename SizeType >
T * WriteHost( SyncHostDeviceArray< T, SizeType > & array )
{
   array.host_valid = true;
   array.device_valid = false;
   return array.data.host_pointer;
}

template < typename T, typename SizeType >
const T * ReadDevice( const SyncHostDeviceArray< T, SizeType > & array )
{
#if defined(GENDIL_USE_DEVICE)
   const size_t count = GetSize( array );
   if ( count == 0 )
   {
      array.host_valid = true;
      array.device_valid = true;
      return array.data.device_pointer;
   }

   GENDIL_VERIFY(
      array.host_valid || array.device_valid,
      "SyncHostDeviceArray data is not valid on either host or device." );

   if ( array.host_valid && !array.device_valid )
   {
      ToDevice( static_cast< GlobalIndex >( count ), array.data );
      array.device_valid = true;
   }
   return array.data.device_pointer;
#else
   return ReadHost( array );
#endif
}

template < typename T, typename SizeType >
T * ReadWriteDevice( SyncHostDeviceArray< T, SizeType > & array )
{
#if defined(GENDIL_USE_DEVICE)
   ReadDevice( array );
   array.host_valid = false;
   return array.data.device_pointer;
#else
   return ReadWriteHost( array );
#endif
}

template < typename T, typename SizeType >
T * WriteDevice( SyncHostDeviceArray< T, SizeType > & array )
{
#if defined(GENDIL_USE_DEVICE)
   array.host_valid = false;
   array.device_valid = true;
   return array.data.device_pointer;
#else
   return WriteHost( array );
#endif
}

template < typename T, typename SizeType >
void Sync( const SyncHostDeviceArray< T, SizeType > & array )
{
   const size_t count = GetSize( array );
   if ( count == 0 )
   {
      array.host_valid = true;
#if defined(GENDIL_USE_DEVICE)
      array.device_valid = true;
#endif
      return;
   }

   GENDIL_VERIFY(
      array.host_valid || array.device_valid,
      "SyncHostDeviceArray data is not valid on either host or device." );

#if defined(GENDIL_USE_DEVICE)
   if ( !array.host_valid )
   {
      ToHost( static_cast< GlobalIndex >( count ), array.data );
      array.host_valid = true;
   }
   if ( !array.device_valid )
   {
      ToDevice( static_cast< GlobalIndex >( count ), array.data );
      array.device_valid = true;
   }
#endif
}

} // namespace gendil
