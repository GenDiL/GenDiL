// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-device-kernel-configuration-one-index-batched skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr GlobalIndex num_work_items = 5;
constexpr long long sentinel = -987654321;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

template < typename T >
struct DeviceBuffer
{
   HostDevicePointer< T > data;
   GlobalIndex size;

   DeviceBuffer( const GlobalIndex n, const T value )
      : size( n )
   {
      AllocateHostPointer( size, data );
      AllocateDevicePointer( size, data );
      Fill( value );
   }

   DeviceBuffer( const DeviceBuffer & ) = delete;
   DeviceBuffer & operator=( const DeviceBuffer & ) = delete;

   ~DeviceBuffer()
   {
      FreeDevicePointer( data );
      FreeHostPointer( data );
   }

   void Fill( const T value )
   {
      for ( GlobalIndex i = 0; i < size; ++i )
      {
         data.host_pointer[ i ] = value;
      }
      ToDevice( size, data );
   }

   void CopyToHost() const
   {
      ToHost( size, data );
   }
};
} // namespace

int main()
{
   using Config =
      DeviceKernelConfiguration< ThreadBlockLayout< 2 >, 1, 4 >;

   const GlobalIndex candidate_count =
      Config::GetLaunchGeometry( num_work_items ).grid_x *
      Config::batch_size;

   DeviceBuffer< long long > visits( candidate_count, sentinel );
   auto visits_data = visits.data;

   Config::BlockLoop(
      num_work_items,
      [=] GENDIL_DEVICE ( GlobalIndex work_item_index ) mutable
      {
         if ( Config::GetLinearThreadIndex() == 0 )
         {
            visits_data[ work_item_index ] =
               static_cast< long long >( 1000 + work_item_index );
         }
      } );

   GENDIL_DEVICE_SYNC;
   visits.CopyToHost();

   bool success = true;
   for ( GlobalIndex candidate = 0;
         candidate < candidate_count;
         ++candidate )
   {
      if ( candidate < num_work_items )
      {
         success = Check(
            visits.data.host_pointer[ candidate ] ==
               static_cast< long long >( 1000 + candidate ),
            "Active one-index BlockLoop candidate was not invoked." ) &&
            success;
      }
      else
      {
         success = Check(
            visits.data.host_pointer[ candidate ] == sentinel,
            "Inactive final-batch candidate invoked production BlockLoop body." ) &&
            success;
      }
   }

   return success ? 0 : 1;
}

#endif
