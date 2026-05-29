// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>

// Manual sanitizer diagnostic:
// num_cells = 10 and BatchSize = 4 make the final block contain candidates
// 8, 9, 10, and 11. CellIterator filters candidates 10 and 11 before invoking
// the body below, while candidates 8 and 9 enter a body that calls
// SyncWorkItem(). DeviceKernelConfiguration::SyncWorkItem() currently maps to
// block-wide __syncthreads(), so this target is intentionally not a normal
// pass/fail correctness test. Run it manually with CUDA compute-sanitizer
// --tool synccheck when auditing the experimental filtered model.

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-cell-iterator-filtered-sync-experiment skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr GlobalIndex num_cells = 10;
constexpr long long sentinel = -987654321;

struct TinyMesh
{
   GlobalIndex num_cells_;

   GlobalIndex GetNumberOfCells() const
   {
      return num_cells_;
   }

   GENDIL_HOST_DEVICE
   GlobalIndex GetCell( const GlobalIndex cell_index ) const
   {
      return cell_index;
   }
};

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

template < typename Config >
bool RunFilteredSyncExperiment( const char * name )
{
   const GlobalIndex candidate_count =
      Config::GetLaunchGeometry( num_cells ).grid_x * Config::batch_size;

   DeviceBuffer< long long > before_sync( candidate_count, sentinel );
   DeviceBuffer< long long > after_sync( candidate_count, sentinel );

   auto before_sync_data = before_sync.data;
   auto after_sync_data = after_sync.data;

   TinyMesh mesh{ num_cells };

   mesh::CellIterator< Config >(
      mesh,
      [=] GENDIL_DEVICE ( auto kernel ) mutable
      {
         const GlobalIndex cell_index = kernel.WorkItemIndex();

         if ( kernel.GetLinearThreadIndex() == 0 )
         {
            before_sync_data[ cell_index ] =
               static_cast< long long >( 1000 + cell_index );
         }

         kernel.SyncWorkItem();

         if ( kernel.GetLinearThreadIndex() == 0 )
         {
            after_sync_data[ cell_index ] =
               static_cast< long long >( 2000 + cell_index );
         }
      } );

   GENDIL_DEVICE_SYNC;

   before_sync.CopyToHost();
   after_sync.CopyToHost();

   bool success = true;
   for ( GlobalIndex candidate = 0;
         candidate < candidate_count;
         ++candidate )
   {
      const bool expected_active = candidate < num_cells;
      if ( expected_active )
      {
         success = Check(
            before_sync.data.host_pointer[ candidate ] ==
               static_cast< long long >( 1000 + candidate ),
            "Active lane did not reach the pre-sync marker." ) && success;
         success = Check(
            after_sync.data.host_pointer[ candidate ] ==
               static_cast< long long >( 2000 + candidate ),
            "Active lane did not reach the post-sync marker." ) && success;
      }
      else
      {
         success = Check(
            before_sync.data.host_pointer[ candidate ] == sentinel,
            "Inactive lane entered the filtered sync experiment body." ) &&
            success;
         success = Check(
            after_sync.data.host_pointer[ candidate ] == sentinel,
            "Inactive lane wrote after the sync experiment." ) && success;
      }
   }

   if ( !success )
   {
      std::cout << "Failure in " << name << ".\n";
   }
   return success;
}

bool TestWarpAlignedLayout()
{
   using Layout = ThreadBlockLayout< 4, 8 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static constexpr Integer BatchSize = 4;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   static_assert( Layout::GetNumberOfThreads() == 32 );

   return RunFilteredSyncExperiment< Config >(
      "filtered CellIterator SyncWorkItem experiment" );
}
} // namespace

int main()
{
   return TestWarpAlignedLayout() ? 0 : 1;
}

#endif
