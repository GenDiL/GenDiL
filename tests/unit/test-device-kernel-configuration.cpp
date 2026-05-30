// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-device-kernel-configuration skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr GlobalIndex num_work_items = 10;
constexpr long long sentinel = -987654321;
constexpr Real real_sentinel = -987654321.0;
constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

GENDIL_HOST_DEVICE
inline Real SharedValue(
   const GlobalIndex work_item_index,
   const GlobalIndex batch_index,
   const GlobalIndex shared_index )
{
   return Real( 1000 ) +
      Real( 100 * work_item_index ) +
      Real( 10 * batch_index ) +
      Real( shared_index );
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
GENDIL_HOST_DEVICE
inline GlobalIndex CandidateCount()
{
   return Config::GetLaunchGeometry( num_work_items ).grid_x *
      Config::batch_size;
}

template < typename Config >
bool RunSmokeCase( const char * name )
{
   using Layout = typename Config::thread_layout_type;

   static constexpr GlobalIndex threads_per_work_item =
      Config::GetNumberOfThreads();
   static constexpr GlobalIndex shared_size = 8;

   const GlobalIndex candidate_count = CandidateCount< Config >();
   const GlobalIndex thread_output_count =
      candidate_count * threads_per_work_item;

   DeviceBuffer< long long > batch_index( candidate_count, sentinel );
   DeviceBuffer< long long > work_item_index( candidate_count, sentinel );
   DeviceBuffer< long long > active_flag( candidate_count, sentinel );
   DeviceBuffer< long long > guarded_output( candidate_count, sentinel );
   DeviceBuffer< long long > post_sync_status( candidate_count, sentinel );
   DeviceBuffer< long long > shared_ok( candidate_count, sentinel );
   DeviceBuffer< long long > shared_offset( candidate_count, sentinel );
   DeviceBuffer< Real > shared_checksum(
      candidate_count,
      real_sentinel );

   DeviceBuffer< long long > linear_index(
      thread_output_count,
      sentinel );
   DeviceBuffer< long long > thread_index_0(
      thread_output_count,
      sentinel );
   DeviceBuffer< long long > thread_index_1(
      thread_output_count,
      sentinel );

   auto batch_index_data = batch_index.data;
   auto work_item_index_data = work_item_index.data;
   auto active_flag_data = active_flag.data;
   auto guarded_output_data = guarded_output.data;
   auto post_sync_status_data = post_sync_status.data;
   auto shared_ok_data = shared_ok.data;
   auto shared_offset_data = shared_offset.data;
   auto shared_checksum_data = shared_checksum.data;
   auto linear_index_data = linear_index.data;
   auto thread_index_0_data = thread_index_0.data;
   auto thread_index_1_data = thread_index_1.data;

   Config::BlockLoop(
      num_work_items,
      [=] GENDIL_HOST_DEVICE ( const Config & kernel ) mutable
      {
         constexpr GlobalIndex SharedSize = shared_size;

         const GlobalIndex candidate = kernel.WorkItemIndex();
         const GlobalIndex linear_thread_index =
            kernel.GetLinearThreadIndex();
         const GlobalIndex thread_slot =
            candidate * threads_per_work_item + linear_thread_index;

         linear_index_data[ thread_slot ] =
            static_cast< long long >( linear_thread_index );
         thread_index_0_data[ thread_slot ] =
            static_cast< long long >(
               kernel.template GetThreadIndex< 0 >() );
         thread_index_1_data[ thread_slot ] =
            static_cast< long long >(
               kernel.template GetThreadIndex< 1 >() );

         if ( linear_thread_index == 0 )
         {
            batch_index_data[ candidate ] =
               static_cast< long long >( kernel.BatchIndex() );
            work_item_index_data[ candidate ] =
               static_cast< long long >( candidate );
            active_flag_data[ candidate ] =
               kernel.IsActive( num_work_items ) ? 1 : 0;

            if ( kernel.IsActive( num_work_items ) )
            {
               guarded_output_data[ candidate ] =
                  static_cast< long long >( 1000 + candidate );
            }
         }

         GENDIL_SHARED Real block_shared[
            Config::SharedMemoryBlockSize( SharedSize ) ];
         KernelContext< Config, SharedSize > context(
            block_shared,
            kernel );
         Real * local_shared =
            context.SharedAllocator.allocate( SharedSize );

         if ( linear_thread_index < SharedSize )
         {
            local_shared[ linear_thread_index ] = SharedValue(
               candidate,
               kernel.BatchIndex(),
               linear_thread_index );
         }

         kernel.Sync();

         if ( linear_thread_index == 0 )
         {
            Real checksum = 0.0;
            long long ok = 1;

            for ( GlobalIndex i = 0; i < SharedSize; ++i )
            {
               const Real expected = SharedValue(
                  candidate,
                  kernel.BatchIndex(),
                  i );
               const Real observed = local_shared[ i ];
               checksum += observed;
               if ( observed != expected )
               {
                  ok = 0;
               }
            }

            shared_ok_data[ candidate ] = ok;
            shared_offset_data[ candidate ] =
               static_cast< long long >( local_shared - block_shared );
            shared_checksum_data[ candidate ] = checksum;
         }

         kernel.Sync();

         if ( linear_thread_index == 0 )
         {
            post_sync_status_data[ candidate ] =
               static_cast< long long >( 2000 + candidate );
         }
      } );

   GENDIL_DEVICE_SYNC;

   batch_index.CopyToHost();
   work_item_index.CopyToHost();
   active_flag.CopyToHost();
   guarded_output.CopyToHost();
   post_sync_status.CopyToHost();
   shared_ok.CopyToHost();
   shared_offset.CopyToHost();
   shared_checksum.CopyToHost();
   linear_index.CopyToHost();
   thread_index_0.CopyToHost();
   thread_index_1.CopyToHost();

   bool success = true;

   for ( GlobalIndex candidate = 0;
         candidate < candidate_count;
         ++candidate )
   {
      const GlobalIndex expected_batch = candidate % Config::batch_size;
      const bool expected_active = candidate < num_work_items;

      success = Check(
         batch_index.data.host_pointer[ candidate ] ==
            static_cast< long long >( expected_batch ),
         "BatchIndex mismatch." ) && success;
      success = Check(
         work_item_index.data.host_pointer[ candidate ] ==
            static_cast< long long >( candidate ),
         "WorkItemIndex mismatch." ) && success;
      success = Check(
         active_flag.data.host_pointer[ candidate ] ==
            ( expected_active ? 1 : 0 ),
         "IsActive mismatch." ) && success;
      success = Check(
         post_sync_status.data.host_pointer[ candidate ] ==
            static_cast< long long >( 2000 + candidate ),
            "A lane did not survive Sync()." ) && success;
      success = Check(
         shared_ok.data.host_pointer[ candidate ] == 1,
         "Batch-local shared-memory contents were not independent." ) &&
         success;
      success = Check(
         shared_offset.data.host_pointer[ candidate ] ==
            static_cast< long long >( expected_batch * shared_size ),
         "Batch-local shared-memory offset mismatch." ) && success;

      Real expected_checksum = 0.0;
      for ( GlobalIndex i = 0; i < shared_size; ++i )
      {
         expected_checksum += SharedValue( candidate, expected_batch, i );
      }
      success = Check(
         Near(
            shared_checksum.data.host_pointer[ candidate ],
            expected_checksum ),
         "Batch-local shared-memory checksum mismatch." ) && success;

      if ( expected_active )
      {
         success = Check(
            guarded_output.data.host_pointer[ candidate ] ==
               static_cast< long long >( 1000 + candidate ),
            "An active lane did not write guarded output." ) && success;
      }
      else
      {
         success = Check(
            guarded_output.data.host_pointer[ candidate ] == sentinel,
            "An inactive lane wrote guarded output." ) && success;
      }

      for ( GlobalIndex linear = 0;
            linear < threads_per_work_item;
            ++linear )
      {
         const GlobalIndex thread_slot =
            candidate * threads_per_work_item + linear;
         success = Check(
            linear_index.data.host_pointer[ thread_slot ] ==
               static_cast< long long >( linear ),
            "GetLinearThreadIndex mismatch." ) && success;
         success = Check(
            thread_index_0.data.host_pointer[ thread_slot ] ==
               static_cast< long long >(
                  Layout::template GetThreadIndex< 0 >( linear ) ),
            "GetThreadIndex<0> mismatch." ) && success;
         success = Check(
            thread_index_1.data.host_pointer[ thread_slot ] ==
               static_cast< long long >(
                  Layout::template GetThreadIndex< 1 >( linear ) ),
            "GetThreadIndex<1> mismatch." ) && success;
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
   static_assert(
      Config::template WorkItemThreadsAlignWithWarpOrWavefront< 32 >() );

   return RunSmokeCase< Config >( "ThreadBlockLayout<4, 8>" );
}

bool TestNonWarpAlignedLayout()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static constexpr Integer BatchSize = 4;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   static_assert( Layout::GetNumberOfThreads() == 15 );
   static_assert(
      !Config::template WorkItemThreadsAlignWithWarpOrWavefront< 32 >() );

   return RunSmokeCase< Config >( "ThreadBlockLayout<3, 5>" );
}
} // namespace

int main()
{
   bool success = true;
   success = TestWarpAlignedLayout() && success;
   success = TestNonWarpAlignedLayout() && success;

   return success ? 0 : 1;
}

#endif
