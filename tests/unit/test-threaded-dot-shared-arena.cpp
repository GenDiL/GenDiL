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
      << "test-threaded-dot-shared-arena skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr GlobalIndex num_work_items = 10;
constexpr Real real_sentinel = -987654321.0;
constexpr long long int_sentinel = -987654321;
constexpr Real tolerance = 1.0e-12;

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
inline Real FirstLeftValue(
   const GlobalIndex candidate,
   const GlobalIndex batch_index,
   const GlobalIndex thread_index )
{
   return Real( 1 ) +
      Real( candidate ) * Real( 0.5 ) +
      Real( batch_index ) * Real( 0.125 ) +
      Real( thread_index ) * Real( 0.25 );
}

GENDIL_HOST_DEVICE
inline Real FirstRightValue(
   const GlobalIndex candidate,
   const GlobalIndex batch_index,
   const GlobalIndex thread_index )
{
   return Real( 2 ) +
      Real( candidate ) * Real( 0.25 ) +
      Real( batch_index ) * Real( 0.5 ) +
      Real( thread_index ) * Real( 0.75 );
}

GENDIL_HOST_DEVICE
inline Real SecondLeftValue(
   const GlobalIndex candidate,
   const GlobalIndex batch_index,
   const GlobalIndex thread_index )
{
   return Real( 3 ) +
      Real( candidate ) * Real( 0.375 ) +
      Real( batch_index ) * Real( 0.25 ) +
      Real( thread_index ) * Real( 0.5 );
}

GENDIL_HOST_DEVICE
inline Real SecondRightValue(
   const GlobalIndex candidate,
   const GlobalIndex batch_index,
   const GlobalIndex thread_index )
{
   return Real( 4 ) +
      Real( candidate ) * Real( 0.125 ) +
      Real( batch_index ) * Real( 0.75 ) +
      Real( thread_index ) * Real( 0.625 );
}

GENDIL_HOST_DEVICE
inline Real GuardValue( const GlobalIndex candidate )
{
   return Real( 10000 ) + Real( candidate );
}

template < typename Config >
Real ExpectedDot(
   const GlobalIndex candidate,
   const bool second )
{
   Real result = 0.0;
   const GlobalIndex batch_index = candidate % Config::batch_size;
   for ( GlobalIndex t = 0;
         t < static_cast< GlobalIndex >( Config::GetNumberOfThreads() );
         ++t )
   {
      if ( second )
      {
         result += SecondLeftValue( candidate, batch_index, t ) *
            SecondRightValue( candidate, batch_index, t );
      }
      else
      {
         result += FirstLeftValue( candidate, batch_index, t ) *
            FirstRightValue( candidate, batch_index, t );
      }
   }
   return result;
}

template < typename Context >
GENDIL_HOST_DEVICE
Real RunDot(
   Context & context,
   const GlobalIndex candidate,
   const GlobalIndex batch_index,
   const bool second )
{
   if constexpr ( Context::thread_block_dim == 0 )
   {
      using Sizes = std::index_sequence< 1 >;
      SerialRecursiveArray< Real, 1 > left_data;
      SerialRecursiveArray< Real, 1 > right_data;
      left_data( 0 ) = second ?
         SecondLeftValue( candidate, batch_index, 0 ) :
         FirstLeftValue( candidate, batch_index, 0 );
      right_data( 0 ) = second ?
         SecondRightValue( candidate, batch_index, 0 ) :
         FirstRightValue( candidate, batch_index, 0 );

      ThreadedView< Sizes, Context, decltype( left_data ) > left{
         left_data };
      ThreadedView< Sizes, Context, decltype( right_data ) > right{
         right_data };
      return Dot( context, left, right );
   }
   else
   {
      using Sizes = std::index_sequence< Context::GetNumberOfThreads() >;
      const GlobalIndex thread_index = context.GetLinearThreadIndex();
      SerialRecursiveArray< Real > left_data;
      SerialRecursiveArray< Real > right_data;
      left_data = second ?
         SecondLeftValue( candidate, batch_index, thread_index ) :
         FirstLeftValue( candidate, batch_index, thread_index );
      right_data = second ?
         SecondRightValue( candidate, batch_index, thread_index ) :
         FirstRightValue( candidate, batch_index, thread_index );

      ThreadedView< Sizes, Context, decltype( left_data ) > left{
         left_data };
      ThreadedView< Sizes, Context, decltype( right_data ) > right{
         right_data };
      return Dot( context, left, right );
   }
}

template < typename Config >
bool RunDotArenaCase( const char * name )
{
   static constexpr Integer RequiredSharedMemory =
      required_threaded_dot_shared_memory_v< Config > + 1;
   using Context = KernelContext< Config, RequiredSharedMemory >;
   static_assert(
      required_threaded_dot_shared_memory_v< Config > == 1,
      "Device ThreadedView Dot needs one arena Real per work item." );
   static_assert(
      Context::per_work_item_shared_memory_size == 2,
      "Dot arena test expects one guard slot plus one Dot scratch slot." );

   const GlobalIndex candidate_count =
      Config::GetLaunchGeometry( num_work_items ).grid_x *
      Config::batch_size;

   DeviceBuffer< Real > first_dot( candidate_count, real_sentinel );
   DeviceBuffer< Real > second_dot( candidate_count, real_sentinel );
   DeviceBuffer< long long > guard_ok( candidate_count, int_sentinel );
   DeviceBuffer< long long > reuse_offset( candidate_count, int_sentinel );
   DeviceBuffer< long long > guard_offset( candidate_count, int_sentinel );

   auto first_dot_data = first_dot.data;
   auto second_dot_data = second_dot.data;
   auto guard_ok_data = guard_ok.data;
   auto reuse_offset_data = reuse_offset.data;
   auto guard_offset_data = guard_offset.data;

   Config::CandidateBlockLoop(
      num_work_items,
      [=] GENDIL_DEVICE () mutable
      {
         GENDIL_SHARED Real block_shared[
            Context::shared_memory_block_size ];
         Context context( block_shared );

         const GlobalIndex candidate = Config::WorkItemIndex();
         const GlobalIndex batch_index = Config::BatchIndex();
         const GlobalIndex linear_thread_index =
            Config::GetLinearThreadIndex();

         Real * guard = context.SharedAllocator.allocate( 1 );
         if ( linear_thread_index == 0 )
         {
            *guard = GuardValue( candidate );
         }
         context.Sync();

         const Real first =
            RunDot( context, candidate, batch_index, false );
         const Real second =
            RunDot( context, candidate, batch_index, true );

         Real * after_dot = context.SharedAllocator.allocate( 1 );
         const bool active = Config::IsActive( num_work_items );
         if ( linear_thread_index == 0 && active )
         {
            first_dot_data[ candidate ] = first;
            second_dot_data[ candidate ] = second;
            guard_ok_data[ candidate ] =
               ( *guard == GuardValue( candidate ) ) ? 1 : 0;
            reuse_offset_data[ candidate ] =
               after_dot == nullptr ?
                  -1 :
                  static_cast< long long >( after_dot - guard );
            guard_offset_data[ candidate ] =
               static_cast< long long >( guard - block_shared );
         }
      } );

   GENDIL_DEVICE_SYNC;

   first_dot.CopyToHost();
   second_dot.CopyToHost();
   guard_ok.CopyToHost();
   reuse_offset.CopyToHost();
   guard_offset.CopyToHost();

   bool success = true;
   for ( GlobalIndex candidate = 0;
         candidate < candidate_count;
         ++candidate )
   {
      const bool active = candidate < num_work_items;
      if ( active )
      {
         success = Check(
            Near(
               first_dot.data.host_pointer[ candidate ],
               ExpectedDot< Config >( candidate, false ) ),
            "First Dot result mismatch." ) && success;
         success = Check(
            Near(
               second_dot.data.host_pointer[ candidate ],
               ExpectedDot< Config >( candidate, true ) ),
            "Second Dot result mismatch." ) && success;
         success = Check(
            guard_ok.data.host_pointer[ candidate ] == 1,
            "Caller-owned guard slot was modified by Dot." ) && success;
         success = Check(
            reuse_offset.data.host_pointer[ candidate ] == 1,
            "Dot did not restore the allocator mark before the next "
            "allocation." ) && success;
         success = Check(
            guard_offset.data.host_pointer[ candidate ] ==
               static_cast< long long >(
                  ( candidate % Config::batch_size ) *
                  Config::SharedMemoryStride( RequiredSharedMemory ) ),
            "KernelContext did not shift the arena by BatchIndex." ) &&
            success;
      }
      else
      {
         success = Check(
            first_dot.data.host_pointer[ candidate ] == real_sentinel,
            "Inactive candidate wrote first Dot output." ) && success;
         success = Check(
            second_dot.data.host_pointer[ candidate ] == real_sentinel,
            "Inactive candidate wrote second Dot output." ) && success;
         success = Check(
            guard_ok.data.host_pointer[ candidate ] == int_sentinel,
            "Inactive candidate wrote guard status." ) && success;
         success = Check(
            reuse_offset.data.host_pointer[ candidate ] == int_sentinel,
            "Inactive candidate wrote allocator offset status." ) &&
            success;
         success = Check(
            guard_offset.data.host_pointer[ candidate ] == int_sentinel,
            "Inactive candidate wrote guard offset." ) && success;
      }
   }

   if ( !success )
   {
      std::cout << "Failure in " << name << ".\n";
   }
   return success;
}

bool TestThreadedLayout()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = 4;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   return RunDotArenaCase< Config >(
      "ThreadedView Dot arena ThreadBlockLayout<4>, BatchSize=4" );
}

bool TestRegisterOnlyWarpBatch()
{
   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = device_warp_size;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   return RunDotArenaCase< Config >(
      "ThreadedView Dot arena ThreadBlockLayout<>, BatchSize=device_warp_size" );
}
} // namespace

int main()
{
   bool success = true;
   success = TestThreadedLayout() && success;
   success = TestRegisterOnlyWarpBatch() && success;
   return success ? 0 : 1;
}

#endif
