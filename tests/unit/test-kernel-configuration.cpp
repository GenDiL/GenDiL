// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cstdint>
#include <iostream>

using namespace gendil;

namespace
{
bool Check( bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << message << '\n';
   }
   return condition;
}
} // namespace

int main( int, char ** )
{
   using LogicalLayout = ThreadBlockLayout< 2, 3, 5 >;
   static_assert( LogicalLayout::thread_block_dim == 3 );
   static_assert( LogicalLayout::GetNumberOfThreads() == 30 );
   static_assert( LogicalLayout::template GetBlockDim< 0 >() == 2 );
   static_assert( LogicalLayout::template GetBlockDim< 1 >() == 3 );
   static_assert( LogicalLayout::template GetBlockDim< 2 >() == 5 );
   static_assert( LogicalLayout::template IsThreadedDimension< 2 >() );
   static_assert( !LogicalLayout::template IsThreadedDimension< 3 >() );
   static_assert( LogicalLayout::template GetThreadIndex< 0 >( 23 ) == 1 );
   static_assert( LogicalLayout::template GetThreadIndex< 1 >( 23 ) == 2 );
   static_assert( LogicalLayout::template GetThreadIndex< 2 >( 23 ) == 3 );

   static_assert( is_host_configuration_v< SerialKernelConfiguration > );
   static_assert( is_host_configuration_v< HostKernelConfiguration<> > );
   static_assert( is_host_configuration_v< HostKernelConfiguration< 1 > > );
   static_assert( is_host_configuration_v< const HostKernelConfiguration< 1 > & > );
   static_assert( is_serial_v< SerialKernelConfiguration > );
   static_assert( is_serial_v< HostKernelConfiguration< 1 > > );
   static_assert( is_serial_v< const HostKernelConfiguration< 1 > & > );
   static_assert( !is_device_configuration_v< SerialKernelConfiguration > );
   static_assert( SerialKernelConfiguration::batch_size == 1 );
   static_assert( HostKernelConfiguration< 4 >::batch_size == 4 );
   static_assert( HostKernelConfiguration< 4 >::GetNumberOfThreads() == 1 );

   using HostContext = KernelContext< HostKernelConfiguration< 1 >, 0 >;
   static_assert( is_host_configuration_v< HostContext > );
   static_assert( is_host_configuration_v< const HostContext & > );
   static_assert( is_serial_v< HostContext > );
   static_assert( is_serial_v< const HostContext & > );
   static_assert( !is_device_configuration_v< HostContext > );
   static_assert( !is_device_configuration_v< const HostContext & > );

   std::array< int, 8 > host_visits{};
   HostKernelConfiguration< 4 >::BlockLoop(
      host_visits.size(),
      [&] ( GlobalIndex work_item_index )
      {
         host_visits[ work_item_index ] += 1;
      } );

   for ( size_t i = 0; i < host_visits.size(); ++i )
   {
      if ( !Check(
              host_visits[ i ] == 1,
              "HostKernelConfiguration::BlockLoop did not visit each item once." ) )
      {
         return 1;
      }
   }

   using LegacyConfig =
      ThreadFirstKernelConfiguration< ThreadBlockLayout< 2, 3 >, 2 >;
   static_assert( is_device_configuration_v< LegacyConfig > );
   static_assert( !is_host_configuration_v< LegacyConfig > );
   static_assert( !is_serial_v< LegacyConfig > );
   static_assert( LegacyConfig::batch_size == 1 );
   static_assert( LegacyConfig::GetNumberOfThreads() == 6 );
   static_assert( LegacyConfig::SharedMemoryStride( 7 ) == 7 );
   static_assert( LegacyConfig::SharedMemoryBlockSize( 7 ) == 7 );

   constexpr auto legacy_geometry = LegacyConfig::GetLaunchGeometry( 11 );
   static_assert( legacy_geometry.grid_x == 11 );
   static_assert( legacy_geometry.block_x == 2 );
   static_assert( legacy_geometry.block_y == 3 );
   static_assert( legacy_geometry.block_z == 1 );

   LegacyConfig legacy_config( 9, 5 );
   if ( !Check(
           legacy_config.WorkItemIndex() == 9 &&
           legacy_config.BatchIndex() == 0 &&
           legacy_config.IsActive( 10 ) &&
           !legacy_config.IsActive( 9 ),
           "ThreadFirstKernelConfiguration work-item API failed." ) )
   {
      return 1;
   }
   if ( !Check(
           legacy_config.template GetThreadIndex< 0 >() == 1 &&
           legacy_config.template GetThreadIndex< 1 >() == 2,
           "ThreadFirstKernelConfiguration logical thread decoding failed." ) )
   {
      return 1;
   }

   using BatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout< 2, 3 >, 2, 4 >;
   static_assert( is_device_configuration_v< BatchedConfig > );
   static_assert( !is_host_configuration_v< BatchedConfig > );
   static_assert( !is_serial_v< BatchedConfig > );
   static_assert( BatchedConfig::batch_size == 4 );
   static_assert( BatchedConfig::GetNumberOfThreads() == 6 );
   static_assert( BatchedConfig::SharedMemoryStride( 7 ) == 7 );
   static_assert( BatchedConfig::SharedMemoryBlockSize( 7 ) == 28 );
   static_assert(
      BatchedConfig::template WorkItemThreadsAlignWithWarpOrWavefront< 3 >() );
   static_assert(
      !BatchedConfig::template WorkItemThreadsAlignWithWarpOrWavefront< 4 >() );

   constexpr auto batched_geometry = BatchedConfig::GetLaunchGeometry( 10 );
   static_assert( batched_geometry.grid_x == 3 );
   static_assert( batched_geometry.block_x == 6 );
   static_assert( batched_geometry.block_y == 4 );
   static_assert( batched_geometry.block_z == 1 );

   BatchedConfig batched_config( 9, 1, 5 );
   if ( !Check(
           batched_config.WorkItemIndex() == 9 &&
           batched_config.BatchIndex() == 1 &&
           batched_config.IsActive( 10 ) &&
           !batched_config.IsActive( 9 ),
           "DeviceKernelConfiguration work-item API failed." ) )
   {
      return 1;
   }
   if ( !Check(
           batched_config.template GetThreadIndex< 0 >() == 1 &&
           batched_config.template GetThreadIndex< 1 >() == 2,
           "DeviceKernelConfiguration logical thread decoding failed." ) )
   {
      return 1;
   }

   using BatchedSingle =
      DeviceKernelConfiguration< ThreadBlockLayout< 2, 3 >, 2, 1 >;
   constexpr auto batched_single_geometry =
      BatchedSingle::GetLaunchGeometry( 11 );
   static_assert( batched_single_geometry.grid_x == 11 );
   static_assert( batched_single_geometry.block_x == 6 );
   static_assert( batched_single_geometry.block_y == 1 );
   BatchedSingle batched_single( 9, 0, 5 );
   if ( !Check(
           batched_single.WorkItemIndex() == legacy_config.WorkItemIndex() &&
           batched_single.BatchIndex() == legacy_config.BatchIndex() &&
           batched_single.IsActive( 10 ) == legacy_config.IsActive( 10 ),
           "BatchSize == 1 semantic equivalence failed." ) )
   {
      return 1;
   }

   Real shared[ BatchedConfig::SharedMemoryBlockSize( 7 ) ];
   Real * batch_slice = batched_config.SharedMemoryForWorkItem( shared, 7 );
   if ( !Check(
           batch_slice == shared + 7,
           "DeviceKernelConfiguration shared-memory slice offset failed." ) )
   {
      return 1;
   }

   auto address = reinterpret_cast< std::uintptr_t >( batch_slice );
   if ( !Check(
           address % alignof( Real ) == 0,
           "DeviceKernelConfiguration shared-memory slice lost Real alignment." ) )
   {
      return 1;
   }

   KernelContext< BatchedConfig, 7 > batched_context(
      shared,
      batched_config );
   Real * arena_slice = batched_context.SharedAllocator.allocate( 7 );
   if ( !Check(
           arena_slice == batch_slice,
           "KernelContext did not use the device batch-local shared slice." ) )
   {
      return 1;
   }

   KernelContext< LegacyConfig, 7 > legacy_context( shared, legacy_config );
   Real * legacy_slice = legacy_context.SharedAllocator.allocate( 7 );
   if ( !Check(
           legacy_slice == shared,
           "KernelContext should keep legacy shared memory as one block arena." ) )
   {
      return 1;
   }

   return 0;
}
