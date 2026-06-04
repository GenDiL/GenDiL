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
template < typename BaseKernelPolicy, typename FaceReadPolicy >
struct KernelPolicyWithFaceReadPolicy : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
};

template <
   typename BaseKernelPolicy,
   typename FaceReadPolicy,
   typename FaceWritePolicy >
struct KernelPolicyWithFacePolicies : public BaseKernelPolicy
{
   using face_read_dofs_policy = FaceReadPolicy;
   using face_write_dofs_policy = FaceWritePolicy;
};

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

   using Shape5 = std::index_sequence< 5 >;
   using Shape57 = std::index_sequence< 5, 7 >;
   static_assert(
      threaded_shape_covered_v< HostKernelConfiguration, Shape57 > );
   static_assert(
      threaded_shape_covered_v<
         DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 1 >,
         Shape57 > );
   static_assert(
      !threaded_shape_covered_v<
         DeviceKernelConfiguration< ThreadBlockLayout< 3, 5 >, 2, 1 >,
         Shape5 > );
   static_assert(
      threaded_shape_covered_v<
         DeviceKernelConfiguration< ThreadBlockLayout< 5, 7 >, 2, 1 >,
         Shape57 > );
   static_assert(
      threaded_shape_covered_v<
         DeviceKernelConfiguration< ThreadBlockLayout< 5, 7, 2 >, 3, 1 >,
         Shape57 > );
   static_assert(
      !threaded_shape_covered_v<
         DeviceKernelConfiguration< ThreadBlockLayout< 5, 3 >, 2, 1 >,
         Shape57 > );

   static_assert( is_host_configuration_v< SerialKernelConfiguration > );
   static_assert( !is_device_configuration_v< SerialKernelConfiguration > );
   static_assert( !is_threaded_v< SerialKernelConfiguration > );
   static_assert( is_host_configuration_v< HostKernelConfiguration > );
   static_assert( is_host_configuration_v< const HostKernelConfiguration & > );
   static_assert( !is_device_configuration_v< HostKernelConfiguration > );
   static_assert( !is_device_configuration_v< const HostKernelConfiguration & > );
   static_assert( !is_threaded_v< HostKernelConfiguration > );
   static_assert( !is_threaded_v< const HostKernelConfiguration & > );
   static_assert(
      !is_batched_device_configuration_v< SerialKernelConfiguration > );
   static_assert( SerialKernelConfiguration::batch_size == 1 );
   static_assert( HostKernelConfiguration::batch_size == 1 );
   static_assert( HostKernelConfiguration::GetNumberOfThreads() == 1 );
   static_assert(
      !is_batched_device_configuration_v< HostKernelConfiguration > );
   static_assert(
      is_unbatched_operator_configuration_allowed_v<
         HostKernelConfiguration > );
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( HostKernelConfiguration );

   using HostContext = KernelContext< HostKernelConfiguration, 0 >;
   static_assert( is_host_configuration_v< HostContext > );
   static_assert( is_host_configuration_v< const HostContext & > );
   static_assert( !is_device_configuration_v< HostContext > );
   static_assert( !is_device_configuration_v< const HostContext & > );
   static_assert( !is_threaded_v< HostContext > );
   static_assert( !is_threaded_v< const HostContext & > );

   std::array< int, 8 > host_visits{};
   HostKernelConfiguration::BlockLoop(
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
   static_assert( !is_batched_device_configuration_v< LegacyConfig > );
   static_assert(
      is_unbatched_operator_configuration_allowed_v< LegacyConfig > );
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( LegacyConfig );
   static_assert( is_threaded_v< LegacyConfig > );
   static_assert( LegacyConfig::batch_size == 1 );
   static_assert( LegacyConfig::GetNumberOfThreads() == 6 );
   static_assert( LegacyConfig::SharedMemoryStride( 7 ) == 7 );
   static_assert( LegacyConfig::SharedMemoryBlockSize( 7 ) == 7 );

   constexpr auto legacy_geometry = LegacyConfig::GetLaunchGeometry( 11 );
   static_assert( legacy_geometry.grid_x == 11 );
   static_assert( legacy_geometry.block_x == 2 );
   static_assert( legacy_geometry.block_y == 3 );
   static_assert( legacy_geometry.block_z == 1 );

   if ( !Check(
           LegacyConfig::WorkItemIndex() == 0 &&
           LegacyConfig::BatchIndex() == 0 &&
           LegacyConfig::IsActive( 1 ) &&
           !LegacyConfig::IsActive( 0 ),
           "ThreadFirstKernelConfiguration static host fallback failed." ) )
   {
      return 1;
   }
   if ( !Check(
           LegacyConfig::template GetThreadIndex< 0 >() == 0 &&
           LegacyConfig::template GetThreadIndex< 1 >() == 0,
           "ThreadFirstKernelConfiguration static thread-index fallback failed." ) )
   {
      return 1;
   }

   using BatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout< 2, 3 >, 2, 4 >;
   static_assert( is_device_configuration_v< BatchedConfig > );
   static_assert( !is_host_configuration_v< BatchedConfig > );
   static_assert( is_batched_device_configuration_v< BatchedConfig > );
   static_assert(
      !is_unbatched_operator_configuration_allowed_v< BatchedConfig > );
   static_assert( is_threaded_v< BatchedConfig > );
   static_assert(
      is_batched_device_configuration_v<
         const KernelContext< BatchedConfig, 7 > & > );
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

   if ( !Check(
           BatchedConfig::WorkItemIndex() == 0 &&
           BatchedConfig::BatchIndex() == 0 &&
           BatchedConfig::IsActive( 1 ) &&
           !BatchedConfig::IsActive( 0 ),
           "DeviceKernelConfiguration static host fallback failed." ) )
   {
      return 1;
   }
   if ( !Check(
           BatchedConfig::template GetThreadIndex< 0 >() == 0 &&
           BatchedConfig::template GetThreadIndex< 1 >() == 0,
           "DeviceKernelConfiguration static thread-index fallback failed." ) )
   {
      return 1;
   }

   if ( !Check(
           BatchedConfig::GetLinearThreadIndex() == 0,
           "DeviceKernelConfiguration static linear thread fallback failed." ) )
   {
      return 1;
   }

   using RegisterOnlyBatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 4 >;
   using RegisterOnlyBatchedContext =
      KernelContext< RegisterOnlyBatchedConfig, 0 >;
   static_assert( is_device_configuration_v< RegisterOnlyBatchedConfig > );
   static_assert( !is_host_configuration_v< RegisterOnlyBatchedConfig > );
   static_assert(
      is_batched_device_configuration_v< RegisterOnlyBatchedConfig > );
   static_assert(
      !is_unbatched_operator_configuration_allowed_v<
         RegisterOnlyBatchedConfig > );
   static_assert( !is_threaded_v< RegisterOnlyBatchedConfig > );
   static_assert( !is_threaded_v< RegisterOnlyBatchedContext > );
   static_assert( !is_threaded_v< const RegisterOnlyBatchedContext & > );

   using OneDimThreadedBatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout< 4 >, 1, 4 >;
   using OneDimThreadedBatchedContext =
      KernelContext< OneDimThreadedBatchedConfig, 0 >;
   static_assert( is_device_configuration_v< OneDimThreadedBatchedConfig > );
   static_assert( !is_host_configuration_v< OneDimThreadedBatchedConfig > );
   static_assert(
      is_batched_device_configuration_v< OneDimThreadedBatchedConfig > );
   static_assert( is_threaded_v< OneDimThreadedBatchedConfig > );
   static_assert( is_threaded_v< OneDimThreadedBatchedContext > );
   static_assert( is_threaded_v< const OneDimThreadedBatchedContext & > );

   using UnitExtentThreadedBatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout< 1 >, 1, 4 >;
   static_assert( is_threaded_v< UnitExtentThreadedBatchedConfig > );

   using TwoDimThreadedBatchedConfig =
      DeviceKernelConfiguration< ThreadBlockLayout< 4, 8 >, 2, 4 >;
   static_assert( is_threaded_v< TwoDimThreadedBatchedConfig > );

   using FaceSharedMemoryCheckSpace =
      FiniteElementSpace<
         Cartesian1DMesh,
         GLLFiniteElement< 3 >,
         L2Restriction >;
   using DirectGlobalEmptyLayout =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 2 >;
   using FullSharedEmptyLayout =
      KernelPolicyWithFaceReadPolicy<
         DirectGlobalEmptyLayout,
         FullSharedFaceReadDofsPolicy >;
   using FullSharedThreadedLayout =
      KernelPolicyWithFaceReadPolicy<
         DeviceKernelConfiguration< ThreadBlockLayout< 4 >, 1, 2 >,
         FullSharedFaceReadDofsPolicy >;
   static_assert(
      face_speed_of_light_required_shared_memory_v<
         FaceSoLType::ReadCell,
         DirectGlobalEmptyLayout,
         FaceSharedMemoryCheckSpace > == 0 );
   static_assert(
      face_speed_of_light_required_shared_memory_v<
         FaceSoLType::ReadCell,
         FullSharedEmptyLayout,
         FaceSharedMemoryCheckSpace > == 0 );
   static_assert(
      face_speed_of_light_required_shared_memory_v<
         FaceSoLType::ReadCell,
         FullSharedThreadedLayout,
         FaceSharedMemoryCheckSpace > ==
      FaceSharedMemoryCheckSpace::finite_element_type::GetNumDofs() );

   using FullSharedFaceEmptyLayout =
      KernelPolicyWithFacePolicies<
         DirectGlobalEmptyLayout,
         FullSharedFaceReadDofsPolicy,
         FullSharedFaceWriteDofsPolicy >;
   using DirectGlobalFaceThreadedLayout =
      DeviceKernelConfiguration< ThreadBlockLayout< 4 >, 1, 2 >;
   using FullSharedFaceThreadedLayout =
      KernelPolicyWithFacePolicies<
         DirectGlobalFaceThreadedLayout,
         FullSharedFaceReadDofsPolicy,
         FullSharedFaceWriteDofsPolicy >;
   static_assert(
      global_face_speed_of_light_required_shared_memory_v<
         DirectGlobalEmptyLayout,
         FaceSharedMemoryCheckSpace > == 0 );
   static_assert(
      global_face_speed_of_light_required_shared_memory_v<
         FullSharedFaceEmptyLayout,
         FaceSharedMemoryCheckSpace > == 0 );
   static_assert(
      global_face_speed_of_light_required_shared_memory_v<
         DirectGlobalFaceThreadedLayout,
         FaceSharedMemoryCheckSpace > == 0 );
   static_assert(
      global_face_speed_of_light_required_shared_memory_v<
         FullSharedFaceThreadedLayout,
         FaceSharedMemoryCheckSpace > ==
      FaceSharedMemoryCheckSpace::finite_element_type::GetNumDofs() );

   using FaceAdvectionRule =
      decltype( MakeIntegrationRule( IntegrationRuleNumPoints< 4 >{} ) );
   static_assert(
      global_face_advection_required_shared_memory_v<
         DirectGlobalEmptyLayout,
         FaceAdvectionRule,
         FaceSharedMemoryCheckSpace > ==
      required_shared_memory_v< DirectGlobalEmptyLayout, FaceAdvectionRule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         FullSharedFaceEmptyLayout,
         FaceAdvectionRule,
         FaceSharedMemoryCheckSpace > ==
      required_shared_memory_v< FullSharedFaceEmptyLayout, FaceAdvectionRule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         DirectGlobalFaceThreadedLayout,
         FaceAdvectionRule,
         FaceSharedMemoryCheckSpace > ==
      required_shared_memory_v<
         DirectGlobalFaceThreadedLayout,
         FaceAdvectionRule > );
   static_assert(
      global_face_advection_required_shared_memory_v<
         FullSharedFaceThreadedLayout,
         FaceAdvectionRule,
         FaceSharedMemoryCheckSpace > ==
      Max(
         FaceSharedMemoryCheckSpace::finite_element_type::GetNumDofs(),
         required_shared_memory_v<
            FullSharedFaceThreadedLayout,
            FaceAdvectionRule > ) );

   using BatchedSingle =
      DeviceKernelConfiguration< ThreadBlockLayout< 2, 3 >, 2, 1 >;
   static_assert( !is_batched_device_configuration_v< BatchedSingle > );
   static_assert(
      is_unbatched_operator_configuration_allowed_v< BatchedSingle > );
   GENDIL_REQUIRE_UNBATCHED_OPERATOR( BatchedSingle );
   constexpr auto batched_single_geometry =
      BatchedSingle::GetLaunchGeometry( 11 );
   static_assert( batched_single_geometry.grid_x == 11 );
   static_assert( batched_single_geometry.block_x == 6 );
   static_assert( batched_single_geometry.block_y == 1 );
   if ( !Check(
           BatchedSingle::WorkItemIndex() == LegacyConfig::WorkItemIndex() &&
           BatchedSingle::BatchIndex() == LegacyConfig::BatchIndex() &&
           BatchedSingle::IsActive( 1 ) == LegacyConfig::IsActive( 1 ),
           "BatchSize == 1 static fallback equivalence failed." ) )
   {
      return 1;
   }

   Real shared[ BatchedConfig::SharedMemoryBlockSize( 7 ) ];
   Real * batch_slice = BatchedConfig::SharedMemoryForWorkItem( shared, 7 );
   if ( !Check(
           batch_slice == shared,
           "DeviceKernelConfiguration host fallback should use batch lane 0." ) )
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

   KernelContext< BatchedConfig, 7 > batched_context( shared );
   Real * arena_slice = batched_context.SharedAllocator.allocate( 7 );
   if ( !Check(
           arena_slice == batch_slice,
           "KernelContext did not use the static configuration shared slice." ) )
   {
      return 1;
   }

   KernelContext< LegacyConfig, 7 > legacy_context( shared );
   Real * legacy_slice = legacy_context.SharedAllocator.allocate( 7 );
   if ( !Check(
           legacy_slice == shared,
           "KernelContext should keep legacy shared memory as one block arena." ) )
   {
      return 1;
   }

   return 0;
}
