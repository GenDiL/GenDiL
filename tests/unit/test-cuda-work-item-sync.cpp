// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>
#include <string>
#include <type_traits>

#if !defined( GENDIL_USE_CUDA )

int main()
{
   std::cout
      << "test-cuda-work-item-sync skipped because CUDA is not enabled.\n";
   return 0;
}

#else

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#ifndef GENDIL_CUDA_STATIC_TILE_1_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_1_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_2_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_2_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_3_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_3_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_4_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_4_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_5_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_5_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_7_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_7_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_8_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_8_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_9_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_9_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_16_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_16_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_25_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_25_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_27_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_27_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_32_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_32_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_49_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_49_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_64_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_64_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_81_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_81_SUPPORTED 0
#endif
#ifndef GENDIL_CUDA_STATIC_TILE_125_SUPPORTED
#define GENDIL_CUDA_STATIC_TILE_125_SUPPORTED 0
#endif

using namespace gendil;

namespace
{
constexpr int batch_size = 4;
constexpr int num_work_items = 2 * batch_size - 1;
constexpr int sentinel = -987654321;

template < int TileSize >
struct static_tile_supported : std::false_type
{};

#define GENDIL_DEFINE_STATIC_TILE_SUPPORT( size )                              \
   template <>                                                                \
   struct static_tile_supported< size >                                        \
      : std::bool_constant<                                                    \
           GENDIL_CUDA_STATIC_TILE_##size##_SUPPORTED != 0 >                  \
   {};

GENDIL_DEFINE_STATIC_TILE_SUPPORT( 1 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 2 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 3 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 4 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 5 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 7 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 8 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 9 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 16 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 25 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 27 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 32 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 49 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 64 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 81 )
GENDIL_DEFINE_STATIC_TILE_SUPPORT( 125 )

#undef GENDIL_DEFINE_STATIC_TILE_SUPPORT

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

template < int TileSize, int BatchSize >
__global__
void WorkItemSyncKernel(
   int * mapping_ok,
   int * shared_ok,
   int * divergent_branch_ok,
   int * partial_batch_ok )
{
   namespace cg = cooperative_groups;

   auto block = cg::this_thread_block();
   auto tile = cg::tiled_partition< TileSize >( block );

   static_assert(
      BatchSize == batch_size,
      "The diagnostic host checker assumes the fixed test batch size." );

   __shared__ int observed_y[ BatchSize * TileSize ];
   __shared__ int observed_rank[ BatchSize * TileSize ];
   __shared__ int shared_payload[ BatchSize * TileSize ];
   __shared__ int branch_payload[ BatchSize * TileSize ];

   const int thread_x = static_cast< int >( threadIdx.x );
   const int batch_index = static_cast< int >( threadIdx.y );
   const int candidate = static_cast< int >( blockIdx.x ) * BatchSize +
      batch_index;
   const int local_slot = batch_index * TileSize + thread_x;

   observed_y[ local_slot ] = batch_index;
   observed_rank[ local_slot ] = static_cast< int >( tile.thread_rank() );
   shared_payload[ local_slot ] = sentinel;
   branch_payload[ local_slot ] = sentinel;

   tile.sync();

   bool local_mapping_ok =
      tile.size() == TileSize &&
      static_cast< int >( tile.thread_rank() ) == thread_x;
   for ( int i = 0; i < TileSize; ++i )
   {
      const int slot = batch_index * TileSize + i;
      local_mapping_ok =
         local_mapping_ok &&
         observed_y[ slot ] == batch_index &&
         observed_rank[ slot ] == i;
   }

   const bool active = candidate < num_work_items;

   if ( active )
   {
      shared_payload[ local_slot ] =
         10000 + 100 * candidate + thread_x;

      tile.sync();

      bool local_shared_ok = true;
      for ( int i = 0; i < TileSize; ++i )
      {
         const int expected = 10000 + 100 * candidate + i;
         local_shared_ok =
            local_shared_ok &&
            shared_payload[ batch_index * TileSize + i ] == expected;
      }

      bool local_branch_ok = true;
      if ( batch_index % 2 == 0 )
      {
         branch_payload[ local_slot ] =
            20000 + 100 * candidate + thread_x;

         tile.sync();

         for ( int i = 0; i < TileSize; ++i )
         {
            const int expected = 20000 + 100 * candidate + i;
            local_branch_ok =
               local_branch_ok &&
               branch_payload[ batch_index * TileSize + i ] == expected;
         }
      }

      if ( thread_x == 0 )
      {
         mapping_ok[ candidate ] = local_mapping_ok ? 1 : 0;
         shared_ok[ candidate ] = local_shared_ok ? 1 : 0;
         divergent_branch_ok[ candidate ] = local_branch_ok ? 1 : 0;
         partial_batch_ok[ candidate ] = 1;
      }
   }
   else if ( thread_x == 0 )
   {
      mapping_ok[ candidate ] = local_mapping_ok ? 1 : 0;
      shared_ok[ candidate ] = 1;
      divergent_branch_ok[ candidate ] = 1;
      partial_batch_ok[ candidate ] = 1;
   }
}

template < int TileSize >
bool LaunchConfigurationFits()
{
   int device = 0;
   cudaError_t error = cudaGetDevice( &device );
   if ( error != cudaSuccess )
   {
      return false;
   }

   cudaDeviceProp properties;
   error = cudaGetDeviceProperties( &properties, device );
   if ( error != cudaSuccess )
   {
      return false;
   }

   const int total_threads = TileSize * batch_size;
   return
      TileSize <= properties.maxThreadsDim[ 0 ] &&
      batch_size <= properties.maxThreadsDim[ 1 ] &&
      total_threads <= properties.maxThreadsPerBlock;
}

void PrintRow(
   const int tile_size,
   const char * status,
   const int mapping_ok,
   const int shared_ok,
   const int divergent_branch_ok,
   const int partial_batch_ok )
{
   std::cout
      << tile_size << ','
      << batch_size << ','
      << tile_size * batch_size << ','
      << status << ','
      << mapping_ok << ','
      << shared_ok << ','
      << divergent_branch_ok << ','
      << partial_batch_ok << '\n';
}

template < int TileSize >
bool RunSupportedStaticTileCase()
{
   if ( !LaunchConfigurationFits< TileSize >() )
   {
      PrintRow(
         TileSize,
         "skipped-launch-limit",
         -1,
         -1,
         -1,
         -1 );
      return true;
   }

   const int grid_size = ( num_work_items + batch_size - 1 ) / batch_size;
   const int candidate_count = grid_size * batch_size;

   DeviceBuffer< int > mapping_ok( candidate_count, sentinel );
   DeviceBuffer< int > shared_ok( candidate_count, sentinel );
   DeviceBuffer< int > divergent_branch_ok( candidate_count, sentinel );
   DeviceBuffer< int > partial_batch_ok( candidate_count, sentinel );

   dim3 grid_dim( grid_size );
   dim3 block_dim( TileSize, batch_size, 1 );

   WorkItemSyncKernel< TileSize, batch_size ><<< grid_dim, block_dim >>>(
      mapping_ok.data,
      shared_ok.data,
      divergent_branch_ok.data,
      partial_batch_ok.data );

   cudaError_t error = cudaGetLastError();
   if ( error != cudaSuccess )
   {
      std::cout
         << "detail,size=" << TileSize
         << ",launch_error=" << cudaGetErrorString( error ) << '\n';
      PrintRow( TileSize, "fail", 0, 0, 0, 0 );
      return false;
   }

   error = cudaDeviceSynchronize();
   if ( error != cudaSuccess )
   {
      std::cout
         << "detail,size=" << TileSize
         << ",sync_error=" << cudaGetErrorString( error ) << '\n';
      PrintRow( TileSize, "fail", 0, 0, 0, 0 );
      return false;
   }

   mapping_ok.CopyToHost();
   shared_ok.CopyToHost();
   divergent_branch_ok.CopyToHost();
   partial_batch_ok.CopyToHost();

   bool all_mapping_ok = true;
   bool all_shared_ok = true;
   bool all_divergent_branch_ok = true;
   bool all_partial_batch_ok = true;

   for ( int candidate = 0; candidate < candidate_count; ++candidate )
   {
      all_mapping_ok =
         all_mapping_ok &&
         mapping_ok.data.host_pointer[ candidate ] == 1;
      all_shared_ok =
         all_shared_ok &&
         shared_ok.data.host_pointer[ candidate ] == 1;
      all_divergent_branch_ok =
         all_divergent_branch_ok &&
         divergent_branch_ok.data.host_pointer[ candidate ] == 1;
      all_partial_batch_ok =
         all_partial_batch_ok &&
         partial_batch_ok.data.host_pointer[ candidate ] == 1;
   }

   const bool success =
      all_mapping_ok &&
      all_shared_ok &&
      all_divergent_branch_ok &&
      all_partial_batch_ok;

   PrintRow(
      TileSize,
      success ? "pass" : "fail",
      all_mapping_ok ? 1 : 0,
      all_shared_ok ? 1 : 0,
      all_divergent_branch_ok ? 1 : 0,
      all_partial_batch_ok ? 1 : 0 );

   return success;
}

template < int TileSize >
bool RunStaticTileCase()
{
   if constexpr ( static_tile_supported< TileSize >::value )
   {
      return RunSupportedStaticTileCase< TileSize >();
   }
   else
   {
      PrintRow(
         TileSize,
         "skipped-unsupported-tile-size",
         -1,
         -1,
         -1,
         -1 );
      return true;
   }
}

void AppendSize(
   std::string & list,
   const int size )
{
   if ( !list.empty() )
   {
      list += ';';
   }
   list += std::to_string( size );
}

template < int TileSize >
void AppendSupportedOrRejected(
   std::string & supported,
   std::string & rejected )
{
   if constexpr ( static_tile_supported< TileSize >::value )
   {
      AppendSize( supported, TileSize );
   }
   else
   {
      AppendSize( rejected, TileSize );
   }
}

void PrintSupportSummary()
{
   std::string supported;
   std::string rejected;

   AppendSupportedOrRejected< 1 >( supported, rejected );
   AppendSupportedOrRejected< 2 >( supported, rejected );
   AppendSupportedOrRejected< 3 >( supported, rejected );
   AppendSupportedOrRejected< 4 >( supported, rejected );
   AppendSupportedOrRejected< 5 >( supported, rejected );
   AppendSupportedOrRejected< 7 >( supported, rejected );
   AppendSupportedOrRejected< 8 >( supported, rejected );
   AppendSupportedOrRejected< 9 >( supported, rejected );
   AppendSupportedOrRejected< 16 >( supported, rejected );
   AppendSupportedOrRejected< 25 >( supported, rejected );
   AppendSupportedOrRejected< 27 >( supported, rejected );
   AppendSupportedOrRejected< 32 >( supported, rejected );
   AppendSupportedOrRejected< 49 >( supported, rejected );
   AppendSupportedOrRejected< 64 >( supported, rejected );
   AppendSupportedOrRejected< 81 >( supported, rejected );
   AppendSupportedOrRejected< 125 >( supported, rejected );

   std::cout
      << "summary,static_tiled_partition_supported_sizes,"
      << supported << '\n'
      << "summary,static_tiled_partition_rejected_sizes,"
      << rejected << '\n';
}

} // namespace

int main()
{
   std::cout
      << "size,batch_size,total_threads,status,mapping_ok,shared_ok,"
      << "divergent_branch_ok,partial_batch_ok\n";

   bool success = true;
   success = RunStaticTileCase< 1 >() && success;
   success = RunStaticTileCase< 2 >() && success;
   success = RunStaticTileCase< 3 >() && success;
   success = RunStaticTileCase< 4 >() && success;
   success = RunStaticTileCase< 5 >() && success;
   success = RunStaticTileCase< 7 >() && success;
   success = RunStaticTileCase< 8 >() && success;
   success = RunStaticTileCase< 9 >() && success;
   success = RunStaticTileCase< 16 >() && success;
   success = RunStaticTileCase< 25 >() && success;
   success = RunStaticTileCase< 27 >() && success;
   success = RunStaticTileCase< 32 >() && success;
   success = RunStaticTileCase< 49 >() && success;
   success = RunStaticTileCase< 64 >() && success;
   success = RunStaticTileCase< 81 >() && success;
   success = RunStaticTileCase< 125 >() && success;

   PrintSupportSummary();

   return success ? 0 : 1;
}

#endif
