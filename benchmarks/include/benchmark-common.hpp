// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <gendil/gendil.hpp>

#include <array>
#include <chrono>

namespace gendil::benchmarks
{

#if defined( GENDIL_USE_DEVICE )
constexpr Integer warmup_iterations = 2;
constexpr Integer timed_iterations = 5;
#else
constexpr Integer warmup_iterations = 1;
constexpr Integer timed_iterations = 1;
#endif

#if defined( GENDIL_USE_CUDA )
constexpr size_t static_shared_memory_compile_limit_bytes = 48 * 1024;
constexpr size_t static_local_memory_compile_limit_bytes = 96 * 1024;
constexpr size_t static_threads_per_block_compile_limit = 1024;
#elif defined( GENDIL_USE_HIP )
constexpr size_t static_shared_memory_compile_limit_bytes = 64 * 1024;
constexpr size_t static_local_memory_compile_limit_bytes = 120 * 1024;
constexpr size_t static_threads_per_block_compile_limit = 1024;
#else
constexpr size_t static_shared_memory_compile_limit_bytes = 96 * 1024;
constexpr size_t static_local_memory_compile_limit_bytes = 1024 * 1024;
constexpr size_t static_threads_per_block_compile_limit = 1024 * 1024;
#endif

template < Integer Value, size_t... Is >
auto MakeOrdersImpl( std::index_sequence< Is... > )
{
   return FiniteElementOrders<
      ( static_cast< void >( Is ), Value )... >{};
}

template < Integer Value, size_t... Is >
auto MakeNumQuadsImpl( std::index_sequence< Is... > )
{
   return IntegrationRuleNumPoints<
      ( static_cast< void >( Is ), Value )... >{};
}

template < Integer Dim, Integer Order >
auto MakeOrders()
{
   return MakeOrdersImpl< Order >( std::make_index_sequence< Dim >{} );
}

template < Integer Dim, Integer NumQuad1D >
auto MakeNumQuads()
{
   return MakeNumQuadsImpl< NumQuad1D >(
      std::make_index_sequence< Dim >{} );
}

template < Integer Dim, Integer Order >
constexpr GlobalIndex DofsPerElement()
{
   GlobalIndex value = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      value *= static_cast< GlobalIndex >( Order + 1 );
   }
   return value;
}

template < Integer Dim, Integer NumQuad1D >
constexpr GlobalIndex QuadraturePointsPerElement()
{
   GlobalIndex value = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      value *= static_cast< GlobalIndex >( NumQuad1D );
   }
   return value;
}

template < Integer Dim >
std::array< Integer, Dim > BalancedExtents( GlobalIndex target_cells )
{
   std::array< Integer, Dim > extents{};
   extents.fill( 1 );

   GlobalIndex product = 1;
   Integer axis = 0;
   while ( product < target_cells )
   {
      product /= static_cast< GlobalIndex >( extents[ axis ] );
      extents[ axis ] *= 2;
      product *= static_cast< GlobalIndex >( extents[ axis ] );
      axis = ( axis + 1 ) % Dim;
   }

   return extents;
}

template < Integer Dim >
GlobalIndex Product( const std::array< Integer, Dim > & extents )
{
   GlobalIndex product = 1;
   for ( Integer d = 0; d < Dim; ++d )
   {
      product *= static_cast< GlobalIndex >( extents[ d ] );
   }
   return product;
}

template < Integer Dim >
auto MakeCartesianBenchmarkMesh( const std::array< Integer, Dim > & n )
{
   constexpr Real h = 1.0;
   if constexpr ( Dim == 1 )
   {
      return Cartesian1DMesh( h, n[ 0 ] );
   }
   else if constexpr ( Dim == 2 )
   {
      return Cartesian2DMesh( h, n[ 0 ], n[ 1 ] );
   }
   else if constexpr ( Dim == 3 )
   {
      return Cartesian3DMesh( h, n[ 0 ], n[ 1 ], n[ 2 ] );
   }
   else if constexpr ( Dim == 4 )
   {
      Cartesian2DMesh mesh_1( h, n[ 0 ], n[ 1 ] );
      Cartesian2DMesh mesh_2( h, n[ 2 ], n[ 3 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
   }
   else if constexpr ( Dim == 5 )
   {
      Cartesian3DMesh mesh_1( h, n[ 0 ], n[ 1 ], n[ 2 ] );
      Cartesian2DMesh mesh_2( h, n[ 3 ], n[ 4 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
   }
   else
   {
      static_assert( Dim == 6 );
      Cartesian3DMesh mesh_1( h, n[ 0 ], n[ 1 ], n[ 2 ] );
      Cartesian3DMesh mesh_2( h, n[ 3 ], n[ 4 ], n[ 5 ] );
      return MakeCartesianProductMesh( mesh_1, mesh_2 );
   }
}

template < typename KernelPolicy >
std::array< GlobalIndex, 3 > BlockDimensions( const GlobalIndex num_items )
{
   if constexpr ( requires { KernelPolicy::GetLaunchGeometry( num_items ); } )
   {
      const auto geometry = KernelPolicy::GetLaunchGeometry( num_items );
      return { geometry.block_x, geometry.block_y, geometry.block_z };
   }
   else
   {
      return { 1, 1, 1 };
   }
}

inline bool DeviceThreadLimitsFit(
   const std::array< GlobalIndex, 3 > & block_dims )
{
#if defined( GENDIL_USE_DEVICE )
   int device = 0;
   GENDIL_DEVICE_CHECK( DeviceGetDevice( &device ) );

   DeviceProperties props;
   GENDIL_DEVICE_CHECK( DeviceGetDeviceProperties( &props, device ) );

   const auto threads_per_block =
      block_dims[ 0 ] * block_dims[ 1 ] * block_dims[ 2 ];
   return block_dims[ 0 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 0 ] ) &&
          block_dims[ 1 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 1 ] ) &&
          block_dims[ 2 ] <= static_cast< GlobalIndex >( props.maxThreadsDim[ 2 ] ) &&
          threads_per_block <= static_cast< GlobalIndex >( props.maxThreadsPerBlock );
#else
   (void) block_dims;
   return true;
#endif
}

inline bool DeviceSharedMemoryFits( const size_t shared_memory_per_block )
{
#if defined( GENDIL_USE_DEVICE )
   int device = 0;
   GENDIL_DEVICE_CHECK( DeviceGetDevice( &device ) );

   DeviceProperties props;
   GENDIL_DEVICE_CHECK( DeviceGetDeviceProperties( &props, device ) );

   return shared_memory_per_block * sizeof( Real ) <=
      static_cast< size_t >( props.sharedMemPerBlock );
#else
   (void) shared_memory_per_block;
   return true;
#endif
}

template < typename Operator >
double TimeOperator( Operator & op, Vector & x, Vector & y )
{
   for ( Integer iter = 0; iter < warmup_iterations; ++iter )
   {
      op( x, y );
      op( y, x );
   }
   GENDIL_DEVICE_SYNC;

   const auto start = std::chrono::steady_clock::now();
   for ( Integer iter = 0; iter < timed_iterations; ++iter )
   {
      op( x, y );
      op( y, x );
   }
   GENDIL_DEVICE_SYNC;
   const auto end = std::chrono::steady_clock::now();

   return std::chrono::duration< double >( end - start ).count() /
      static_cast< double >( 2 * timed_iterations );
}

} // namespace gendil::benchmarks
