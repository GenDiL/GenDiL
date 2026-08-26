// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#if !defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
#error "deviceprimitives.hpp requires GPU sparse finalization support"
#endif

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#if defined(GENDIL_USE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#elif defined(GENDIL_USE_HIP)
#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_reduce_by_key.hpp>
#include <rocprim/device/device_scan.hpp>
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif

#include <climits>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil::details
{

/** Move-only, device-only storage used by vendor primitive adapters. */
template < typename T >
class DeviceOnlyBuffer
{
public:
   DeviceOnlyBuffer() = default;

   explicit DeviceOnlyBuffer( const size_t count )
   : count_( count )
   {
      AllocateDevicePointer( count, storage_ );
   }

   DeviceOnlyBuffer( const DeviceOnlyBuffer & ) = delete;
   DeviceOnlyBuffer & operator=( const DeviceOnlyBuffer & ) = delete;

   DeviceOnlyBuffer( DeviceOnlyBuffer && other ) noexcept
   : storage_( std::move( other.storage_ ) ),
     count_( std::exchange( other.count_, size_t( 0 ) ) )
   { }

   DeviceOnlyBuffer & operator=( DeviceOnlyBuffer && other ) noexcept
   {
      if ( this != &other )
      {
         Release();
         storage_ = std::move( other.storage_ );
         count_ = std::exchange( other.count_, size_t( 0 ) );
      }
      return *this;
   }

   ~DeviceOnlyBuffer()
   {
      Release();
   }

   T * data()
   {
      return storage_.device_pointer;
   }

   const T * data() const
   {
      return storage_.device_pointer;
   }

   size_t size() const
   {
      return count_;
   }

private:
   void Release()
   {
      FreeDevicePointer( storage_ );
      storage_.device_pointer = nullptr;
      count_ = 0;
   }

   HostDevicePointer< T > storage_{};
   size_t count_ = 0;
};

/** Fill a device-memory range with a byte value. */
inline void DeviceMemset( void * destination, const int value, const size_t bytes )
{
   if ( bytes == 0 )
   {
      return;
   }
#if defined(GENDIL_USE_CUDA)
   GENDIL_GPU_CHECK( cudaMemset( destination, value, bytes ) );
#elif defined(GENDIL_USE_HIP)
   GENDIL_GPU_CHECK( hipMemset( destination, value, bytes ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/** Copy a byte range between two device-memory locations. */
inline void DeviceCopyToDevice(
   void * destination,
   const void * source,
   const size_t bytes )
{
   if ( bytes == 0 )
   {
      return;
   }
#if defined(GENDIL_USE_CUDA)
   GENDIL_GPU_CHECK(
      cudaMemcpy( destination, source, bytes, cudaMemcpyDeviceToDevice ) );
#elif defined(GENDIL_USE_HIP)
   GENDIL_GPU_CHECK(
      hipMemcpy( destination, source, bytes, hipMemcpyDeviceToDevice ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/** Copy a byte range from device memory to host memory. */
inline void DeviceCopyToHost(
   void * destination,
   const void * source,
   const size_t bytes )
{
   if ( bytes == 0 )
   {
      return;
   }
#if defined(GENDIL_USE_CUDA)
   GENDIL_GPU_CHECK(
      cudaMemcpy( destination, source, bytes, cudaMemcpyDeviceToHost ) );
#elif defined(GENDIL_USE_HIP)
   GENDIL_GPU_CHECK(
      hipMemcpy( destination, source, bytes, hipMemcpyDeviceToHost ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/** Return whether the vendor primitives can represent a nonnegative count. */
template < typename IndexType >
bool DevicePrimitiveCanProcess( const IndexType count )
{
   if constexpr ( std::is_signed_v< IndexType > )
   {
      if ( count < IndexType( 0 ) )
      {
         return false;
      }
   }

#if defined(GENDIL_USE_CUDA)
   using UnsignedIndex = std::make_unsigned_t< IndexType >;
   return static_cast< UnsignedIndex >( count ) <=
      static_cast< UnsignedIndex >( std::numeric_limits< int >::max() );
#elif defined(GENDIL_USE_HIP)
   using UnsignedIndex = std::make_unsigned_t< IndexType >;
   if constexpr ( sizeof( UnsignedIndex ) > sizeof( size_t ) )
   {
      return static_cast< UnsignedIndex >( count ) <=
         static_cast< UnsignedIndex >( std::numeric_limits< size_t >::max() );
   }
   else
   {
      return true;
   }
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/** Stable-sort device key/value pairs by the complete integral key. */
template < typename Key, typename Value, typename IndexType >
void DeviceStableRadixSortPairs(
   const Key * keys_input,
   Key * keys_output,
   const Value * values_input,
   Value * values_output,
   const IndexType count )
{
   static_assert(
      std::is_integral_v< Key >,
      "Sparse coordinate radix keys must be integral." );
   if ( count == IndexType( 0 ) )
   {
      return;
   }

   size_t temporary_storage_bytes = 0;
#if defined(GENDIL_USE_CUDA)
   const int item_count = static_cast< int >( count );
   GENDIL_GPU_CHECK(
      cub::DeviceRadixSort::SortPairs(
         nullptr,
         temporary_storage_bytes,
         keys_input,
         keys_output,
         values_input,
         values_output,
         item_count,
         0,
         int( sizeof( Key ) * CHAR_BIT ) ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      cub::DeviceRadixSort::SortPairs(
         temporary_storage.data(),
         temporary_storage_bytes,
         keys_input,
         keys_output,
         values_input,
         values_output,
         item_count,
         0,
         int( sizeof( Key ) * CHAR_BIT ) ) );
#elif defined(GENDIL_USE_HIP)
   const size_t item_count = static_cast< size_t >( count );
   GENDIL_GPU_CHECK(
      rocprim::radix_sort_pairs(
         nullptr,
         temporary_storage_bytes,
         keys_input,
         keys_output,
         values_input,
         values_output,
         item_count,
         0,
         unsigned( sizeof( Key ) * CHAR_BIT ) ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      rocprim::radix_sort_pairs(
         temporary_storage.data(),
         temporary_storage_bytes,
         keys_input,
         keys_output,
         values_input,
         values_output,
         item_count,
         0,
         unsigned( sizeof( Key ) * CHAR_BIT ) ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/**
 * Reduce consecutive equivalent device keys and report the unique count.
 * CUB uses `Key::operator==`; rocPRIM uses the supplied equality operation.
 */
template <
   typename Key,
   typename Value,
   typename IndexType,
   typename Reduction,
   typename Equality >
void DeviceReduceByKey(
   const Key * keys_input,
   Key * unique_keys_output,
   const Value * values_input,
   Value * aggregates_output,
   IndexType * unique_count_output,
   const IndexType count,
   Reduction reduction,
   Equality equality )
{
   size_t temporary_storage_bytes = 0;
#if defined(GENDIL_USE_CUDA)
   (void) equality;
   const int item_count = static_cast< int >( count );
   GENDIL_GPU_CHECK(
      cub::DeviceReduce::ReduceByKey(
         nullptr,
         temporary_storage_bytes,
         keys_input,
         unique_keys_output,
         values_input,
         aggregates_output,
         unique_count_output,
         reduction,
         item_count ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      cub::DeviceReduce::ReduceByKey(
         temporary_storage.data(),
         temporary_storage_bytes,
         keys_input,
         unique_keys_output,
         values_input,
         aggregates_output,
         unique_count_output,
         reduction,
         item_count ) );
#elif defined(GENDIL_USE_HIP)
   const size_t item_count = static_cast< size_t >( count );
   GENDIL_GPU_CHECK(
      rocprim::reduce_by_key(
         nullptr,
         temporary_storage_bytes,
         keys_input,
         values_input,
         item_count,
         unique_keys_output,
         aggregates_output,
         unique_count_output,
         reduction,
         equality ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      rocprim::reduce_by_key(
         temporary_storage.data(),
         temporary_storage_bytes,
         keys_input,
         values_input,
         item_count,
         unique_keys_output,
         aggregates_output,
         unique_count_output,
         reduction,
         equality ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

/** Compute an inclusive scan over device input into separate device output. */
template < typename T, typename IndexType, typename BinaryOperation >
void DeviceInclusiveScan(
   const T * input,
   T * output,
   const IndexType count,
   BinaryOperation operation )
{
   if ( count == IndexType( 0 ) )
   {
      return;
   }

   size_t temporary_storage_bytes = 0;
#if defined(GENDIL_USE_CUDA)
   const int item_count = static_cast< int >( count );
   GENDIL_GPU_CHECK(
      cub::DeviceScan::InclusiveScan(
         nullptr,
         temporary_storage_bytes,
         input,
         output,
         operation,
         item_count ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      cub::DeviceScan::InclusiveScan(
         temporary_storage.data(),
         temporary_storage_bytes,
         input,
         output,
         operation,
         item_count ) );
#elif defined(GENDIL_USE_HIP)
   const size_t item_count = static_cast< size_t >( count );
   GENDIL_GPU_CHECK(
      rocprim::inclusive_scan(
         nullptr,
         temporary_storage_bytes,
         input,
         output,
         item_count,
         operation ) );
   DeviceOnlyBuffer< unsigned char > temporary_storage(
      temporary_storage_bytes == 0 ? size_t( 1 ) : temporary_storage_bytes );
   GENDIL_GPU_CHECK(
      rocprim::inclusive_scan(
         temporary_storage.data(),
         temporary_storage_bytes,
         input,
         output,
         item_count,
         operation ) );
#else
#error "GPU sparse finalization requires CUDA or HIP"
#endif
}

} // namespace gendil::details
