// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

#include <type_traits>

namespace gendil {

/**
 * Raw duplicated unsorted COO triplet stream.
 *
 * Entries are additive contributions in assembly order. Duplicate `(row, col)`
 * pairs are allowed and expected, and no sorting or duplicate reduction is
 * performed by this buffer.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct RawCOOTripletBuffer
{
   using value_type = ValueType;
   using index_type = IndexType;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz_raw = 0;

   HostDevicePointer< IndexType > rows;
   HostDevicePointer< IndexType > cols;
   HostDevicePointer< ValueType > values;
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
auto MakeRawCOOTripletBuffer(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz_raw )
{
   RawCOOTripletBuffer< ValueType, IndexType > buffer{};
   buffer.num_rows = num_rows;
   buffer.num_cols = num_cols;
   buffer.nnz_raw = nnz_raw;

   AllocateHostPointer( nnz_raw, buffer.rows );
   AllocateDevicePointer( nnz_raw, buffer.rows );
   AllocateHostPointer( nnz_raw, buffer.cols );
   AllocateDevicePointer( nnz_raw, buffer.cols );
   AllocateHostPointer( nnz_raw, buffer.values );
   AllocateDevicePointer( nnz_raw, buffer.values );

   for ( IndexType i = 0; i < nnz_raw; ++i )
   {
      buffer.rows[i] = IndexType( 0 );
      buffer.cols[i] = IndexType( 0 );
      buffer.values[i] = ValueType( 0 );
   }

   ToDevice( static_cast< GlobalIndex >( nnz_raw ), buffer.rows );
   ToDevice( static_cast< GlobalIndex >( nnz_raw ), buffer.cols );
   ToDevice( static_cast< GlobalIndex >( nnz_raw ), buffer.values );

   return buffer;
}

template < typename KernelPolicy, typename Buffer >
void SyncRawCOOTripletBuffer( Buffer & buffer )
{
#if defined(GENDIL_USE_DEVICE)
   using Policy = std::remove_cvref_t< KernelPolicy >;

   if constexpr ( Policy::is_host_configuration )
   {
      ToDevice( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.rows );
      ToDevice( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.cols );
      ToDevice( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.values );
   }
   else
   {
      GENDIL_DEVICE_SYNC;
      ToHost( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.rows );
      ToHost( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.cols );
      ToHost( static_cast< GlobalIndex >( buffer.nnz_raw ), buffer.values );
   }
#else
   (void) buffer;
#endif
}

template < typename Buffer >
void FreeRawCOOTripletBuffer( Buffer & buffer )
{
   FreeHostPointer( buffer.rows );
   FreeDevicePointer( buffer.rows );
   FreeHostPointer( buffer.cols );
   FreeDevicePointer( buffer.cols );
   FreeHostPointer( buffer.values );
   FreeDevicePointer( buffer.values );
}

} // namespace gendil
