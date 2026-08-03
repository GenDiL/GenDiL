// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/synchostdevicearray.hpp"

#include <utility>

namespace gendil
{

/**
 * Move-only owner of a raw duplicated, unsorted COO triplet stream.
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

   SyncHostDeviceArray< IndexType, IndexType > rows{};
   SyncHostDeviceArray< IndexType, IndexType > cols{};
   SyncHostDeviceArray< ValueType, IndexType > values{};

   RawCOOTripletBuffer() = default;
   RawCOOTripletBuffer( const RawCOOTripletBuffer & ) = delete;
   RawCOOTripletBuffer & operator=( const RawCOOTripletBuffer & ) = delete;

   RawCOOTripletBuffer( RawCOOTripletBuffer && other ) noexcept
   : num_rows( std::exchange( other.num_rows, IndexType( 0 ) ) ),
     num_cols( std::exchange( other.num_cols, IndexType( 0 ) ) ),
     nnz_raw( std::exchange( other.nnz_raw, IndexType( 0 ) ) ),
     rows( std::move( other.rows ) ),
     cols( std::move( other.cols ) ),
     values( std::move( other.values ) )
   { }

   RawCOOTripletBuffer & operator=( RawCOOTripletBuffer && other ) noexcept
   {
      if ( this != &other )
      {
         num_rows = std::exchange( other.num_rows, IndexType( 0 ) );
         num_cols = std::exchange( other.num_cols, IndexType( 0 ) );
         nnz_raw = std::exchange( other.nnz_raw, IndexType( 0 ) );
         rows = std::move( other.rows );
         cols = std::move( other.cols );
         values = std::move( other.values );
      }
      return *this;
   }

   ~RawCOOTripletBuffer() = default;
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
   buffer.rows = MakeSyncHostDeviceArray< IndexType >( nnz_raw );
   buffer.cols = MakeSyncHostDeviceArray< IndexType >( nnz_raw );
   buffer.values = MakeSyncHostDeviceArray< ValueType >( nnz_raw );

   auto * rows = WriteHost( buffer.rows );
   auto * cols = WriteHost( buffer.cols );
   auto * values = WriteHost( buffer.values );
   for ( IndexType i = 0; i < nnz_raw; ++i )
   {
      rows[i] = IndexType( 0 );
      cols[i] = IndexType( 0 );
      values[i] = ValueType( 0 );
   }

   Sync( buffer.rows );
   Sync( buffer.cols );
   Sync( buffer.values );
   return buffer;
}

} // namespace gendil
