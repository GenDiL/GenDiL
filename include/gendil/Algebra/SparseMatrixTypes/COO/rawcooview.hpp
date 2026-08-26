// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/rawcootripletbuffer.hpp"

#include <type_traits>

namespace gendil
{

/**
 * Borrowed raw COO triplet view for one selected memory space.
 * Const-qualified template arguments express read-only access. The
 * device-copyable view owns nothing and must not outlive its triplet buffer.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct RawCOOTripletView
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;

   index_type num_rows = 0;
   index_type num_cols = 0;
   index_type nnz_raw = 0;

   IndexType * rows = nullptr;
   IndexType * cols = nullptr;
   ValueType * values = nullptr;
};

/// Return a host read view, synchronizing every triplet array to host as needed.
template < typename ValueType, typename IndexType >
auto GetHostReadView(
   const RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< const ValueType, const IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      ReadHost( buffer.rows ),
      ReadHost( buffer.cols ),
      ReadHost( buffer.values ) };
}

/// Return a host read-write triplet view and invalidate the device copies.
template < typename ValueType, typename IndexType >
auto GetHostReadWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< ValueType, IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      ReadWriteHost( buffer.rows ),
      ReadWriteHost( buffer.cols ),
      ReadWriteHost( buffer.values ) };
}

/// Return a host write triplet view without preserving the previous contents.
template < typename ValueType, typename IndexType >
auto GetHostWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< ValueType, IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      WriteHost( buffer.rows ),
      WriteHost( buffer.cols ),
      WriteHost( buffer.values ) };
}

/// Return a device read view, synchronizing every triplet array as needed.
template < typename ValueType, typename IndexType >
auto GetDeviceReadView(
   const RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< const ValueType, const IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      ReadDevice( buffer.rows ),
      ReadDevice( buffer.cols ),
      ReadDevice( buffer.values ) };
}

/// Return a device read-write triplet view and invalidate the host copies.
template < typename ValueType, typename IndexType >
auto GetDeviceReadWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< ValueType, IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      ReadWriteDevice( buffer.rows ),
      ReadWriteDevice( buffer.cols ),
      ReadWriteDevice( buffer.values ) };
}

/// Return a device write triplet view without preserving previous contents.
template < typename ValueType, typename IndexType >
auto GetDeviceWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   return RawCOOTripletView< ValueType, IndexType >{
      buffer.num_rows,
      buffer.num_cols,
      buffer.nnz_raw,
      WriteDevice( buffer.rows ),
      WriteDevice( buffer.cols ),
      WriteDevice( buffer.values ) };
}

/// Return a triplet read view, synchronizing arrays to the selected space.
template < bool OnDevice, typename ValueType, typename IndexType >
auto GetKernelReadView(
   const RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadView( buffer );
   }
   else
   {
      return GetHostReadView( buffer );
   }
}

/// Return a triplet read-write view and invalidate all opposite-space copies.
template < bool OnDevice, typename ValueType, typename IndexType >
auto GetKernelReadWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadWriteView( buffer );
   }
   else
   {
      return GetHostReadWriteView( buffer );
   }
}

/// Return a triplet write view without syncing and invalidate opposite copies.
template < bool OnDevice, typename ValueType, typename IndexType >
auto GetKernelWriteView(
   RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceWriteView( buffer );
   }
   else
   {
      return GetHostWriteView( buffer );
   }
}

/// Synchronize every initialized RawCOO triplet array between host and device.
template < typename ValueType, typename IndexType >
void Sync( const RawCOOTripletBuffer< ValueType, IndexType > & buffer )
{
   Sync( buffer.rows );
   Sync( buffer.cols );
   Sync( buffer.values );
}

} // namespace gendil
