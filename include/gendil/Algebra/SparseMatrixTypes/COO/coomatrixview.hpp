// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixstorage.hpp"

#include <type_traits>

namespace gendil
{

/**
 * Borrowed COO view for one selected memory space.
 *
 * Const-qualified template arguments express read-only values and indices.
 * The device-copyable view owns nothing and must not outlive its matrix.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct COOMatrixView
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;

   index_type num_rows = 0;
   index_type num_cols = 0;
   index_type nnz = 0;

   IndexType * rows = nullptr;
   IndexType * cols = nullptr;
   ValueType * values = nullptr;
};

/// Return a host read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadView(
   const COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.rows ),
      ReadHost( matrix.cols ),
      ReadHost( matrix.values ) };
}

/// Return a host read-write view, preserving contents and invalidating device
/// copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteHost( matrix.rows ),
      ReadWriteHost( matrix.cols ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host write view without synchronizing old contents, invalidating
/// device copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteHost( matrix.rows ),
      WriteHost( matrix.cols ),
      WriteHost( matrix.values ) };
}

/// Return a device read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadView(
   const COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.rows ),
      ReadDevice( matrix.cols ),
      ReadDevice( matrix.values ) };
}

/// Return a device read-write view, preserving contents and invalidating host
/// copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteDevice( matrix.rows ),
      ReadWriteDevice( matrix.cols ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device write view without synchronizing old contents, invalidating
/// host copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteDevice( matrix.rows ),
      WriteDevice( matrix.cols ),
      WriteDevice( matrix.values ) };
}

/// Return a host values read-write view, synchronizing values and structure,
/// invalidating device values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesReadWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.rows ),
      ReadHost( matrix.cols ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.rows ),
      ReadHost( matrix.cols ),
      WriteHost( matrix.values ) };
}

/// Return a device values read-write view, synchronizing values and structure,
/// invalidating host values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesReadWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.rows ),
      ReadDevice( matrix.cols ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   return COOMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.rows ),
      ReadDevice( matrix.cols ),
      WriteDevice( matrix.values ) };
}

/// Return a read view, synchronizing all arrays to the selected memory space.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   typename Backend >
auto GetKernelReadView(
   const COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadView( matrix );
   }
   else
   {
      return GetHostReadView( matrix );
   }
}

/// Return a read-write view, preserving arrays and invalidating opposite copies.
/// Cached vendor backends remain unchanged.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   typename Backend >
auto GetKernelReadWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadWriteView( matrix );
   }
   else
   {
      return GetHostReadWriteView( matrix );
   }
}

/// Return a write view without syncing and invalidate all opposite copies.
/// Cached vendor backends remain unchanged.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   typename Backend >
auto GetKernelWriteView(
   COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceWriteView( matrix );
   }
   else
   {
      return GetHostWriteView( matrix );
   }
}

/// Synchronize all initialized COO arrays.
template < typename ValueType, typename IndexType, typename Backend >
void Sync( const COOMatrix< ValueType, IndexType, Backend > & matrix )
{
   Sync( matrix.rows );
   Sync( matrix.cols );
   Sync( matrix.values );
}

} // namespace gendil
