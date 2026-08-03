// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixstorage.hpp"

#include <type_traits>

namespace gendil
{

/**
 * Borrowed CSR view for one selected memory space.
 *
 * Const-qualified template arguments express read-only values and indices.
 * The device-copyable view owns nothing and must not outlive its matrix.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct CSRMatrixView
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;

   index_type num_rows = 0;
   index_type num_cols = 0;
   index_type nnz = 0;

   IndexType * row_ptr = nullptr;
   IndexType * col_ind = nullptr;
   ValueType * values = nullptr;
};

/// Return a host read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadView(
   const CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.row_ptr ),
      ReadHost( matrix.col_ind ),
      ReadHost( matrix.values ) };
}

/// Return a host read-write view, preserving contents and invalidating device
/// copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteHost( matrix.row_ptr ),
      ReadWriteHost( matrix.col_ind ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host write view without synchronizing old contents, invalidating
/// device copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteHost( matrix.row_ptr ),
      WriteHost( matrix.col_ind ),
      WriteHost( matrix.values ) };
}

/// Return a device read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadView(
   const CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.row_ptr ),
      ReadDevice( matrix.col_ind ),
      ReadDevice( matrix.values ) };
}

/// Return a device read-write view, preserving contents and invalidating host
/// copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteDevice( matrix.row_ptr ),
      ReadWriteDevice( matrix.col_ind ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device write view without synchronizing old contents, invalidating
/// host copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteDevice( matrix.row_ptr ),
      WriteDevice( matrix.col_ind ),
      WriteDevice( matrix.values ) };
}

/// Return a host values read-write view, synchronizing values and structure,
/// invalidating device values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesReadWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.row_ptr ),
      ReadHost( matrix.col_ind ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.row_ptr ),
      ReadHost( matrix.col_ind ),
      WriteHost( matrix.values ) };
}

/// Return a device values read-write view, synchronizing values and structure,
/// invalidating host values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesReadWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.row_ptr ),
      ReadDevice( matrix.col_ind ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesWriteView(
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSRMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.row_ptr ),
      ReadDevice( matrix.col_ind ),
      WriteDevice( matrix.values ) };
}

/// Return a read view, synchronizing all arrays to the selected memory space.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   typename Backend >
auto GetKernelReadView(
   const CSRMatrix< ValueType, IndexType, Backend > & matrix )
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
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
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
   CSRMatrix< ValueType, IndexType, Backend > & matrix )
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

/// Synchronize all initialized CSR arrays.
template < typename ValueType, typename IndexType, typename Backend >
void Sync( const CSRMatrix< ValueType, IndexType, Backend > & matrix )
{
   Sync( matrix.row_ptr );
   Sync( matrix.col_ind );
   Sync( matrix.values );
}

} // namespace gendil
