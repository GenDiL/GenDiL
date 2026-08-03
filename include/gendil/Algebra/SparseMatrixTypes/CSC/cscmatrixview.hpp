// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixstorage.hpp"

#include <type_traits>

namespace gendil
{

/**
 * Borrowed CSC view for one selected memory space.
 *
 * Const-qualified template arguments express read-only values and indices.
 * The device-copyable view owns nothing and must not outlive its matrix.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct CSCMatrixView
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;

   index_type num_rows = 0;
   index_type num_cols = 0;
   index_type nnz = 0;

   IndexType * col_ptr = nullptr;
   IndexType * row_ind = nullptr;
   ValueType * values = nullptr;
};

/// Return a host read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadView(
   const CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.col_ptr ),
      ReadHost( matrix.row_ind ),
      ReadHost( matrix.values ) };
}

/// Return a host read-write view, preserving contents and invalidating device
/// copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostReadWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteHost( matrix.col_ptr ),
      ReadWriteHost( matrix.row_ind ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host write view without synchronizing old contents, invalidating
/// device copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteHost( matrix.col_ptr ),
      WriteHost( matrix.row_ind ),
      WriteHost( matrix.values ) };
}

/// Return a device read view, synchronizing all arrays.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadView(
   const CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< const ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.col_ptr ),
      ReadDevice( matrix.row_ind ),
      ReadDevice( matrix.values ) };
}

/// Return a device read-write view, preserving contents and invalidating host
/// copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceReadWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadWriteDevice( matrix.col_ptr ),
      ReadWriteDevice( matrix.row_ind ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device write view without synchronizing old contents, invalidating
/// host copies.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      WriteDevice( matrix.col_ptr ),
      WriteDevice( matrix.row_ind ),
      WriteDevice( matrix.values ) };
}

/// Return a host values read-write view, synchronizing values and structure,
/// invalidating device values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesReadWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.col_ptr ),
      ReadHost( matrix.row_ind ),
      ReadWriteHost( matrix.values ) };
}

/// Return a host values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetHostValuesWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadHost( matrix.col_ptr ),
      ReadHost( matrix.row_ind ),
      WriteHost( matrix.values ) };
}

/// Return a device values read-write view, synchronizing values and structure,
/// invalidating host values.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesReadWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.col_ptr ),
      ReadDevice( matrix.row_ind ),
      ReadWriteDevice( matrix.values ) };
}

/// Return a device values write view without synchronizing old values, while
/// reading structure.
template < typename ValueType, typename IndexType, typename Backend >
auto GetDeviceValuesWriteView(
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   return CSCMatrixView< ValueType, const IndexType >{
      matrix.num_rows,
      matrix.num_cols,
      matrix.nnz,
      ReadDevice( matrix.col_ptr ),
      ReadDevice( matrix.row_ind ),
      WriteDevice( matrix.values ) };
}

/// Return a read view, synchronizing all arrays to the selected memory space.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   typename Backend >
auto GetKernelReadView(
   const CSCMatrix< ValueType, IndexType, Backend > & matrix )
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
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
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
   CSCMatrix< ValueType, IndexType, Backend > & matrix )
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

/// Synchronize all initialized CSC arrays.
template < typename ValueType, typename IndexType, typename Backend >
void Sync( const CSCMatrix< ValueType, IndexType, Backend > & matrix )
{
   Sync( matrix.col_ptr );
   Sync( matrix.row_ind );
   Sync( matrix.values );
}

} // namespace gendil
