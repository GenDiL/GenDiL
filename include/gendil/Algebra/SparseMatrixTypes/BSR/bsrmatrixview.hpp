// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixstorage.hpp"

#include <type_traits>

namespace gendil
{

/**
 * Borrowed BSR view for one selected memory space.
 *
 * ValueType and IndexType may be const-qualified to express access. The view
 * is device-copyable, contains no ownership or backend state, and must not
 * outlive its matrix.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor >
struct BSRMatrixView
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;

   index_type block_rows = 0;
   index_type block_cols = 0;
   index_type num_row_blocks = 0;
   index_type num_col_blocks = 0;
   index_type num_blocks = 0;

   static constexpr BlockLayout block_layout = Layout;

   ValueType * values = nullptr;
   IndexType * row_offsets = nullptr;
   IndexType * col_indices = nullptr;

   /// Return the flat offset of one entry in the selected block layout.
   GENDIL_HOST_DEVICE
   constexpr index_type GetBlockEntryOffset(
      const index_type block_index,
      const index_type local_row,
      const index_type local_col ) const
   {
      const index_type block_offset =
         block_index * block_rows * block_cols;

      if constexpr ( block_layout == BlockLayout::ColumnMajor )
      {
         return block_offset + local_col * block_rows + local_row;
      }
      else
      {
         return block_offset + local_row * block_cols + local_col;
      }
   }

   /// Return a reference to one entry in the selected block layout.
   GENDIL_HOST_DEVICE
   constexpr ValueType & GetBlockEntry(
      const index_type block_index,
      const index_type local_row,
      const index_type local_col ) const
   {
      return values[
         GetBlockEntryOffset( block_index, local_row, local_col ) ];
   }
};

/// Return a host read view, synchronizing all arrays.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetHostReadView(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< const ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadHost( matrix.values ),
      ReadHost( matrix.row_offsets ),
      ReadHost( matrix.col_indices ) };
}

/// Return a host read-write view, preserving contents and invalidating device
/// copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetHostReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadWriteHost( matrix.values ),
      ReadWriteHost( matrix.row_offsets ),
      ReadWriteHost( matrix.col_indices ) };
}

/// Return a host write view without synchronizing old contents, invalidating
/// device copies without changing cached vendor state.
/// Cached vendor backend should be reset after modifying sparse structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetHostWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      WriteHost( matrix.values ),
      WriteHost( matrix.row_offsets ),
      WriteHost( matrix.col_indices ) };
}

/// Return a device read view, synchronizing all arrays.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetDeviceReadView(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< const ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadDevice( matrix.values ),
      ReadDevice( matrix.row_offsets ),
      ReadDevice( matrix.col_indices ) };
}

/// Return a device read-write view, preserving contents and invalidating host
/// copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetDeviceReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadWriteDevice( matrix.values ),
      ReadWriteDevice( matrix.row_offsets ),
      ReadWriteDevice( matrix.col_indices ) };
}

/// Return a device write view without synchronizing old contents, invalidating
/// host copies.
/// Cached vendor backend should be reset after modifying sparse structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetDeviceWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      WriteDevice( matrix.values ),
      WriteDevice( matrix.row_offsets ),
      WriteDevice( matrix.col_indices ) };
}

/// Return a host values read-write view, synchronizing values and structure,
/// invalidating device values.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetHostValuesReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadWriteHost( matrix.values ),
      ReadHost( matrix.row_offsets ),
      ReadHost( matrix.col_indices ) };
}

/// Return a host values write view without synchronizing old values, while
/// reading structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetHostValuesWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      WriteHost( matrix.values ),
      ReadHost( matrix.row_offsets ),
      ReadHost( matrix.col_indices ) };
}

/// Return a device values read-write view, synchronizing values and structure,
/// invalidating host values.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetDeviceValuesReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      ReadWriteDevice( matrix.values ),
      ReadDevice( matrix.row_offsets ),
      ReadDevice( matrix.col_indices ) };
}

/// Return a device values write view without synchronizing old values, while
/// reading structure.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetDeviceValuesWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return BSRMatrixView< ValueType, const IndexType, Layout >{
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks,
      WriteDevice( matrix.values ),
      ReadDevice( matrix.row_offsets ),
      ReadDevice( matrix.col_indices ) };
}

/// Return a read view, synchronizing all arrays to the selected memory space.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetKernelReadView(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
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
   BlockLayout Layout,
   typename Backend >
auto GetKernelReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
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
   BlockLayout Layout,
   typename Backend >
auto GetKernelWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
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

/// Return a values read-write view, preserving values and reading structure.
/// Cached vendor sparsity preprocessing remains valid.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetKernelValuesReadWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceValuesReadWriteView( matrix );
   }
   else
   {
      return GetHostValuesReadWriteView( matrix );
   }
}

/// Return a values write view without syncing values, while reading structure.
/// Cached vendor sparsity preprocessing remains valid.
template <
   bool OnDevice,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
auto GetKernelValuesWriteView(
   BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceValuesWriteView( matrix );
   }
   else
   {
      return GetHostValuesWriteView( matrix );
   }
}

/// Synchronize all initialized BSR arrays.
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
void Sync(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   Sync( matrix.values );
   Sync( matrix.row_offsets );
   Sync( matrix.col_indices );
}

} // namespace gendil
