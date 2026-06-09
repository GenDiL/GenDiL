// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"

namespace gendil {

enum class BlockLayout
{
   RowMajor,
   ColumnMajor
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct HostBSRBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct NativeDeviceBSRBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

// Standalone BSR storage default. GenericAssembly dispatch uses
// DefaultBackendFor_t<MatrixAssemblyType::BSR> instead.
using DefaultBSRBackend = HostBSRBackend<>;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend >
struct BSRMatrix;

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
struct BSRMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType block_rows = 0; // ntest
   IndexType block_cols = 0; // ntrial
   IndexType num_row_blocks = 0;
   IndexType num_col_blocks = 0;
   IndexType num_blocks = 0;

   static constexpr BlockLayout block_layout = Layout;

   HostDevicePointer<ValueType> values;       // size = num_blocks * block_rows * block_cols
   HostDevicePointer<IndexType> row_offsets;  // size = num_row_blocks + 1
   HostDevicePointer<IndexType> col_indices;  // size = num_blocks

   // BSRMatrix is currently a shallow handle over HostDevicePointer buffers.
   // Backend objects stored here must therefore be safely copyable as long as
   // BSRMatrix itself remains copyable.
   Backend backend{};

   GENDIL_HOST_DEVICE
   constexpr ValueType GetBlockEntry(
      const IndexType block_index,
      const IndexType local_row,
      const IndexType local_col ) const
   {
      const IndexType block_offset = block_index * block_rows * block_cols;

      if constexpr (block_layout == BlockLayout::ColumnMajor)
      {
         return values[block_offset + local_col * block_rows + local_row];
      }
      else
      {
         return values[block_offset + local_row * block_cols + local_col];
      }
   }

   template <typename InputVector, typename OutputVector>
   void operator()( const InputVector & x, OutputVector & y ) const;
};

} // namespace gendil
