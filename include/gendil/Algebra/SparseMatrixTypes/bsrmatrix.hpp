// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"

namespace gendil {

enum class BlockLayout
{
   RowMajor,
   ColumnMajor
};

template <typename ValueType = Real, typename IndexType = GlobalIndex, BlockLayout Layout = BlockLayout::ColumnMajor>
struct BSRMatrix
{
   IndexType block_rows = 0; // ntest
   IndexType block_cols = 0; // ntrial
   IndexType num_row_blocks = 0;
   IndexType num_col_blocks = 0;
   IndexType num_blocks = 0;

   static constexpr BlockLayout block_layout = Layout;

   HostDevicePointer<ValueType> values;       // size = num_blocks * block_rows * block_cols
   HostDevicePointer<IndexType> row_offsets;  // size = num_row_blocks + 1
   HostDevicePointer<IndexType> col_indices;  // size = num_blocks

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
   GENDIL_HOST_DEVICE
   void operator()( const InputVector & x, OutputVector & y ) const
   {
      #pragma omp parallel for
      for (IndexType block_row = 0; block_row < num_row_blocks; ++block_row)
      {
         for (IndexType local_row = 0; local_row < block_rows; ++local_row)
         {
            ValueType sum = ValueType(0);

            for (IndexType block_it = row_offsets[block_row];
                 block_it < row_offsets[block_row + 1];
                 ++block_it)
            {
               const IndexType block_col = col_indices[block_it];

               for (IndexType local_col = 0; local_col < block_cols; ++local_col)
               {
                  const IndexType global_col = block_col * block_cols + local_col;

                  sum += GetBlockEntry(block_it, local_row, local_col) * x[global_col];
               }
            }

            const IndexType global_row = block_row * block_rows + local_row;
            y[global_row] = sum;
         }
      }
   }
};

} // namespace gendil
