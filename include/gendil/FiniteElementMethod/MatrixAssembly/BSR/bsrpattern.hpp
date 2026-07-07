// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrix.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"

#include <algorithm>
#include <vector>

namespace gendil {

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend >
auto MakeBlockDiagonalDGBSRPattern(
   const GlobalIndex num_elements,
   const IndexType block_rows,
   const IndexType block_cols,
   Backend backend = Backend{} )
{
   BSRMatrix<ValueType, IndexType, Layout, Backend> bsr_matrix{};
   bsr_matrix.backend = backend;
   bsr_matrix.block_rows = block_rows;
   bsr_matrix.block_cols = block_cols;
   bsr_matrix.num_row_blocks = num_elements;
   bsr_matrix.num_col_blocks = num_elements;
   bsr_matrix.num_blocks = num_elements;
   AllocateHostPointer( bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets );
   AllocateDevicePointer( bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets );
   AllocateHostPointer( bsr_matrix.num_blocks, bsr_matrix.col_indices );
   AllocateDevicePointer( bsr_matrix.num_blocks, bsr_matrix.col_indices );
   AllocateHostPointer( bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values );
   AllocateDevicePointer( bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values );

   for (GlobalIndex e = 0; e <= num_elements; ++e)
   {
      bsr_matrix.row_offsets[e] = e;
   }

   for (GlobalIndex e = 0; e < num_elements; ++e)
   {
      bsr_matrix.col_indices[e] = e;
   }

   for (GlobalIndex i = 0; i < num_elements * block_rows * block_cols; ++i)
   {
      bsr_matrix.values[i] = 0.0;
   }

   ToDevice( bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets );
   ToDevice( bsr_matrix.num_blocks, bsr_matrix.col_indices );
   ToDevice( bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values );

   return bsr_matrix;
}

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend,
   typename Mesh,
   typename FiniteElement,
   typename Restriction >
auto MakeBlockDiagonalDGBSRPattern(
   const FiniteElementSpace<Mesh, FiniteElement, Restriction>& fe_space,
   Backend backend = Backend{} )
{
   const auto num_elements = fe_space.GetNumberOfFiniteElements();
   const auto block_rows = fe_space.finite_element.GetNumDofs();
   const auto block_cols = fe_space.finite_element.GetNumDofs();

   return MakeBlockDiagonalDGBSRPattern<ValueType, IndexType, Layout, Backend>(
      num_elements,
      block_rows,
      block_cols,
      backend );
}

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend,
   typename FESpace>
auto MakeDGBSRPattern(
   const FESpace& fe_space,
   Backend backend = Backend{} )
{
   const GlobalIndex num_elements = fe_space.GetNumberOfFiniteElements();
   const IndexType block_rows = fe_space.finite_element.GetNumDofs();
   const IndexType block_cols = fe_space.finite_element.GetNumDofs();

   std::vector<IndexType> host_row_offsets(num_elements + 1, 0);
   std::vector<IndexType> host_col_indices;

   for (GlobalIndex e = 0; e < num_elements; ++e)
   {
      std::vector<IndexType> cols;
      cols.push_back(e); // diagonal block always present

      FaceLoop(
         fe_space,
         e,
         [&] (auto const& face_info)
         {
            if ( !IsBoundaryFace( face_info ) )
            {
               const IndexType nb = static_cast<IndexType>(face_info.PlusSide().GetCellIndex());
               cols.push_back(nb);
            }
         });

      std::sort(cols.begin(), cols.end());
      cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

      host_row_offsets[e + 1] = host_row_offsets[e] + static_cast<IndexType>(cols.size());
      host_col_indices.insert(host_col_indices.end(), cols.begin(), cols.end());
   }

   BSRMatrix<ValueType, IndexType, Layout, Backend> bsr_matrix{};
   bsr_matrix.backend = backend;
   bsr_matrix.block_rows = block_rows;
   bsr_matrix.block_cols = block_cols;
   bsr_matrix.num_row_blocks = num_elements;
   bsr_matrix.num_col_blocks = num_elements;
   bsr_matrix.num_blocks = static_cast<IndexType>(host_col_indices.size());

   AllocateHostPointer(bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets);
   AllocateDevicePointer(bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets);

   AllocateHostPointer(bsr_matrix.num_blocks, bsr_matrix.col_indices);
   AllocateDevicePointer(bsr_matrix.num_blocks, bsr_matrix.col_indices);

   AllocateHostPointer(bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values);
   AllocateDevicePointer(bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values);

   for (IndexType i = 0; i < bsr_matrix.num_row_blocks + 1; ++i)
   {
      bsr_matrix.row_offsets[i] = host_row_offsets[i];
   }

   for (IndexType i = 0; i < bsr_matrix.num_blocks; ++i)
   {
      bsr_matrix.col_indices[i] = host_col_indices[i];
   }

   for (IndexType i = 0; i < bsr_matrix.num_blocks * block_rows * block_cols; ++i)
   {
      bsr_matrix.values[i] = ValueType(0);
   }

   ToDevice(bsr_matrix.num_row_blocks + 1, bsr_matrix.row_offsets);
   ToDevice(bsr_matrix.num_blocks, bsr_matrix.col_indices);
   ToDevice(bsr_matrix.num_blocks * block_rows * block_cols, bsr_matrix.values);

   return bsr_matrix;
}

}
