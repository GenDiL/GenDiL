// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrix.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"

#include <algorithm>
#include <utility>
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
   bsr_matrix.block_rows = block_rows;
   bsr_matrix.block_cols = block_cols;
   bsr_matrix.num_row_blocks = num_elements;
   bsr_matrix.num_col_blocks = num_elements;
   bsr_matrix.num_blocks = num_elements;
   bsr_matrix.row_offsets =
      MakeSyncHostDeviceArray< IndexType >(
         bsr_matrix.num_row_blocks + IndexType( 1 ) );
   bsr_matrix.col_indices =
      MakeSyncHostDeviceArray< IndexType >( bsr_matrix.num_blocks );
   bsr_matrix.values =
      MakeSyncHostDeviceArray< ValueType >(
         bsr_matrix.num_blocks * block_rows * block_cols );
   bsr_matrix.backend = std::move( backend );

   auto data = GetHostWriteView( bsr_matrix );
   for (GlobalIndex e = 0; e <= num_elements; ++e)
   {
      data.row_offsets[e] = e;
   }

   for (GlobalIndex e = 0; e < num_elements; ++e)
   {
      data.col_indices[e] = e;
   }

   for (GlobalIndex i = 0; i < num_elements * block_rows * block_cols; ++i)
   {
      data.values[i] = 0.0;
   }

   Sync( bsr_matrix );
   ConfigureBSRBackend(
      bsr_matrix.backend,
      bsr_matrix.block_rows,
      bsr_matrix.block_cols,
      bsr_matrix.num_row_blocks,
      bsr_matrix.num_col_blocks,
      bsr_matrix.num_blocks );

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
      std::move( backend ) );
}

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend,
   typename TrialFESpace,
   typename TestFESpace >
auto MakeDGBSRPattern(
   const TrialFESpace & trial_space,
   const TestFESpace & test_space,
   Backend backend )
{
   const IndexType num_row_blocks =
      static_cast< IndexType >(
         test_space.GetNumberOfFiniteElements() );
   const IndexType num_col_blocks =
      static_cast< IndexType >(
         trial_space.GetNumberOfFiniteElements() );
   GENDIL_VERIFY(
      num_row_blocks == num_col_blocks,
      "MakeDGBSRPattern currently requires trial and test spaces with "
      "matching element topology." );

   const IndexType block_rows =
      static_cast< IndexType >(
         test_space.finite_element.GetNumDofs() );
   const IndexType block_cols =
      static_cast< IndexType >(
         trial_space.finite_element.GetNumDofs() );

   std::vector<IndexType> host_row_offsets(num_row_blocks + 1, 0);
   std::vector<IndexType> host_col_indices;

   for (IndexType e = 0; e < num_row_blocks; ++e)
   {
      std::vector<IndexType> cols;
      cols.push_back(e); // diagonal block always present

      FaceLoop(
         test_space,
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
   bsr_matrix.block_rows = block_rows;
   bsr_matrix.block_cols = block_cols;
   bsr_matrix.num_row_blocks = num_row_blocks;
   bsr_matrix.num_col_blocks = num_col_blocks;
   bsr_matrix.num_blocks =
      static_cast< IndexType >( host_col_indices.size() );
   bsr_matrix.row_offsets =
      MakeSyncHostDeviceArray< IndexType >(
         bsr_matrix.num_row_blocks + IndexType( 1 ) );
   bsr_matrix.col_indices =
      MakeSyncHostDeviceArray< IndexType >( bsr_matrix.num_blocks );
   bsr_matrix.values =
      MakeSyncHostDeviceArray< ValueType >(
         bsr_matrix.num_blocks * block_rows * block_cols );
   bsr_matrix.backend = std::move( backend );

   auto data = GetHostWriteView( bsr_matrix );
   for ( IndexType i = 0; i < bsr_matrix.num_row_blocks + 1; ++i )
   {
      data.row_offsets[i] = host_row_offsets[i];
   }

   for ( IndexType i = 0; i < bsr_matrix.num_blocks; ++i )
   {
      data.col_indices[i] = host_col_indices[i];
   }

   for ( IndexType i = 0;
         i < bsr_matrix.num_blocks * block_rows * block_cols;
         ++i )
   {
      data.values[i] = ValueType(0);
   }

   Sync( bsr_matrix );
   ConfigureBSRBackend(
      bsr_matrix.backend,
      bsr_matrix.block_rows,
      bsr_matrix.block_cols,
      bsr_matrix.num_row_blocks,
      bsr_matrix.num_col_blocks,
      bsr_matrix.num_blocks );

   return bsr_matrix;
}

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend,
   typename FESpace >
auto MakeDGBSRPattern(
   const FESpace & fe_space,
   Backend backend = Backend{} )
{
   return MakeDGBSRPattern<
      ValueType,
      IndexType,
      Layout,
      Backend >(
         fe_space,
         fe_space,
         std::move( backend ) );
}

}
