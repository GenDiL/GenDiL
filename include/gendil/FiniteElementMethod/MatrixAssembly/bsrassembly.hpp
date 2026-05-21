// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace gendil {

template <typename ValueType = Real, typename IndexType = GlobalIndex>
auto MakeBlockDiagonalDGBSRPattern( const GlobalIndex num_elements, const IndexType block_rows, const IndexType block_cols )
{
   BSRMatrix<ValueType, IndexType> bsr_matrix{};
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

template <typename ValueType = Real, typename IndexType = GlobalIndex, typename Mesh, typename FiniteElement, typename Restriction >
auto MakeBlockDiagonalDGBSRPattern( const FiniteElementSpace<Mesh, FiniteElement, Restriction>& fe_space )
{
   const auto num_elements = fe_space.GetNumberOfFiniteElements();
   const auto block_rows = fe_space.finite_element.GetNumDofs();
   const auto block_cols = fe_space.finite_element.GetNumDofs();

   return MakeBlockDiagonalDGBSRPattern<ValueType, IndexType>( num_elements, block_rows, block_cols );
}

template <typename ValueType = Real, typename IndexType = GlobalIndex, typename FESpace>
auto MakeDGBSRPattern(const FESpace& fe_space)
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

   BSRMatrix<ValueType, IndexType> bsr_matrix{};
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

template <typename ValueType, typename IndexType>
GENDIL_HOST_DEVICE
IndexType FindBSRBlockIndex(
   const BSRMatrix<ValueType, IndexType>& bsr_matrix,
   const IndexType row_block,
   const IndexType col_block )
{
   constexpr IndexType Invalid = std::numeric_limits<IndexType>::max();

   for (IndexType k = bsr_matrix.row_offsets[row_block];
        k < bsr_matrix.row_offsets[row_block + 1];
        ++k)
   {
      if (bsr_matrix.col_indices[k] == col_block)
      {
         return k;
      }
   }

   return Invalid;
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofIndices,
   typename ElementVector,
   typename ValueType,
   typename IndexType >
GENDIL_HOST_DEVICE
void SetSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const TrialDofIndices & trial_dof_indices,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType> & bsr_matrix )
{
   using TrialFE = typename std::remove_cvref_t<TrialFESpace>::finite_element_type;
   using TestFE  = typename std::remove_cvref_t<TestFESpace>::finite_element_type;

   using TrialDofShape =
      orders_to_num_dofs<typename TrialFE::shape_functions::orders>;
   using TestDofShape =
      orders_to_num_dofs<typename TestFE::shape_functions::orders>;

   constexpr LocalIndex ntrial = Product(TrialDofShape{}); // replace with your utility
   constexpr LocalIndex ntest  = Product(TestDofShape{});  // replace with your utility

   const LocalIndex local_col =
      FlattenMultiIndex<TrialDofShape>(trial_dof_indices);

   // DG-local case: exactly one block per element, located at block index = element_index
   const GlobalIndex block_index = element_index; // TODO: generalize for face contributions
   const GlobalIndex block_offset = block_index * ntest * ntrial;

   using DofShape = orders_to_num_dofs< typename TestFESpace::finite_element_type::shape_functions::orders >;
   using tshape = subsequence_t< DofShape, typename KernelContext::template threaded_dimensions< DofShape::size() > >;
   using rshape = subsequence_t< DofShape, typename KernelContext::template register_dimensions< DofShape::size() > >;

   ThreadLoop< tshape >( kernel_context, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         const ValueType value = y(k...);
         LocalIndex local_row = FlattenMultiIndex<TestDofShape>(std::array{ t..., k... });

         if constexpr (bsr_matrix.block_layout == BlockLayout::ColumnMajor)
         {
            bsr_matrix.values[block_offset + local_col * ntest + local_row] += value;
         }
         else
         {
            bsr_matrix.values[block_offset + local_row * ntrial + local_col] += value;
         }
      });
   });
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ValueType,
   typename IndexType >
requires is_local_dof_descriptor_v< TrialDofDescriptor >
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType> & bsr_matrix )
{
   using Matrix = BSRMatrix<ValueType, IndexType>;
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;

   constexpr LocalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr LocalIndex ntest = LocalDofCount< TestShapeFunctions >();

   const LocalIndex local_col =
      FlattenLocalDof(
         trial_fe_space,
         typename std::remove_cvref_t< TrialDofDescriptor >::component{},
         trial_dof.indices );

   const IndexType block_index =
      FindBSRBlockIndex(
         bsr_matrix,
         static_cast<IndexType>(row_element_index),
         static_cast<IndexType>(col_element_index) );

   constexpr IndexType Invalid = std::numeric_limits<IndexType>::max();
   GENDIL_VERIFY(block_index != Invalid,
      "Missing BSR block for row element / col element pair.");

   const GlobalIndex block_offset = block_index * ntest * ntrial;

   ForEachLocalResidualDof(
      kernel_context,
      test_fe_space,
      y,
      [&] ( const auto & test_dof, const auto & value )
      {
         const LocalIndex local_row =
            FlattenLocalDof(
               test_fe_space,
               typename std::remove_cvref_t< decltype(test_dof) >::component{},
               test_dof.indices );

         if constexpr (Matrix::block_layout == BlockLayout::ColumnMajor)
         {
            bsr_matrix.values[block_offset + local_col * ntest + local_row] += value;
         }
         else
         {
            bsr_matrix.values[block_offset + local_row * ntrial + local_col] += value;
         }
      });
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofIndices,
   typename ElementVector,
   typename ValueType,
   typename IndexType >
requires (!is_local_dof_descriptor_v< TrialDofIndices >)
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofIndices & trial_dof_indices,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType> & bsr_matrix )
{
   using Matrix = BSRMatrix<ValueType, IndexType>;
   using TrialFE = typename std::remove_cvref_t<TrialFESpace>::finite_element_type;
   using TestFE  = typename std::remove_cvref_t<TestFESpace>::finite_element_type;

   using TrialDofShape =
      orders_to_num_dofs<typename TrialFE::shape_functions::orders>;
   using TestDofShape =
      orders_to_num_dofs<typename TestFE::shape_functions::orders>;

   constexpr LocalIndex ntrial = Product(TrialDofShape{});
   constexpr LocalIndex ntest  = Product(TestDofShape{});

   const LocalIndex local_col =
      FlattenMultiIndex<TrialDofShape>(trial_dof_indices);

   const IndexType block_index =
      FindBSRBlockIndex(
         bsr_matrix,
         static_cast<IndexType>(row_element_index),
         static_cast<IndexType>(col_element_index) );

   constexpr IndexType Invalid = std::numeric_limits<IndexType>::max();
   GENDIL_VERIFY(block_index != Invalid,
      "Missing BSR block for row element / col element pair.");

   const GlobalIndex block_offset = block_index * ntest * ntrial;

   using DofShape =
      orders_to_num_dofs<typename TestFESpace::finite_element_type::shape_functions::orders>;
   using tshape =
      subsequence_t<DofShape, typename KernelContext::template threaded_dimensions<DofShape::size()>>;
   using rshape =
      subsequence_t<DofShape, typename KernelContext::template register_dimensions<DofShape::size()>>;

   ThreadLoop<tshape>(kernel_context, [&] (auto... t)
   {
      UnitLoop<rshape>([&] (auto... k)
      {
         const ValueType value = y(k...);
         const LocalIndex local_row =
            FlattenMultiIndex<TestDofShape>(std::array<GlobalIndex, sizeof...(t) + sizeof...(k)>{ t..., k... });

         if constexpr (Matrix::block_layout == BlockLayout::ColumnMajor)
         {
            bsr_matrix.values[block_offset + local_col * ntest + local_row] += value;
         }
         else
         {
            bsr_matrix.values[block_offset + local_row * ntrial + local_col] += value;
         }
      });
   });
}

}
