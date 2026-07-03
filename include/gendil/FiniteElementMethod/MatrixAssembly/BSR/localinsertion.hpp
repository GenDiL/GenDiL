// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/bsrmatrix.hpp"
#include "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

#include <limits>

namespace gendil {

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
GENDIL_HOST_DEVICE
IndexType FindBSRBlockIndex(
   const BSRMatrix<ValueType, IndexType, Layout, Backend>& bsr_matrix,
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
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
GENDIL_HOST_DEVICE
void SetSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const TrialDofIndices & trial_dof_indices,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType, Layout, Backend> & bsr_matrix )
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
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
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
   BSRMatrix<ValueType, IndexType, Layout, Backend> & bsr_matrix )
{
   using Matrix = BSRMatrix<ValueType, IndexType, Layout, Backend>;
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;

   constexpr GlobalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex ntest = LocalDofCount< TestShapeFunctions >();

   const GlobalIndex local_col =
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
         const GlobalIndex local_row =
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
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
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
   BSRMatrix<ValueType, IndexType, Layout, Backend> & bsr_matrix )
{
   using Matrix = BSRMatrix<ValueType, IndexType, Layout, Backend>;
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

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType, Layout, Backend> & bsr_matrix )
{
   AddSparseMatrixEntry(
      kernel_context,
      trial_fe_space,
      test_fe_space,
      element_index,
      element_index,
      trial_dof,
      y,
      bsr_matrix );
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename FaceInfo,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const FaceInfo & face_info,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   BSRMatrix<ValueType, IndexType, Layout, Backend> & bsr_matrix )
{
   AddSparseMatrixEntry(
      kernel_context,
      trial_fe_space,
      test_fe_space,
      element_index,
      face_info.PlusSide().GetCellIndex(),
      trial_dof,
      y,
      bsr_matrix );
}

}
