// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrix.hpp"
#include "gendil/FiniteElementMethod/Restrictions/finiteelementdoflayout.hpp"
#include "gendil/FiniteElementMethod/finiteelementspace.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/DoFIO/localdofoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

#include <limits>

namespace gendil {

template <
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
GENDIL_HOST_DEVICE
auto FindBSRBlockIndex(
   const BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix,
   const typename BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout >::index_type row_block,
   const typename BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout >::index_type col_block )
{
   using MatrixView =
      BSRMatrixView< ViewValueType, ViewIndexType, Layout >;
   using IndexType = typename MatrixView::index_type;
   constexpr IndexType Invalid = std::numeric_limits< IndexType >::max();

   for ( IndexType k = bsr_matrix.row_offsets[row_block];
         k < bsr_matrix.row_offsets[row_block + 1];
         ++k )
   {
      if ( bsr_matrix.col_indices[k] == col_block )
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
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
GENDIL_HOST_DEVICE
void SetSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const TrialDofIndices & trial_dof_indices,
   const ElementVector & y,
   BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix )
{
   using MatrixView =
      BSRMatrixView< ViewValueType, ViewIndexType, Layout >;
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using TrialShapeFunctions =
      finite_element_space_shape_functions_t< TrialFESpace >;
   using TestShapeFunctions =
      finite_element_space_shape_functions_t< TestFESpace >;

   using TrialDofShape =
      orders_to_num_dofs< typename TrialShapeFunctions::orders >;
   using TestDofShape =
      orders_to_num_dofs< typename TestShapeFunctions::orders >;

   const LocalIndex local_col =
      FlattenMultiIndex< TrialDofShape >( trial_dof_indices );
   const GlobalIndex block_index = element_index;

   using DofShape = TestDofShape;
   using tshape =
      subsequence_t<
         DofShape,
         typename KernelContext::template threaded_dimensions<
            DofShape::size() > >;
   using rshape =
      subsequence_t<
         DofShape,
         typename KernelContext::template register_dimensions<
            DofShape::size() > >;

   ThreadLoop< tshape >( kernel_context, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         const ValueType value = y(k...);
         const LocalIndex local_row =
            FlattenMultiIndex< TestDofShape >(
               std::array{ t..., k... } );

         bsr_matrix.GetBlockEntry(
            static_cast< IndexType >( block_index ),
            static_cast< IndexType >( local_row ),
            static_cast< IndexType >( local_col ) ) += value;
      });
   });
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
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
   BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix )
{
   using MatrixView =
      BSRMatrixView< ViewValueType, ViewIndexType, Layout >;
   using IndexType = typename MatrixView::index_type;
   const GlobalIndex local_col =
      FlattenLocalDof(
         trial_fe_space,
         typename std::remove_cvref_t<
            TrialDofDescriptor >::component{},
         trial_dof.indices );

   const IndexType block_index =
      FindBSRBlockIndex(
         bsr_matrix,
         static_cast< IndexType >( row_element_index ),
         static_cast< IndexType >( col_element_index ) );

   constexpr IndexType Invalid = std::numeric_limits< IndexType >::max();
   GENDIL_VERIFY(
      block_index != Invalid,
      "Missing BSR block for row element / col element pair." );

   ForEachLocalResidualDof(
      kernel_context,
      test_fe_space,
      y,
      [&] ( const auto & test_dof, const auto & value )
      {
         const GlobalIndex local_row =
            FlattenLocalDof(
               test_fe_space,
               typename std::remove_cvref_t<
                  decltype(test_dof) >::component{},
               test_dof.indices );

         bsr_matrix.GetBlockEntry(
            block_index,
            static_cast< IndexType >( local_row ),
            static_cast< IndexType >( local_col ) ) += value;
      });
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofIndices,
   typename ElementVector,
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
requires ( !is_local_dof_descriptor_v< TrialDofIndices > )
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofIndices & trial_dof_indices,
   const ElementVector & y,
   BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix )
{
   using MatrixView =
      BSRMatrixView< ViewValueType, ViewIndexType, Layout >;
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using TrialShapeFunctions =
      finite_element_space_shape_functions_t< TrialFESpace >;
   using TestShapeFunctions =
      finite_element_space_shape_functions_t< TestFESpace >;

   using TrialDofShape =
      orders_to_num_dofs< typename TrialShapeFunctions::orders >;
   using TestDofShape =
      orders_to_num_dofs< typename TestShapeFunctions::orders >;

   const LocalIndex local_col =
      FlattenMultiIndex< TrialDofShape >( trial_dof_indices );

   const IndexType block_index =
      FindBSRBlockIndex(
         bsr_matrix,
         static_cast< IndexType >( row_element_index ),
         static_cast< IndexType >( col_element_index ) );

   constexpr IndexType Invalid = std::numeric_limits< IndexType >::max();
   GENDIL_VERIFY(
      block_index != Invalid,
      "Missing BSR block for row element / col element pair." );

   using DofShape = TestDofShape;
   using tshape =
      subsequence_t<
         DofShape,
         typename KernelContext::template threaded_dimensions<
            DofShape::size() > >;
   using rshape =
      subsequence_t<
         DofShape,
         typename KernelContext::template register_dimensions<
            DofShape::size() > >;

   ThreadLoop< tshape >( kernel_context, [&] ( auto... t )
   {
      UnitLoop< rshape >( [&] ( auto... k )
      {
         const ValueType value = y(k...);
         const LocalIndex local_row =
            FlattenMultiIndex< TestDofShape >(
               std::array<
                  GlobalIndex,
                  sizeof...(t) + sizeof...(k) >{ t..., k... } );

         bsr_matrix.GetBlockEntry(
            block_index,
            static_cast< IndexType >( local_row ),
            static_cast< IndexType >( local_col ) ) += value;
      });
   });
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix )
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
   typename ViewValueType,
   typename ViewIndexType,
   BlockLayout Layout >
GENDIL_HOST_DEVICE
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & element_index,
   const FaceInfo & face_info,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   BSRMatrixView<
      ViewValueType,
      ViewIndexType,
      Layout > & bsr_matrix )
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

} // namespace gendil
