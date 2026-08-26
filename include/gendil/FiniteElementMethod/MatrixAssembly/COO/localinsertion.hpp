// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

namespace gendil {

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename COOBuffer >
requires is_local_dof_descriptor_v< TrialDofDescriptor >
GENDIL_HOST_DEVICE
void AddRawCOOBlockEntries(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   const GlobalIndex raw_entry_base,
   COOBuffer & coo_buffer )
{
   using Buffer = std::remove_cvref_t< COOBuffer >;
   using ValueType = typename Buffer::value_type;
   using IndexType = typename Buffer::index_type;
   using TestShapeFunctions =
      finite_element_space_shape_functions_t< TestFESpace >;
   using TrialDescriptor = std::remove_cvref_t< TrialDofDescriptor >;

   constexpr GlobalIndex ntest = LocalDofCount< TestShapeFunctions >();
   const auto block_entry_count = coo_buffer.block_entry_count;

   GENDIL_VERIFY(
      IsActiveRawCOOOffset(
         raw_entry_base,
         block_entry_count,
         static_cast< GlobalIndex >( coo_buffer.nnz_raw ) ),
      "Raw COO emission received an inactive or out-of-range block offset." );

   const GlobalIndex local_col =
      FlattenLocalDof(
         trial_fe_space,
         typename TrialDescriptor::component{},
         trial_dof.indices );
   const GlobalIndex algebraic_col =
      GetGlobalDofIndex(
         trial_fe_space,
         col_element_index,
         trial_dof );

   // Compact RawCOO blocks can receive multiple terms. Coordinates are
   // rewritten deterministically on every contribution while values accumulate.
   ForEachLocalResidualDof(
      kernel_context,
      test_fe_space,
      y,
      [&] ( const auto & test_dof, const auto & value )
      {
         using TestDescriptor = std::remove_cvref_t< decltype(test_dof) >;
         const GlobalIndex local_row =
            FlattenLocalDof(
               test_fe_space,
               typename TestDescriptor::component{},
               test_dof.indices );
         const GlobalIndex algebraic_row =
            GetGlobalDofIndex(
               test_fe_space,
               row_element_index,
               test_dof );
         const GlobalIndex raw_index =
            raw_entry_base +
            local_col * ntest +
            local_row;

         GENDIL_VERIFY(
            raw_index < static_cast< GlobalIndex >( coo_buffer.nnz_raw ),
            "Raw COO emission wrote past the allocated triplet buffer." );
         GENDIL_ASSERT(
            algebraic_row < static_cast< GlobalIndex >( coo_buffer.num_rows ),
            "Raw COO row coordinate exceeds the test algebraic DoF extent." );
         GENDIL_ASSERT(
            algebraic_col < static_cast< GlobalIndex >( coo_buffer.num_cols ),
            "Raw COO column coordinate exceeds the trial algebraic DoF extent." );

         coo_buffer.rows[raw_index] = static_cast< IndexType >( algebraic_row );
         coo_buffer.cols[raw_index] = static_cast< IndexType >( algebraic_col );
         coo_buffer.values[raw_index] += static_cast< ValueType >( value );
      });
}

/**
 * @brief Emit one reference-local face block through native oriented DoFs.
 *
 * The block storage order remains reference-local and deterministic.  Only
 * the restriction coordinates are mapped back to each side's native element
 * ordering before global lookup.
 */
template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename TrialOrientation,
   typename TestOrientation,
   typename TrialDofDescriptor,
   typename ElementVector,
   typename ValueType,
   typename IndexType >
requires is_local_dof_descriptor_v<TrialDofDescriptor>
GENDIL_HOST_DEVICE
void AddOrientedRawCOOEntityBlockEntries(
   const KernelContext& kernel_context,
   const TrialFESpace& trial_fe_space,
   const TestFESpace& test_fe_space,
   const GlobalIndex row_element_index,
   const GlobalIndex col_element_index,
   const TrialOrientation& trial_orientation,
   const TestOrientation& test_orientation,
   const TrialDofDescriptor& reference_trial_dof,
   const ElementVector& reference_residual,
   const GlobalIndex entity_index,
   RawCOOEntityBlockTarget<ValueType, IndexType>& target )
{
   using TestShapeFunctions =
      finite_element_space_shape_functions_t<TestFESpace>;
   using TrialDescriptor =
      std::remove_cvref_t<TrialDofDescriptor>;
   constexpr GlobalIndex ntest =
      LocalDofCount<TestShapeFunctions>();

   const GlobalIndex raw_entry_base =
      RawCOOEntityBlockOffset(target, entity_index);
   GENDIL_VERIFY(
      IsActiveRawCOOOffset(
         raw_entry_base,
         target.block_entry_count,
         static_cast<GlobalIndex>(target.nnz_raw)),
      "RawCOO face emission received an out-of-range block offset." );

   const GlobalIndex local_col = FlattenLocalDof(
      trial_fe_space,
      typename TrialDescriptor::component{},
      reference_trial_dof.indices);
   const auto native_trial_dof = OrientReferenceDofToNative(
      trial_fe_space,
      reference_trial_dof,
      trial_orientation);
   const GlobalIndex algebraic_col = GetGlobalDofIndex(
      trial_fe_space,
      col_element_index,
      native_trial_dof);

   ForEachLocalResidualDof(
      kernel_context,
      test_fe_space,
      reference_residual,
      [&] (const auto& reference_test_dof, const auto& value)
      {
         using TestDescriptor =
            std::remove_cvref_t<decltype(reference_test_dof)>;
         const GlobalIndex local_row = FlattenLocalDof(
            test_fe_space,
            typename TestDescriptor::component{},
            reference_test_dof.indices);
         const auto native_test_dof = OrientReferenceDofToNative(
            test_fe_space,
            reference_test_dof,
            test_orientation);
         const GlobalIndex algebraic_row = GetGlobalDofIndex(
            test_fe_space,
            row_element_index,
            native_test_dof);
         const GlobalIndex raw_index =
            raw_entry_base + local_col * ntest + local_row;

         GENDIL_VERIFY(
            raw_index < static_cast<GlobalIndex>(target.nnz_raw),
            "RawCOO face emission wrote past its target segment." );
         GENDIL_ASSERT(
            algebraic_row < static_cast<GlobalIndex>(target.num_rows),
            "RawCOO face row exceeds the test algebraic extent." );
         GENDIL_ASSERT(
            algebraic_col < static_cast<GlobalIndex>(target.num_cols),
            "RawCOO face column exceeds the trial algebraic extent." );

         target.rows[raw_index] = static_cast<IndexType>(algebraic_row);
         target.cols[raw_index] = static_cast<IndexType>(algebraic_col);
         target.values[raw_index] += static_cast<ValueType>(value);
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
   const GlobalIndex & element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   RawCOOAssemblyTarget< ValueType, IndexType > & coo_target )
{
   const GlobalIndex raw_entry_base =
      RawCOODiagonalBlockOffset( coo_target, element_index );

   AddRawCOOBlockEntries(
      kernel_context,
      trial_fe_space,
      test_fe_space,
      element_index,
      element_index,
      trial_dof,
      y,
      raw_entry_base,
      coo_target );
}

template <
   typename KernelContext,
   typename TrialFESpace,
   typename TestFESpace,
   typename FaceInfo,
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
   const GlobalIndex & element_index,
   const FaceInfo & face_info,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   RawCOOAssemblyTarget< ValueType, IndexType > & coo_target )
{
   const GlobalIndex raw_entry_base =
      RawCOOOffdiagBlockOffset(
         coo_target,
         element_index,
         face_info );
   const GlobalIndex neighbor_element_index =
      face_info.PlusSide().GetCellIndex();

   AddRawCOOBlockEntries(
      kernel_context,
      trial_fe_space,
      test_fe_space,
      element_index,
      neighbor_element_index,
      trial_dof,
      y,
      raw_entry_base,
      coo_target );
}

} // namespace gendil
