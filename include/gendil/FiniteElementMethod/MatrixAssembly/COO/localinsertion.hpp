// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/globaldofindex.hpp"
#include "gendil/FiniteElementMethod/Restrictions/restrictionoperations.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/localrestrictionentrylayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/Generic/localdoforientation.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

#include <limits>

namespace gendil {

namespace details {

template < typename COOBuffer >
GENDIL_HOST_DEVICE
void AssertRawCOOCoordinateWriteIsConsistent(
   const COOBuffer & buffer,
   const GlobalIndex raw_index,
   const GlobalIndex algebraic_row,
   const GlobalIndex algebraic_col )
{
#ifndef NDEBUG
   using Buffer = std::remove_cvref_t< COOBuffer >;
   using IndexType = typename Buffer::index_type;
   const IndexType row = static_cast< IndexType >( algebraic_row );
   const IndexType col = static_cast< IndexType >( algebraic_col );
   constexpr IndexType unset = std::numeric_limits< IndexType >::max();
   const bool initialized =
      buffer.rows[raw_index] != unset || buffer.cols[raw_index] != unset;
   GENDIL_ASSERT(
      !initialized ||
         ( buffer.rows[raw_index] == row &&
           buffer.cols[raw_index] == col ),
      "Every RawCOO slot must retain one stable algebraic coordinate pair." );
#else
   (void)buffer;
   (void)raw_index;
   (void)algebraic_row;
   (void)algebraic_col;
#endif
}

} // namespace details

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
   using TrialDescriptor = std::remove_cvref_t< TrialDofDescriptor >;

   constexpr GlobalIndex test_entry_count =
      CountElementRestrictionEntries< TestFESpace >();
   const auto block_entry_count = coo_buffer.block_entry_count;

   GENDIL_VERIFY(
      IsActiveRawCOOOffset(
         raw_entry_base,
         block_entry_count,
         static_cast< GlobalIndex >( coo_buffer.nnz_raw ) ),
      "Raw COO emission received an inactive or out-of-range block offset." );

   const auto trial_component = typename TrialDescriptor::component{};
   const auto trial_local_dof = MakeRestrictionLocalDof(
      trial_fe_space,
      trial_component,
      trial_dof.indices );
   GlobalIndex trial_restriction_entry = 0;
   ForEachRestrictionEntry(
      GetRestriction( trial_fe_space ),
      col_element_index,
      trial_local_dof,
      [&] ( const GlobalIndex algebraic_col, const auto & trial_weight )
      {
         const GlobalIndex local_trial_entry =
            FlattenLocalRestrictionEntry(
               trial_fe_space,
               trial_component,
               trial_dof.indices,
               trial_restriction_entry++ );

         // Compact RawCOO slots may receive several weak-form or quadrature
         // contributions. Every visit resolves the same deterministic global
         // coordinate pair and accumulates its weighted value.
         ForEachLocalResidualDof(
            kernel_context,
            test_fe_space,
            y,
            [&] ( const auto & test_dof, const auto & value )
            {
               using TestDescriptor =
                  std::remove_cvref_t< decltype(test_dof) >;
               const auto test_component =
                  typename TestDescriptor::component{};
               const auto test_local_dof = MakeRestrictionLocalDof(
                  test_fe_space,
                  test_component,
                  test_dof.indices );
               GlobalIndex test_restriction_entry = 0;
               ForEachRestrictionEntry(
                  GetRestriction( test_fe_space ),
                  row_element_index,
                  test_local_dof,
                  [&] (
                     const GlobalIndex algebraic_row,
                     const auto & test_weight )
                  {
                     const GlobalIndex local_test_entry =
                        FlattenLocalRestrictionEntry(
                           test_fe_space,
                           test_component,
                           test_dof.indices,
                           test_restriction_entry++ );
                     const GlobalIndex raw_index =
                        raw_entry_base +
                        local_trial_entry * test_entry_count +
                        local_test_entry;

                     GENDIL_VERIFY(
                        raw_index < static_cast< GlobalIndex >( coo_buffer.nnz_raw ),
                        "Raw COO emission wrote past the allocated triplet buffer." );
                     GENDIL_ASSERT(
                        algebraic_row < static_cast< GlobalIndex >( coo_buffer.num_rows ),
                        "Raw COO row coordinate exceeds the test algebraic DoF extent." );
                     GENDIL_ASSERT(
                        algebraic_col < static_cast< GlobalIndex >( coo_buffer.num_cols ),
                        "Raw COO column coordinate exceeds the trial algebraic DoF extent." );

                     details::AssertRawCOOCoordinateWriteIsConsistent(
                        coo_buffer,
                        raw_index,
                        algebraic_row,
                        algebraic_col );
                     coo_buffer.rows[raw_index] = static_cast< IndexType >( algebraic_row );
                     coo_buffer.cols[raw_index] = static_cast< IndexType >( algebraic_col );
                     coo_buffer.values[raw_index] +=
                        static_cast< ValueType >(
                           ApplyAdjointRestrictionWeight(
                              test_weight,
                              ApplyRestrictionWeight(
                                 trial_weight,
                                 value ) ) );
                  } );
               GENDIL_ASSERT(
                  test_restriction_entry ==
                     CountLocalDofRestrictionEntries(
                        test_fe_space,
                        test_component,
                        test_dof.indices ),
                  "RawCOO restriction visitation must match its exact static test-row count." );
            } );
      } );
   GENDIL_ASSERT(
      trial_restriction_entry ==
         CountLocalDofRestrictionEntries(
            trial_fe_space,
            trial_component,
            trial_dof.indices ),
      "RawCOO restriction visitation must match its exact static trial-row count." );
}

/**
 * @brief Emit one reference-local face block through native oriented DoFs.
 *
 * Orientation is applied before restriction visitation. Local restriction
 * entries are then flattened in deterministic native-row order, so cardinality
 * lookup, visitation, global coordinates, and slot placement use one convention.
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
   using TrialDescriptor =
      std::remove_cvref_t<TrialDofDescriptor>;
   constexpr GlobalIndex test_entry_count =
      CountElementRestrictionEntries<TestFESpace>();

   const GlobalIndex raw_entry_base =
      RawCOOEntityBlockOffset(target, entity_index);
   GENDIL_VERIFY(
      IsActiveRawCOOOffset(
         raw_entry_base,
         target.block_entry_count,
         static_cast<GlobalIndex>(target.nnz_raw)),
      "RawCOO face emission received an out-of-range block offset." );

   const auto native_trial_dof = OrientReferenceDofToNative(
      trial_fe_space,
      reference_trial_dof,
      trial_orientation);
   const auto trial_component = typename TrialDescriptor::component{};
   const auto trial_local_dof = MakeRestrictionLocalDof(
      trial_fe_space,
      trial_component,
      native_trial_dof.indices );
   GlobalIndex trial_restriction_entry = 0;
   ForEachRestrictionEntry(
      GetRestriction( trial_fe_space ),
      col_element_index,
      trial_local_dof,
      [&] ( const GlobalIndex algebraic_col, const auto & trial_weight )
      {
         const GlobalIndex local_trial_entry =
            FlattenLocalRestrictionEntry(
               trial_fe_space,
               trial_component,
               native_trial_dof.indices,
               trial_restriction_entry++ );
         ForEachLocalResidualDof(
            kernel_context,
            test_fe_space,
            reference_residual,
            [&] (const auto& reference_test_dof, const auto& value)
            {
               using TestDescriptor =
                  std::remove_cvref_t<decltype(reference_test_dof)>;
               const auto native_test_dof = OrientReferenceDofToNative(
                  test_fe_space,
                  reference_test_dof,
                  test_orientation);
               const auto test_component =
                  typename TestDescriptor::component{};
               const auto test_local_dof = MakeRestrictionLocalDof(
                  test_fe_space,
                  test_component,
                  native_test_dof.indices );
               GlobalIndex test_restriction_entry = 0;
               ForEachRestrictionEntry(
                  GetRestriction( test_fe_space ),
                  row_element_index,
                  test_local_dof,
                  [&] (
                     const GlobalIndex algebraic_row,
                     const auto & test_weight )
                  {
                     const GlobalIndex local_test_entry =
                        FlattenLocalRestrictionEntry(
                           test_fe_space,
                           test_component,
                           native_test_dof.indices,
                           test_restriction_entry++ );
                     const GlobalIndex raw_index = raw_entry_base +
                        local_trial_entry * test_entry_count +
                        local_test_entry;

                     GENDIL_VERIFY(
                        raw_index < static_cast<GlobalIndex>(target.nnz_raw),
                        "RawCOO face emission wrote past its target segment." );
                     GENDIL_ASSERT(
                        algebraic_row < static_cast<GlobalIndex>(target.num_rows),
                        "RawCOO face row exceeds the test algebraic extent." );
                     GENDIL_ASSERT(
                        algebraic_col < static_cast<GlobalIndex>(target.num_cols),
                        "RawCOO face column exceeds the trial algebraic extent." );

                     details::AssertRawCOOCoordinateWriteIsConsistent(
                        target,
                        raw_index,
                        algebraic_row,
                        algebraic_col );
                     target.rows[raw_index] = static_cast<IndexType>(algebraic_row);
                     target.cols[raw_index] = static_cast<IndexType>(algebraic_col);
                     target.values[raw_index] +=
                        static_cast<ValueType>(
                           ApplyAdjointRestrictionWeight(
                              test_weight,
                              ApplyRestrictionWeight(
                                 trial_weight,
                                 value ) ) );
                  } );
               GENDIL_ASSERT(
                  test_restriction_entry ==
                     CountLocalDofRestrictionEntries(
                        test_fe_space,
                        test_component,
                        native_test_dof.indices ),
                  "RawCOO restriction visitation must match its exact static test-row count." );
            } );
      } );
   GENDIL_ASSERT(
      trial_restriction_entry ==
         CountLocalDofRestrictionEntries(
            trial_fe_space,
            trial_component,
            native_trial_dof.indices ),
      "RawCOO restriction visitation must match its exact static trial-row count." );
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
