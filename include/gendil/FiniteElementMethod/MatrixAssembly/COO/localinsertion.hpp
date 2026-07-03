// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/FiniteElementMethod/Restrictions/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoolayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

namespace gendil {

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
void AddRawCOOBlockEntries(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   const GlobalIndex raw_entry_base,
   RawCOOTripletBuffer< ValueType, IndexType > & coo_buffer )
{
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;
   using TrialDescriptor = std::remove_cvref_t< TrialDofDescriptor >;

   constexpr GlobalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr GlobalIndex ntest = LocalDofCount< TestShapeFunctions >();
   constexpr GlobalIndex block_entry_count = ntest * ntrial;

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
   const GlobalIndex global_col =
      GlobalDofIndex(
         trial_fe_space,
         typename TrialDescriptor::component{},
         col_element_index,
         trial_dof.indices );

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
         const GlobalIndex global_row =
            GlobalDofIndex(
               test_fe_space,
               typename TestDescriptor::component{},
               row_element_index,
               test_dof.indices );
         const GlobalIndex raw_index =
            raw_entry_base +
            local_col * ntest +
            local_row;

         GENDIL_VERIFY(
            raw_index < static_cast< GlobalIndex >( coo_buffer.nnz_raw ),
            "Raw COO emission wrote past the allocated triplet buffer." );

         coo_buffer.rows[raw_index] = static_cast< IndexType >( global_row );
         coo_buffer.cols[raw_index] = static_cast< IndexType >( global_col );
         coo_buffer.values[raw_index] += static_cast< ValueType >( value );
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
      coo_target.buffer );
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
      coo_target.buffer );
}

} // namespace gendil
