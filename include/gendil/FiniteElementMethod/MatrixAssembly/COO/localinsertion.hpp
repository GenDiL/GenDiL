// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
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
void AddSparseMatrixEntry(
   const KernelContext & kernel_context,
   const TrialFESpace & trial_fe_space,
   const TestFESpace & test_fe_space,
   const GlobalIndex & row_element_index,
   const GlobalIndex & col_element_index,
   const TrialDofDescriptor & trial_dof,
   const ElementVector & y,
   RawCOOTripletBuffer< ValueType, IndexType > & coo_buffer )
{
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;
   using TrialDescriptor = std::remove_cvref_t< TrialDofDescriptor >;

   constexpr LocalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr LocalIndex ntest = LocalDofCount< TestShapeFunctions >();

   GENDIL_VERIFY(
      row_element_index == col_element_index,
      "Raw COO first implementation supports cell-local emission only." );

   const LocalIndex local_col =
      FlattenLocalDof(
         trial_fe_space,
         typename TrialDescriptor::component{},
         trial_dof.indices );
   const GlobalIndex global_col =
      ElementToGlobalDofIndex(
         trial_fe_space,
         typename TrialDescriptor::component{},
         col_element_index,
         trial_dof.indices );
   const GlobalIndex entity_base =
      row_element_index *
      static_cast< GlobalIndex >( ntest ) *
      static_cast< GlobalIndex >( ntrial );

   ForEachLocalResidualDof(
      kernel_context,
      test_fe_space,
      y,
      [&] ( const auto & test_dof, const auto & value )
      {
         using TestDescriptor = std::remove_cvref_t< decltype(test_dof) >;
         const LocalIndex local_row =
            FlattenLocalDof(
               test_fe_space,
               typename TestDescriptor::component{},
               test_dof.indices );
         const GlobalIndex global_row =
            ElementToGlobalDofIndex(
               test_fe_space,
               typename TestDescriptor::component{},
               row_element_index,
               test_dof.indices );
         const GlobalIndex raw_index =
            entity_base +
            static_cast< GlobalIndex >( local_col ) *
               static_cast< GlobalIndex >( ntest ) +
            static_cast< GlobalIndex >( local_row );

         GENDIL_VERIFY(
            raw_index < static_cast< GlobalIndex >( coo_buffer.nnz_raw ),
            "Raw COO emission wrote past the allocated triplet buffer." );

         coo_buffer.rows[raw_index] = static_cast< IndexType >( global_row );
         coo_buffer.cols[raw_index] = static_cast< IndexType >( global_col );
         coo_buffer.values[raw_index] = static_cast< ValueType >( value );
      });
}

} // namespace gendil
