// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/doflayout.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/localdofloop.hpp"

#include <limits>

namespace gendil {

inline constexpr GlobalIndex RawCOOInactiveOffset =
   std::numeric_limits< GlobalIndex >::max();

GENDIL_HOST_DEVICE
inline bool IsActiveRawCOOOffset(
   const GlobalIndex offset,
   const GlobalIndex block_entry_count,
   const GlobalIndex nnz_raw )
{
   return offset != RawCOOInactiveOffset &&
      block_entry_count <= nnz_raw &&
      offset <= nnz_raw - block_entry_count;
}

struct RawCOOAssemblyLayout
{
   GlobalIndex num_elements = 0;
   GlobalIndex num_faces = 0;
   GlobalIndex block_entry_count = 0;
   GlobalIndex nnz_raw = 0;

   // Compact algebraic block bases. Cell, boundary self, and interior self
   // terms share diagonal_offsets[e]. Directed interior neighbor terms use
   // offdiag_offsets[e * num_faces + local_face].
   HostDevicePointer< GlobalIndex > diagonal_offsets;
   HostDevicePointer< GlobalIndex > offdiag_offsets;
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct RawCOOAssemblyTarget
{
   RawCOOTripletBuffer< ValueType, IndexType > buffer;
   RawCOOAssemblyLayout layout;
};

template < typename FaceInfo >
GENDIL_HOST_DEVICE
GlobalIndex RawCOOLocalFaceIndex( const FaceInfo & face_info )
{
   using LocalFaceIndex =
      std::remove_cvref_t< decltype(face_info.MinusSide().local_face_index) >;
   static_assert(
      requires { LocalFaceIndex::value; },
      "RawCOO face assembly requires compile-time local face indices." );
   return static_cast< GlobalIndex >( LocalFaceIndex::value );
}

GENDIL_HOST_DEVICE
inline GlobalIndex RawCOOFaceOffsetArrayIndex(
   const RawCOOAssemblyLayout & layout,
   const GlobalIndex element_index,
   const GlobalIndex local_face_index )
{
   GENDIL_VERIFY(
      local_face_index < layout.num_faces,
      "RawCOO face offset local face index is out of range." );
   GENDIL_VERIFY(
      layout.num_faces == 0 ||
         element_index <=
            std::numeric_limits< GlobalIndex >::max() / layout.num_faces,
      "RawCOO face offset array index overflow." );

   const GlobalIndex element_base = element_index * layout.num_faces;

   GENDIL_VERIFY(
      local_face_index <=
         std::numeric_limits< GlobalIndex >::max() - element_base,
      "RawCOO face offset array index overflow." );

   return element_base + local_face_index;
}

template < typename ValueType, typename IndexType >
GENDIL_HOST_DEVICE
GlobalIndex RawCOODiagonalBlockOffset(
   const RawCOOAssemblyTarget< ValueType, IndexType > & coo_target,
   const GlobalIndex element_index )
{
   GENDIL_VERIFY(
      element_index < coo_target.layout.num_elements,
      "RawCOO diagonal offset element index is out of range." );
   return coo_target.layout.diagonal_offsets[element_index];
}

template < typename ValueType, typename IndexType, typename FaceInfo >
GENDIL_HOST_DEVICE
GlobalIndex RawCOOOffdiagBlockOffset(
   const RawCOOAssemblyTarget< ValueType, IndexType > & coo_target,
   const GlobalIndex element_index,
   const FaceInfo & face_info )
{
   const GlobalIndex offset_index =
      RawCOOFaceOffsetArrayIndex(
         coo_target.layout,
         element_index,
         RawCOOLocalFaceIndex( face_info ) );
   return coo_target.layout.offdiag_offsets[offset_index];
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

   constexpr LocalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr LocalIndex ntest = LocalDofCount< TestShapeFunctions >();
   constexpr GlobalIndex block_entry_count =
      static_cast< GlobalIndex >( ntest ) *
      static_cast< GlobalIndex >( ntrial );

   GENDIL_VERIFY(
      IsActiveRawCOOOffset(
         raw_entry_base,
         block_entry_count,
         static_cast< GlobalIndex >( coo_buffer.nnz_raw ) ),
      "Raw COO emission received an inactive or out-of-range block offset." );

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
            raw_entry_base +
            static_cast< GlobalIndex >( local_col ) *
               static_cast< GlobalIndex >( ntest ) +
            static_cast< GlobalIndex >( local_row );

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
   RawCOOTripletBuffer< ValueType, IndexType > & coo_buffer )
{
   using TrialShapeFunctions =
      typename std::remove_cvref_t< TrialFESpace >::finite_element_type::shape_functions;
   using TestShapeFunctions =
      typename std::remove_cvref_t< TestFESpace >::finite_element_type::shape_functions;

   constexpr LocalIndex ntrial = LocalDofCount< TrialShapeFunctions >();
   constexpr LocalIndex ntest = LocalDofCount< TestShapeFunctions >();
   constexpr GlobalIndex block_entry_count =
      static_cast< GlobalIndex >( ntest ) *
      static_cast< GlobalIndex >( ntrial );

   GENDIL_VERIFY(
      block_entry_count == 0 ||
         element_index <=
            std::numeric_limits< GlobalIndex >::max() / block_entry_count,
      "Raw COO cell-local offset overflow." );

   const GlobalIndex raw_entry_base = element_index * block_entry_count;

   AddRawCOOBlockEntries(
      kernel_context,
      trial_fe_space,
      test_fe_space,
      element_index,
      element_index,
      trial_dof,
      y,
      raw_entry_base,
      coo_buffer );
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
