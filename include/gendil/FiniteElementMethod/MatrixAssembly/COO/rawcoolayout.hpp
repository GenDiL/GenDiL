// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcootripletbuffer.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"

#include <limits>
#include <type_traits>

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

inline GlobalIndex CheckedRawCOOAdd(
   const GlobalIndex lhs,
   const GlobalIndex rhs,
   const char * message )
{
   GENDIL_VERIFY(
      rhs <= std::numeric_limits< GlobalIndex >::max() - lhs,
      message );
   return lhs + rhs;
}

inline GlobalIndex CheckedRawCOOMultiply(
   const GlobalIndex lhs,
   const GlobalIndex rhs,
   const char * message )
{
   GENDIL_VERIFY(
      lhs == 0 || rhs <= std::numeric_limits< GlobalIndex >::max() / lhs,
      message );
   return lhs * rhs;
}

inline void AllocateRawCOOOffsetArray(
   const GlobalIndex count,
   HostDevicePointer< GlobalIndex > & offsets )
{
   AllocateHostPointer( count, offsets );
   AllocateDevicePointer( count, offsets );

   for ( GlobalIndex i = 0; i < count; ++i )
   {
      offsets[i] = RawCOOInactiveOffset;
   }

   if ( count > 0 )
   {
      ToDevice( count, offsets );
   }
}

inline void SyncRawCOOAssemblyLayoutToDevice(
   const RawCOOAssemblyLayout & layout )
{
   if ( layout.num_elements > 0 )
   {
      ToDevice( layout.num_elements, layout.diagonal_offsets );
   }

   const GlobalIndex face_offset_count =
      CheckedRawCOOMultiply(
         layout.num_elements,
         layout.num_faces,
         "RawCOO face offset array size overflow." );

   if ( face_offset_count > 0 )
   {
      ToDevice( face_offset_count, layout.offdiag_offsets );
   }
}

inline void FreeRawCOOAssemblyLayout( RawCOOAssemblyLayout & layout )
{
   FreeHostPointer( layout.diagonal_offsets );
   FreeDevicePointer( layout.diagonal_offsets );
   FreeHostPointer( layout.offdiag_offsets );
   FreeDevicePointer( layout.offdiag_offsets );
}

inline void ActivateRawCOODiagonalBlock(
   RawCOOAssemblyLayout & layout,
   const GlobalIndex element_index,
   const GlobalIndex block_entry_count,
   GlobalIndex & next_offset )
{
   GENDIL_VERIFY(
      element_index < layout.num_elements,
      "RawCOO diagonal activation element index is out of range." );

   if ( layout.diagonal_offsets[element_index] == RawCOOInactiveOffset )
   {
      layout.diagonal_offsets[element_index] = next_offset;
      next_offset =
         CheckedRawCOOAdd(
            next_offset,
            block_entry_count,
            "RawCOO diagonal block offset overflow." );
   }
}

template <
   bool IncludeCellTerms,
   bool IncludeBoundaryFaceTerms,
   bool IncludeInteriorFaceTerms,
   typename FESpace >
auto MakeRawCOOAssemblyLayout(
   const FESpace & fe_space,
   const GlobalIndex block_entry_count )
{
   using Space = std::remove_cvref_t< FESpace >;
   constexpr GlobalIndex num_faces =
      static_cast< GlobalIndex >(
         Space::finite_element_type::geometry::num_faces );

   RawCOOAssemblyLayout layout{};
   layout.num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   layout.num_faces = num_faces;
   layout.block_entry_count = block_entry_count;

   const GlobalIndex face_offset_count =
      CheckedRawCOOMultiply(
         layout.num_elements,
         layout.num_faces,
         "RawCOO face offset array size overflow." );

   AllocateRawCOOOffsetArray( layout.num_elements, layout.diagonal_offsets );
   AllocateRawCOOOffsetArray(
      face_offset_count,
      layout.offdiag_offsets );

   GlobalIndex next_offset = 0;

   for ( GlobalIndex element_index = 0;
         element_index < layout.num_elements;
         ++element_index )
   {
      if constexpr ( IncludeCellTerms )
      {
         ActivateRawCOODiagonalBlock(
            layout,
            element_index,
            block_entry_count,
            next_offset );
      }

      if constexpr ( IncludeInteriorFaceTerms )
      {
         InteriorFaceLoop(
            fe_space,
            element_index,
            [&] ( const auto & face_info )
            {
               using FaceInfo =
                  std::remove_cvref_t< decltype(face_info) >;
               static_assert(
                  FaceInfo::minus_side_type::is_conforming &&
                  FaceInfo::plus_side_type::is_conforming,
                  "RawCOO face assembly supports conforming faces only." );

               const GlobalIndex offset_index =
                  RawCOOFaceOffsetArrayIndex(
                     layout,
                     element_index,
                     RawCOOLocalFaceIndex( face_info ) );

               ActivateRawCOODiagonalBlock(
                  layout,
                  element_index,
                  block_entry_count,
                  next_offset );

               layout.offdiag_offsets[offset_index] =
                  next_offset;
               next_offset =
                  CheckedRawCOOAdd(
                     next_offset,
                     block_entry_count,
                     "RawCOO interior offdiag block offset overflow." );
            });
      }

      if constexpr ( IncludeBoundaryFaceTerms )
      {
         BoundaryFaceLoop(
            fe_space,
            element_index,
            [&] ( const auto & face_info )
            {
               using FaceInfo =
                  std::remove_cvref_t< decltype(face_info) >;
               static_assert(
                  FaceInfo::minus_side_type::is_conforming &&
                  FaceInfo::plus_side_type::is_conforming,
                  "RawCOO face assembly supports conforming faces only." );

               ActivateRawCOODiagonalBlock(
                  layout,
                  element_index,
                  block_entry_count,
                  next_offset );
            });
      }
   }

   layout.nnz_raw = next_offset;
   SyncRawCOOAssemblyLayoutToDevice( layout );

   return layout;
}

} // namespace gendil
