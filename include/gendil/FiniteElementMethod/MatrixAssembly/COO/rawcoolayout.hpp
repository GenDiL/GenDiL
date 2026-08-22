// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcooview.hpp"
#include "gendil/FiniteElementMethod/MatrixFreeOperators/KernelOperators/LoopHelpers/faceloop.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"

#include <limits>
#include <type_traits>
#include <utility>

namespace gendil
{

/**
 * Move-only owner of the offsets used by RawCOO finite-element assembly.
 */
struct RawCOOAssemblyLayout
{
   using offset_type = GlobalIndex;

   offset_type num_elements = 0;
   offset_type num_faces = 0;
   offset_type block_entry_count = 0;
   offset_type nnz_raw = 0;

   // Compact algebraic block bases. Cell, boundary self, and interior self
   // terms share diagonal_offsets[e]. Directed interior neighbor terms use
   // offdiag_offsets[e * num_faces + local_face].
   SyncHostDeviceArray< GlobalIndex, GlobalIndex > diagonal_offsets{};
   SyncHostDeviceArray< GlobalIndex, GlobalIndex > offdiag_offsets{};

   RawCOOAssemblyLayout() = default;
   RawCOOAssemblyLayout( const RawCOOAssemblyLayout & ) = delete;
   RawCOOAssemblyLayout & operator=( const RawCOOAssemblyLayout & ) = delete;

   RawCOOAssemblyLayout( RawCOOAssemblyLayout && other ) noexcept
   : num_elements( std::exchange( other.num_elements, GlobalIndex( 0 ) ) ),
     num_faces( std::exchange( other.num_faces, GlobalIndex( 0 ) ) ),
     block_entry_count(
        std::exchange( other.block_entry_count, GlobalIndex( 0 ) ) ),
     nnz_raw( std::exchange( other.nnz_raw, GlobalIndex( 0 ) ) ),
     diagonal_offsets( std::move( other.diagonal_offsets ) ),
     offdiag_offsets( std::move( other.offdiag_offsets ) )
   { }

   RawCOOAssemblyLayout & operator=( RawCOOAssemblyLayout && other ) noexcept
   {
      if ( this != &other )
      {
         num_elements =
            std::exchange( other.num_elements, GlobalIndex( 0 ) );
         num_faces = std::exchange( other.num_faces, GlobalIndex( 0 ) );
         block_entry_count =
            std::exchange( other.block_entry_count, GlobalIndex( 0 ) );
         nnz_raw = std::exchange( other.nnz_raw, GlobalIndex( 0 ) );
         diagonal_offsets = std::move( other.diagonal_offsets );
         offdiag_offsets = std::move( other.offdiag_offsets );
      }
      return *this;
   }

   ~RawCOOAssemblyLayout() = default;
};

/**
 * Borrowed view of the offsets that map finite-element contributions into a
 * RawCOO triplet buffer. It is device-copyable and must not outlive its layout.
 */
template < typename IndexType = const GlobalIndex >
struct RawCOOAssemblyLayoutView
{
   using index_type = std::remove_const_t< IndexType >;
   using offset_type = RawCOOAssemblyLayout::offset_type;

   offset_type num_elements = 0;
   offset_type num_faces = 0;
   offset_type block_entry_count = 0;
   offset_type nnz_raw = 0;
   IndexType * diagonal_offsets = nullptr;
   IndexType * offdiag_offsets = nullptr;
};

/**
 * Device-copyable assembly payload combining mutable triplets with read-only
 * assembly offsets. It owns nothing and must not outlive either source view.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex >
struct RawCOOAssemblyTarget
{
   using value_type = std::remove_const_t< ValueType >;
   using index_type = std::remove_const_t< IndexType >;
   using offset_type = RawCOOAssemblyLayout::offset_type;

   index_type num_rows = 0;
   index_type num_cols = 0;
   index_type nnz_raw = 0;
   IndexType * rows = nullptr;
   IndexType * cols = nullptr;
   ValueType * values = nullptr;

   offset_type num_elements = 0;
   offset_type num_faces = 0;
   offset_type block_entry_count = 0;
   const offset_type * diagonal_offsets = nullptr;
   const offset_type * offdiag_offsets = nullptr;
};

/// Combine prepared triplet and layout views into a borrowed assembly target.
template <
   typename ValueType,
   typename IndexType,
   typename LayoutIndexType >
auto MakeRawCOOAssemblyTarget(
   const RawCOOTripletView< ValueType, IndexType > triplets,
   const RawCOOAssemblyLayoutView< LayoutIndexType > layout )
{
   GENDIL_VERIFY(
      static_cast<RawCOOAssemblyLayout::offset_type>(triplets.nnz_raw) ==
         layout.nnz_raw,
      "RawCOO triplet buffer and assembly layout capacities disagree.");
   GENDIL_VERIFY(
      layout.nnz_raw == 0 ||
         layout.block_entry_count <= layout.nnz_raw,
      "RawCOO assembly layout block capacity exceeds its triplet capacity.");

   return RawCOOAssemblyTarget< ValueType, IndexType >{
      triplets.num_rows,
      triplets.num_cols,
      triplets.nnz_raw,
      triplets.rows,
      triplets.cols,
      triplets.values,
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.diagonal_offsets,
      layout.offdiag_offsets };
}

/// Return a host read view, synchronizing both layout arrays to host as needed.
inline auto GetHostReadView( const RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< const GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      ReadHost( layout.diagonal_offsets ),
      ReadHost( layout.offdiag_offsets ) };
}

/// Return a host read-write layout view and invalidate the device copies.
inline auto GetHostReadWriteView( RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      ReadWriteHost( layout.diagonal_offsets ),
      ReadWriteHost( layout.offdiag_offsets ) };
}

/// Return a host write layout view without preserving previous offsets.
inline auto GetHostWriteView( RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      WriteHost( layout.diagonal_offsets ),
      WriteHost( layout.offdiag_offsets ) };
}

/// Return a device read view, synchronizing both layout arrays as needed.
inline auto GetDeviceReadView( const RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< const GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      ReadDevice( layout.diagonal_offsets ),
      ReadDevice( layout.offdiag_offsets ) };
}

/// Return a device read-write layout view and invalidate the host copies.
inline auto GetDeviceReadWriteView( RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      ReadWriteDevice( layout.diagonal_offsets ),
      ReadWriteDevice( layout.offdiag_offsets ) };
}

/// Return a device write layout view without preserving previous offsets.
inline auto GetDeviceWriteView( RawCOOAssemblyLayout & layout )
{
   return RawCOOAssemblyLayoutView< GlobalIndex >{
      layout.num_elements,
      layout.num_faces,
      layout.block_entry_count,
      layout.nnz_raw,
      WriteDevice( layout.diagonal_offsets ),
      WriteDevice( layout.offdiag_offsets ) };
}

/// Return a layout read view, synchronizing offsets to the selected space.
template < bool OnDevice >
auto GetKernelReadView( const RawCOOAssemblyLayout & layout )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadView( layout );
   }
   else
   {
      return GetHostReadView( layout );
   }
}

/// Return a layout read-write view and invalidate opposite-space offsets.
template < bool OnDevice >
auto GetKernelReadWriteView( RawCOOAssemblyLayout & layout )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceReadWriteView( layout );
   }
   else
   {
      return GetHostReadWriteView( layout );
   }
}

/// Return a layout write view without syncing and invalidate opposite offsets.
template < bool OnDevice >
auto GetKernelWriteView( RawCOOAssemblyLayout & layout )
{
   if constexpr ( OnDevice )
   {
      return GetDeviceWriteView( layout );
   }
   else
   {
      return GetHostWriteView( layout );
   }
}

/// Build a RawCOO target in the memory space selected by OnDevice.
template < bool OnDevice, typename ValueType, typename IndexType >
auto MakeRawCOOAssemblyTarget(
   RawCOOTripletBuffer< ValueType, IndexType > & triplets,
   const RawCOOAssemblyLayout & layout )
{
   return MakeRawCOOAssemblyTarget(
      GetKernelReadWriteView< OnDevice >( triplets ),
      GetKernelReadView< OnDevice >( layout ) );
}

/// Synchronize every initialized RawCOO layout array between host and device.
inline void Sync( const RawCOOAssemblyLayout & layout )
{
   Sync( layout.diagonal_offsets );
   Sync( layout.offdiag_offsets );
}

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

template < typename Layout >
GENDIL_HOST_DEVICE
inline GlobalIndex RawCOOFaceOffsetArrayIndex(
   const Layout & layout,
   const GlobalIndex element_index,
   const GlobalIndex local_face_index )
{
   GENDIL_VERIFY(
      local_face_index < layout.num_faces,
      "RawCOO face offset local face index is out of range." );
   const GlobalIndex element_base = CheckedMultiply(
      element_index,
      layout.num_faces,
      "RawCOO face offset array index overflow." );
   return CheckedAdd(
      element_base,
      local_face_index,
      "RawCOO face offset array index overflow." );
}

template < typename ValueType, typename IndexType >
GENDIL_HOST_DEVICE
GlobalIndex RawCOODiagonalBlockOffset(
   const RawCOOAssemblyTarget< ValueType, IndexType > & coo_target,
   const GlobalIndex element_index )
{
   GENDIL_VERIFY(
      element_index < coo_target.num_elements,
      "RawCOO diagonal offset element index is out of range." );
   return coo_target.diagonal_offsets[element_index];
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
         coo_target,
         element_index,
         RawCOOLocalFaceIndex( face_info ) );
   return coo_target.offdiag_offsets[offset_index];
}

inline auto AllocateRawCOOOffsetArray( const GlobalIndex count )
{
   return MakeSyncHostDeviceArray< GlobalIndex >( count );
}

inline void ActivateRawCOODiagonalBlock(
   const RawCOOAssemblyLayout & layout,
   GlobalIndex * diagonal_offsets,
   const GlobalIndex element_index,
   const GlobalIndex block_entry_count,
   GlobalIndex & next_offset )
{
   GENDIL_VERIFY(
      element_index < layout.num_elements,
      "RawCOO diagonal activation element index is out of range." );

   if ( diagonal_offsets[element_index] == RawCOOInactiveOffset )
   {
      diagonal_offsets[element_index] = next_offset;
      next_offset =
         CheckedAdd(
            next_offset,
            block_entry_count,
            "RawCOO diagonal block offset overflow." );
   }
}

template <
   bool IncludeCellTerms,
   bool IncludeBoundaryFaceTerms,
   bool IncludeInteriorFaceTerms,
   typename DomainMesh >
auto MakeRawCOOAssemblyLayout(
   const DomainMesh & domain_mesh,
   const GlobalIndex block_entry_count )
{
   using Geometry = mesh::mesh_geometry_t<DomainMesh>;
   constexpr GlobalIndex num_faces =
      static_cast< GlobalIndex >(
         Geometry::num_faces );

   RawCOOAssemblyLayout layout{};
   layout.num_elements =
      static_cast< GlobalIndex >( domain_mesh.GetNumberOfCells() );
   layout.num_faces = num_faces;
   layout.block_entry_count = block_entry_count;

   const GlobalIndex face_offset_count =
      CheckedMultiply(
         layout.num_elements,
         layout.num_faces,
         "RawCOO face offset array size overflow." );

   layout.diagonal_offsets =
      AllocateRawCOOOffsetArray( layout.num_elements );
   layout.offdiag_offsets =
      AllocateRawCOOOffsetArray( face_offset_count );

   auto * diagonal_offsets = WriteHost( layout.diagonal_offsets );
   auto * offdiag_offsets = WriteHost( layout.offdiag_offsets );
   for ( GlobalIndex i = 0; i < layout.num_elements; ++i )
   {
      diagonal_offsets[i] = RawCOOInactiveOffset;
   }
   for ( GlobalIndex i = 0; i < face_offset_count; ++i )
   {
      offdiag_offsets[i] = RawCOOInactiveOffset;
   }

   GlobalIndex next_offset = 0;

   for ( GlobalIndex element_index = 0;
         element_index < layout.num_elements;
         ++element_index )
   {
      if constexpr ( IncludeCellTerms )
      {
         ActivateRawCOODiagonalBlock(
            layout,
            diagonal_offsets,
            element_index,
            block_entry_count,
            next_offset );
      }

      if constexpr ( IncludeInteriorFaceTerms )
      {
         InteriorFaceLoop(
            domain_mesh,
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
                  diagonal_offsets,
                  element_index,
                  block_entry_count,
                  next_offset );

               offdiag_offsets[offset_index] = next_offset;
               next_offset =
                  CheckedAdd(
                     next_offset,
                     block_entry_count,
                     "RawCOO interior offdiag block offset overflow." );
            });
      }

      if constexpr ( IncludeBoundaryFaceTerms )
      {
         BoundaryFaceLoop(
            domain_mesh,
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
                  diagonal_offsets,
                  element_index,
                  block_entry_count,
                  next_offset );
            });
      }
   }

   layout.nnz_raw = next_offset;
   Sync( layout );

   return layout;
}

template <
   bool IncludeCellTerms,
   bool IncludeBoundaryFaceTerms,
   bool IncludeInteriorFaceTerms,
   typename DomainMesh >
auto MakeRawCOOElementBlockDiagonalAssemblyLayout(
   const DomainMesh & domain_mesh,
   const GlobalIndex block_entry_count )
{
   RawCOOAssemblyLayout layout{};
   layout.num_elements =
      static_cast<GlobalIndex>(
         domain_mesh.GetNumberOfCells());
   layout.num_faces = 0;
   layout.block_entry_count = block_entry_count;

   layout.diagonal_offsets =
      AllocateRawCOOOffsetArray( layout.num_elements );
   layout.offdiag_offsets = AllocateRawCOOOffsetArray( 0 );

   auto * diagonal_offsets = WriteHost( layout.diagonal_offsets );
   WriteHost( layout.offdiag_offsets );
   for ( GlobalIndex i = 0; i < layout.num_elements; ++i )
   {
      diagonal_offsets[i] = RawCOOInactiveOffset;
   }

   GlobalIndex next_offset = 0;

   for (GlobalIndex element_index = 0;
        element_index < layout.num_elements;
        ++element_index)
   {
      if constexpr (IncludeCellTerms)
      {
         ActivateRawCOODiagonalBlock(
            layout,
            diagonal_offsets,
            element_index,
            block_entry_count,
            next_offset);
      }

      if constexpr (IncludeInteriorFaceTerms)
      {
         InteriorFaceLoop(
            domain_mesh,
            element_index,
            [&] (const auto & face_info)
            {
               using FaceInfo =
                  std::remove_cvref_t<decltype(face_info)>;
               static_assert(
                  FaceInfo::minus_side_type::is_conforming &&
                  FaceInfo::plus_side_type::is_conforming,
                  "RawCOO face assembly supports conforming faces only.");

               ActivateRawCOODiagonalBlock(
                  layout,
                  diagonal_offsets,
                  element_index,
                  block_entry_count,
                  next_offset);
            });
      }

      if constexpr (IncludeBoundaryFaceTerms)
      {
         BoundaryFaceLoop(
            domain_mesh,
            element_index,
            [&] (const auto & face_info)
            {
               using FaceInfo =
                  std::remove_cvref_t<decltype(face_info)>;
               static_assert(
                  FaceInfo::minus_side_type::is_conforming &&
                  FaceInfo::plus_side_type::is_conforming,
                  "RawCOO face assembly supports conforming faces only.");

               ActivateRawCOODiagonalBlock(
                  layout,
                  diagonal_offsets,
                  element_index,
                  block_entry_count,
                  next_offset);
            });
      }
   }

   layout.nnz_raw = next_offset;
   Sync( layout );

   return layout;
}

} // namespace gendil
