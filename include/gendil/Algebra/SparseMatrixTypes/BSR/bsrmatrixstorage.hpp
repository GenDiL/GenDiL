// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrbackendconfiguration.hpp"
#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/prelude.hpp"
#include "gendil/Utilities/checkedarithmetic.hpp"
#include "gendil/Utilities/MemoryManagement/synchostdevicearray.hpp"

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil
{

enum class BlockLayout
{
   RowMajor,
   ColumnMajor
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct HostBSRBackend : HostMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct NativeDeviceBSRBackend : DeviceMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

#if defined(GENDIL_USE_DEVICE)
#  if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
      defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
using DefaultBSRBackend = VendorDeviceBSRBackend<>;
#  else
using DefaultBSRBackend = NativeDeviceBSRBackend<>;
#  endif
#else
using DefaultBSRBackend = HostBSRBackend<>;
#endif

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend >
struct BSRMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   static constexpr BlockLayout block_layout = Layout;

   IndexType block_rows = 0;
   IndexType block_cols = 0;
   IndexType num_row_blocks = 0;
   IndexType num_col_blocks = 0;
   IndexType num_blocks = 0;

   SyncHostDeviceArray< ValueType, IndexType > values{};
   SyncHostDeviceArray< IndexType, IndexType > row_offsets{};
   SyncHostDeviceArray< IndexType, IndexType > col_indices{};

   // Keep the backend last so cached descriptors are destroyed before arrays.
   Backend backend{};

   BSRMatrix() = default;
   BSRMatrix( const BSRMatrix & ) = delete;
   BSRMatrix & operator=( const BSRMatrix & ) = delete;

   BSRMatrix( BSRMatrix && other )
      noexcept( std::is_nothrow_move_constructible_v< Backend > )
   : block_rows( std::exchange( other.block_rows, IndexType( 0 ) ) ),
     block_cols( std::exchange( other.block_cols, IndexType( 0 ) ) ),
     num_row_blocks(
        std::exchange( other.num_row_blocks, IndexType( 0 ) ) ),
     num_col_blocks(
        std::exchange( other.num_col_blocks, IndexType( 0 ) ) ),
     num_blocks( std::exchange( other.num_blocks, IndexType( 0 ) ) ),
     values( std::move( other.values ) ),
     row_offsets( std::move( other.row_offsets ) ),
     col_indices( std::move( other.col_indices ) ),
     backend( std::move( other.backend ) )
   { }

   BSRMatrix & operator=( BSRMatrix && other )
      noexcept( std::is_nothrow_move_assignable_v< Backend > )
   {
      if ( this != &other )
      {
         ResetState( backend );

         block_rows = std::exchange( other.block_rows, IndexType( 0 ) );
         block_cols = std::exchange( other.block_cols, IndexType( 0 ) );
         num_row_blocks =
            std::exchange( other.num_row_blocks, IndexType( 0 ) );
         num_col_blocks =
            std::exchange( other.num_col_blocks, IndexType( 0 ) );
         num_blocks = std::exchange( other.num_blocks, IndexType( 0 ) );
         values = std::move( other.values );
         row_offsets = std::move( other.row_offsets );
         col_indices = std::move( other.col_indices );
         backend = std::move( other.backend );
      }
      return *this;
   }

   ~BSRMatrix() = default;

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const
      requires requires
      {
         { Apply( backend, *this, x, y ) } -> std::same_as< void >;
      }
   {
      Apply( backend, *this, x, y );
   }
};

namespace details
{

template < typename IndexType >
std::size_t CheckedBSRScalarExtent(
   const IndexType block_count,
   const IndexType block_size,
   const char * message )
{
   if constexpr ( std::is_signed_v< IndexType > )
   {
      GENDIL_VERIFY(
         block_count >= 0 && block_size >= 0,
         message );
   }
   const auto count = static_cast< GlobalIndex >( block_count );
   const auto size = static_cast< GlobalIndex >( block_size );
   const GlobalIndex extent = CheckedMultiply( count, size, message );
   GENDIL_VERIFY(
      extent <= static_cast< GlobalIndex >(
         std::numeric_limits< std::size_t >::max() ),
      message );
   return static_cast< std::size_t >( extent );
}

} // namespace details

/** Return the scalar input-vector extent of a block sparse matrix. */
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
std::size_t GetInputSize(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return details::CheckedBSRScalarExtent(
      matrix.num_col_blocks,
      matrix.block_cols,
      "BSRMatrix input extent is invalid or overflows size_t." );
}

/** Return the scalar output-vector extent of a block sparse matrix. */
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
std::size_t GetOutputSize(
   const BSRMatrix< ValueType, IndexType, Layout, Backend > & matrix )
{
   return details::CheckedBSRScalarExtent(
      matrix.num_row_blocks,
      matrix.block_rows,
      "BSRMatrix output extent is invalid or overflows size_t." );
}

} // namespace gendil
