// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

enum class BlockLayout
{
   RowMajor,
   ColumnMajor
};

struct HostBSRBackend
{ };
struct NativeDeviceBSRBackend
{ };

using DefaultBSRBackend = HostBSRBackend;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   BlockLayout Layout = BlockLayout::ColumnMajor,
   typename Backend = DefaultBSRBackend >
struct BSRMatrix;

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y );

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const NativeDeviceBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y );

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const NativeDeviceBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );
#endif

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
GENDIL_HOST_DEVICE
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y );

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename Backend >
struct BSRMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType block_rows = 0; // ntest
   IndexType block_cols = 0; // ntrial
   IndexType num_row_blocks = 0;
   IndexType num_col_blocks = 0;
   IndexType num_blocks = 0;

   static constexpr BlockLayout block_layout = Layout;

   HostDevicePointer<ValueType> values;       // size = num_blocks * block_rows * block_cols
   HostDevicePointer<IndexType> row_offsets;  // size = num_row_blocks + 1
   HostDevicePointer<IndexType> col_indices;  // size = num_blocks

   // BSRMatrix is currently a shallow handle over HostDevicePointer buffers.
   // Backend objects stored here must therefore be safely copyable as long as
   // BSRMatrix itself remains copyable.
   Backend backend{};

   GENDIL_HOST_DEVICE
   constexpr ValueType GetBlockEntry(
      const IndexType block_index,
      const IndexType local_row,
      const IndexType local_col ) const
   {
      const IndexType block_offset = block_index * block_rows * block_cols;

      if constexpr (block_layout == BlockLayout::ColumnMajor)
      {
         return values[block_offset + local_col * block_rows + local_row];
      }
      else
      {
         return values[block_offset + local_row * block_cols + local_col];
      }
   }

   template <typename InputVector, typename OutputVector>
   GENDIL_HOST_DEVICE
   void operator()( const InputVector & x, OutputVector & y ) const
   {
      Apply( backend, *this, x, y );
   }
};

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
GENDIL_HOST_DEVICE
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
   static_assert(
      dependent_false_v< InputVector >,
      "Apply(HostBSRBackend, BSRMatrix, x, y) supports only explicit "
      "host vector overloads. Add an overload that acquires the correct "
      "read/write memory space before entering the BSR loop." );
}

template <
   typename Matrix,
   typename InputVector,
   typename OutputVector >
void Apply(
   const NativeDeviceBSRBackend &,
   const Matrix & matrix,
   const InputVector & x,
   OutputVector & y )
{
   static_assert(
      dependent_false_v< Matrix >,
      "Apply(NativeDeviceBSRBackend, BSRMatrix, x, y) supports only "
      "explicit device vector overloads. Add an overload that acquires "
      "device pointers before launching the BSR kernel." );
}

namespace details
{

template < typename Matrix >
void ApplyHostBSRToRawPointers(
   const Matrix & matrix,
   const Real * x_data,
   Real * y_data )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;

   #pragma omp parallel for
   for (IndexType block_row = 0;
        block_row < matrix.num_row_blocks;
        ++block_row)
   {
      for (IndexType local_row = 0;
           local_row < matrix.block_rows;
           ++local_row)
      {
         ValueType sum = ValueType(0);

         for (IndexType block_it = matrix.row_offsets[block_row];
              block_it < matrix.row_offsets[block_row + 1];
              ++block_it)
         {
            const IndexType block_col = matrix.col_indices[block_it];

            for (IndexType local_col = 0;
                 local_col < matrix.block_cols;
                 ++local_col)
            {
               const IndexType global_col =
                  block_col * matrix.block_cols + local_col;

               sum +=
                  matrix.GetBlockEntry(block_it, local_row, local_col) *
                  x_data[global_col];
            }
         }

         const IndexType global_row =
            block_row * matrix.block_rows + local_row;
         y_data[global_row] = sum;
      }
   }
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y )
{
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   details::ApplyHostBSRToRawPointers( matrix, x_data, y_data );
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const HostBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   const Real * x_data = x.HostRead();
   Real * y_data = y.HostWrite();

   details::ApplyHostBSRToRawPointers( matrix, x_data, y_data );
}
#endif

namespace details
{

template < typename Matrix >
GlobalIndex CheckBSRApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      matrix.block_rows > 0,
      "Apply(BSR backend, ...) requires a positive row block size." );
   GENDIL_VERIFY(
      matrix.block_cols > 0,
      "Apply(BSR backend, ...) requires a positive column block size." );

   const GlobalIndex expected_x_size =
      static_cast< GlobalIndex >( matrix.num_col_blocks ) *
      static_cast< GlobalIndex >( matrix.block_cols );
   const GlobalIndex expected_y_size =
      static_cast< GlobalIndex >( matrix.num_row_blocks ) *
      static_cast< GlobalIndex >( matrix.block_rows );

   GENDIL_VERIFY(
      x_size == static_cast< size_t >( expected_x_size ),
      "Apply(BSR backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( expected_y_size ),
      "Apply(BSR backend, ...) output vector has the wrong size." );

   return expected_y_size;
}

#if defined(GENDIL_USE_DEVICE)
template < typename Matrix >
__global__
void BSRDeviceApplyKernel(
   const Matrix matrix,
   const Real * x,
   Real * y,
   const GlobalIndex total_rows )
{
   const GlobalIndex stride =
      static_cast< GlobalIndex >( blockDim.x ) *
      static_cast< GlobalIndex >( gridDim.x );
   const GlobalIndex block_rows =
      static_cast< GlobalIndex >( matrix.block_rows );
   const GlobalIndex block_cols =
      static_cast< GlobalIndex >( matrix.block_cols );

   for ( GlobalIndex global_row =
            static_cast< GlobalIndex >( blockIdx.x ) *
            static_cast< GlobalIndex >( blockDim.x ) +
            static_cast< GlobalIndex >( threadIdx.x );
         global_row < total_rows;
         global_row += stride )
   {
      const GlobalIndex block_row = global_row / block_rows;
      const GlobalIndex local_row = global_row % block_rows;

      Real sum = Real( 0 );

      for ( auto block_it = matrix.row_offsets[ block_row ];
            block_it < matrix.row_offsets[ block_row + 1 ];
            ++block_it )
      {
         const auto block_col = matrix.col_indices[ block_it ];

         for ( GlobalIndex local_col = 0;
               local_col < block_cols;
               ++local_col )
         {
            const GlobalIndex global_col =
               static_cast< GlobalIndex >( block_col ) * block_cols +
               local_col;

            sum +=
               matrix.GetBlockEntry(
                  block_it,
                  static_cast< decltype( matrix.block_rows ) >( local_row ),
                  static_cast< decltype( matrix.block_cols ) >( local_col ) ) *
               x[ global_col ];
         }
      }

      y[ global_row ] = sum;
   }
}

template < typename Matrix >
void ApplyDeviceBSRToRawPointers(
   const Matrix & matrix,
   const Real * x_data,
   Real * y_data,
   const GlobalIndex total_rows )
{
   if ( total_rows == 0 )
   {
      return;
   }

   constexpr unsigned int threads_per_block = 256;
   const GlobalIndex grid_x_size =
      ( total_rows + threads_per_block - 1 ) / threads_per_block;

   GENDIL_VERIFY(
      grid_x_size <=
         static_cast< GlobalIndex >( std::numeric_limits< unsigned int >::max() ),
      "Apply(NativeDeviceBSRBackend, ...) launch grid is too large." );

   const dim3 block_dim( threads_per_block );
   const dim3 grid_dim( static_cast< unsigned int >( grid_x_size ) );

   CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
   GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
      "Apply(NativeDeviceBSRBackend, ...): before launch" );
   BSRDeviceApplyKernel<<< grid_dim, block_dim >>>(
      matrix,
      x_data,
      y_data,
      total_rows );
   GENDIL_CHECK_LAST_DEVICE_LAUNCH(
      "Apply(NativeDeviceBSRBackend, ...)" );
}
#endif

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const NativeDeviceBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix,
         x.Size(),
         y.Size() );

   const Real * x_data = x.ReadDeviceData();
   Real * y_data = y.WriteDeviceData();

   details::ApplyDeviceBSRToRawPointers(
      matrix,
      x_data,
      y_data,
      expected_y_size );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "Apply(NativeDeviceBSRBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use BSRMatrix::operator() for CPU execution." );
#endif
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend >
void Apply(
   const NativeDeviceBSRBackend &,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         matrix,
         static_cast< size_t >( x.Size() ),
         static_cast< size_t >( y.Size() ) );

   const Real * x_data = x.DeviceRead();
   Real * y_data = y.DeviceWrite();

   details::ApplyDeviceBSRToRawPointers(
      matrix,
      x_data,
      y_data,
      expected_y_size );
#else
   static_assert(
      dependent_false_v< BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> >,
      "Apply(NativeDeviceBSRBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

} // namespace gendil
