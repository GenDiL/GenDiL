// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <vector>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-bsr-device-apply skipped because GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

template < typename Matrix >
void SetBlockEntry(
   Matrix & matrix,
   const GlobalIndex block,
   const GlobalIndex local_row,
   const GlobalIndex local_col,
   const Real value )
{
   const GlobalIndex block_offset =
      block *
      static_cast< GlobalIndex >( matrix.block_rows ) *
      static_cast< GlobalIndex >( matrix.block_cols );

   if constexpr ( Matrix::block_layout == BlockLayout::ColumnMajor )
   {
      matrix.values[
         block_offset +
         local_col * static_cast< GlobalIndex >( matrix.block_rows ) +
         local_row ] = value;
   }
   else
   {
      matrix.values[
         block_offset +
         local_row * static_cast< GlobalIndex >( matrix.block_cols ) +
         local_col ] = value;
   }
}

template < BlockLayout Layout >
auto MakeManualBsrMatrix(
   const GlobalIndex block_rows,
   const GlobalIndex block_cols,
   const GlobalIndex num_row_blocks,
   const GlobalIndex num_col_blocks,
   const std::vector< GlobalIndex > & row_offsets,
   const std::vector< GlobalIndex > & col_indices )
{
   BSRMatrix< Real, GlobalIndex, Layout > matrix{};
   matrix.block_rows = block_rows;
   matrix.block_cols = block_cols;
   matrix.num_row_blocks = num_row_blocks;
   matrix.num_col_blocks = num_col_blocks;
   matrix.num_blocks = static_cast< GlobalIndex >( col_indices.size() );

   AllocateHostPointer( matrix.num_row_blocks + 1, matrix.row_offsets );
   AllocateDevicePointer( matrix.num_row_blocks + 1, matrix.row_offsets );
   AllocateHostPointer( matrix.num_blocks, matrix.col_indices );
   AllocateDevicePointer( matrix.num_blocks, matrix.col_indices );
   AllocateHostPointer(
      matrix.num_blocks * matrix.block_rows * matrix.block_cols,
      matrix.values );
   AllocateDevicePointer(
      matrix.num_blocks * matrix.block_rows * matrix.block_cols,
      matrix.values );

   for ( GlobalIndex i = 0; i < matrix.num_row_blocks + 1; ++i )
   {
      matrix.row_offsets[ i ] = row_offsets[ i ];
   }

   for ( GlobalIndex i = 0; i < matrix.num_blocks; ++i )
   {
      matrix.col_indices[ i ] = col_indices[ i ];
   }

   for ( GlobalIndex block = 0; block < matrix.num_blocks; ++block )
   {
      for ( GlobalIndex local_row = 0;
            local_row < matrix.block_rows;
            ++local_row )
      {
         for ( GlobalIndex local_col = 0;
               local_col < matrix.block_cols;
               ++local_col )
         {
            const Real value =
               Real( 0.25 ) +
               Real( 10 * ( block + 1 ) ) +
               Real( 2 * local_row ) -
               Real( 3 * local_col );
            SetBlockEntry( matrix, block, local_row, local_col, value );
         }
      }
   }

   ToDevice( matrix.num_row_blocks + 1, matrix.row_offsets );
   ToDevice( matrix.num_blocks, matrix.col_indices );
   ToDevice(
      matrix.num_blocks * matrix.block_rows * matrix.block_cols,
      matrix.values );

   return matrix;
}

template < typename Matrix >
bool CompareDeviceApplyWithCpu(
   const Matrix & matrix,
   const char * failure_message )
{
   const GlobalIndex x_size =
      static_cast< GlobalIndex >( matrix.num_col_blocks ) *
      static_cast< GlobalIndex >( matrix.block_cols );
   const GlobalIndex y_size =
      static_cast< GlobalIndex >( matrix.num_row_blocks ) *
      static_cast< GlobalIndex >( matrix.block_rows );

   Vector x( static_cast< size_t >( x_size ) );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x_size; ++i )
   {
      x_data[ i ] = Real( 1.0 ) + Real( i ) * Real( 0.5 );
   }

   Vector y_cpu( static_cast< size_t >( y_size ) );
   Vector y_gpu( static_cast< size_t >( y_size ) );
   y_cpu = 0.0;
   y_gpu = 0.0;

   matrix( x, y_cpu );
   NativeDeviceBSRBackend device_backend{};
   Apply( device_backend, matrix, x, y_gpu );

   const Real * cpu_data = y_cpu.ReadHostData();
   const Real * gpu_data = y_gpu.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < y_size; ++i )
   {
      success = Check(
         Near( cpu_data[ i ], gpu_data[ i ] ),
         failure_message ) && success;
   }

   return success;
}

bool TestSquareBsrDeviceApply()
{
   const auto matrix =
      MakeManualBsrMatrix< BlockLayout::ColumnMajor >(
         2,
         2,
         2,
         3,
         { 0, 2, 4 },
         { 0, 2, 1, 2 } );

   return CompareDeviceApplyWithCpu(
      matrix,
      "Square BSR device apply disagrees with CPU BSR apply." );
}

bool TestRectangularBsrDeviceApply()
{
   const auto matrix =
      MakeManualBsrMatrix< BlockLayout::RowMajor >(
         2,
         3,
         3,
         2,
         { 0, 1, 3, 4 },
         { 1, 0, 1, 0 } );

   return CompareDeviceApplyWithCpu(
      matrix,
      "Rectangular BSR device apply disagrees with CPU BSR apply." );
}

} // namespace

int main()
{
   bool success = true;
   success = TestSquareBsrDeviceApply() && success;
   success = TestRectangularBsrDeviceApply() && success;

   return success ? 0 : 1;
}

#endif
