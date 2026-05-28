// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

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

Real LogicalBlockEntry(
   const GlobalIndex block_index,
   const GlobalIndex local_row,
   const GlobalIndex local_col )
{
   return
      10.0 * static_cast< Real >( block_index + 1 ) +
      3.0 * static_cast< Real >( local_row + 1 ) +
      0.25 * static_cast< Real >( local_col + 1 );
}

template < typename Matrix >
void SetStoredBlockEntry(
   Matrix & matrix,
   const GlobalIndex block_index,
   const GlobalIndex local_row,
   const GlobalIndex local_col,
   const Real value )
{
   const GlobalIndex block_offset =
      block_index * matrix.block_rows * matrix.block_cols;

   if constexpr ( Matrix::block_layout == BlockLayout::ColumnMajor )
   {
      matrix.values[block_offset + local_col * matrix.block_rows + local_row] =
         value;
   }
   else
   {
      matrix.values[block_offset + local_row * matrix.block_cols + local_col] =
         value;
   }
}

template < typename Matrix >
void FillLogicalBlocks( Matrix & matrix )
{
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
            SetStoredBlockEntry(
               matrix,
               block,
               local_row,
               local_col,
               LogicalBlockEntry( block, local_row, local_col ) );
         }
      }
   }
}

template < typename Matrix >
bool CheckLogicalBlockEntries(
   const Matrix & matrix,
   const char * message )
{
   bool success = true;
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
            success = Check(
               Near(
                  matrix.GetBlockEntry( block, local_row, local_col ),
                  LogicalBlockEntry( block, local_row, local_col ) ),
               message ) && success;
         }
      }
   }
   return success;
}

bool TestRawApplySupportsBothBlockLayouts()
{
   auto column_matrix =
      MakeBlockDiagonalDGBSRPattern<
         Real,
         GlobalIndex,
         BlockLayout::ColumnMajor >( 3, 2, 3 );
   auto row_matrix =
      MakeBlockDiagonalDGBSRPattern<
         Real,
         GlobalIndex,
         BlockLayout::RowMajor >( 3, 2, 3 );

   FillLogicalBlocks( column_matrix );
   FillLogicalBlocks( row_matrix );

   bool success = true;
   success = CheckLogicalBlockEntries(
      column_matrix,
      "Column-major raw BSR GetBlockEntry returned the wrong logical entry." ) &&
      success;
   success = CheckLogicalBlockEntries(
      row_matrix,
      "Row-major raw BSR GetBlockEntry returned the wrong logical entry." ) &&
      success;

   Vector x( column_matrix.num_col_blocks * column_matrix.block_cols );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = -0.5 + 0.75 * static_cast< Real >( i );
   }

   Vector y_column( column_matrix.num_row_blocks * column_matrix.block_rows );
   Vector y_row( row_matrix.num_row_blocks * row_matrix.block_rows );
   y_column = 0.0;
   y_row = 0.0;
   column_matrix( x, y_column );
   row_matrix( x, y_row );

   const Real * y_column_data = y_column.ReadHostData();
   const Real * y_row_data = y_row.ReadHostData();
   for ( GlobalIndex block_row = 0;
         block_row < column_matrix.num_row_blocks;
         ++block_row )
   {
      for ( GlobalIndex local_row = 0;
            local_row < column_matrix.block_rows;
            ++local_row )
      {
         Real expected = 0.0;
         for ( GlobalIndex block_it = column_matrix.row_offsets[block_row];
               block_it < column_matrix.row_offsets[block_row + 1];
               ++block_it )
         {
            const GlobalIndex block_col = column_matrix.col_indices[block_it];
            for ( GlobalIndex local_col = 0;
                  local_col < column_matrix.block_cols;
                  ++local_col )
            {
               const GlobalIndex global_col =
                  block_col * column_matrix.block_cols + local_col;
               expected +=
                  LogicalBlockEntry( block_it, local_row, local_col ) *
                  x_data[global_col];
            }
         }

         const GlobalIndex global_row =
            block_row * column_matrix.block_rows + local_row;
         success = Check(
            Near( y_column_data[global_row], expected ),
            "Column-major raw BSR apply produced the wrong value." ) && success;
         success = Check(
            Near( y_row_data[global_row], expected ),
            "Row-major raw BSR apply produced the wrong value." ) && success;
      }
   }

   return success;
}

bool TestAssemblySupportsBothBlockLayouts()
{
   const Integer n = 4;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 2;
   FiniteElementOrders< order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto rho =
      MakeCoefficient< "density", PhysicalCoordinate >(
         [] ( const auto & x_phys )
         {
            return 1.0 + x_phys[0] * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   auto column_matrix =
      MakeDGBSRPattern< Real, GlobalIndex, BlockLayout::ColumnMajor >(
         fe_space );
   auto row_matrix =
      MakeDGBSRPattern< Real, GlobalIndex, BlockLayout::RowMajor >(
         fe_space );

   using KernelPolicy = SerialKernelConfiguration;
   GenericAssembly< KernelPolicy >(
      weak_form,
      wf_context,
      integration_rule,
      column_matrix );
   GenericAssembly< KernelPolicy >(
      weak_form,
      wf_context,
      integration_rule,
      row_matrix );

   bool success = true;
   success = Check(
      column_matrix.num_blocks == row_matrix.num_blocks,
      "Row-major and column-major BSR patterns have different block counts." ) &&
      success;
   success = Check(
      column_matrix.block_rows == row_matrix.block_rows &&
      column_matrix.block_cols == row_matrix.block_cols,
      "Row-major and column-major BSR patterns have different block sizes." ) &&
      success;

   for ( GlobalIndex block = 0; block < column_matrix.num_blocks; ++block )
   {
      for ( GlobalIndex local_row = 0;
            local_row < column_matrix.block_rows;
            ++local_row )
      {
         for ( GlobalIndex local_col = 0;
               local_col < column_matrix.block_cols;
               ++local_col )
         {
            success = Check(
               Near(
                  column_matrix.GetBlockEntry( block, local_row, local_col ),
                  row_matrix.GetBlockEntry( block, local_row, local_col ) ),
               "Assembled row-major and column-major BSR logical entries differ." ) &&
               success;
         }
      }
   }

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.125 + 0.5 * static_cast< Real >( i );
   }

   Vector y_column( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_row( fe_space.GetNumberOfFiniteElementDofs() );
   y_column = 0.0;
   y_row = 0.0;
   column_matrix( x, y_column );
   row_matrix( x, y_row );

   const Real * y_column_data = y_column.ReadHostData();
   const Real * y_row_data = y_row.ReadHostData();
   for ( GlobalIndex i = 0; i < y_column.Size(); ++i )
   {
      success = Check(
         Near( y_column_data[i], y_row_data[i] ),
         "Assembled row-major and column-major BSR apply results differ." ) &&
         success;
   }

   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRawApplySupportsBothBlockLayouts() && success;
   success = TestAssemblySupportsBothBlockLayouts() && success;

   return success ? 0 : 1;
}
