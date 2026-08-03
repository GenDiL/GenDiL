// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

template < typename T >
concept HasBackendType = requires
{
   typename T::backend_type;
};

template < typename T >
concept HasDataType = requires
{
   typename T::data_type;
};

template < typename T, typename SizeType >
bool IsNull( const SyncHostDeviceArray< T, SizeType > & array )
{
   bool is_null = array.data.host_pointer == nullptr;
#if defined(GENDIL_USE_DEVICE)
   is_null = is_null && array.data.device_pointer == nullptr;
#endif
   return is_null;
}

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool TestBSROwnershipTransfer()
{
   auto source =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 3 );
   const auto * values = source.values.data.host_pointer;
   const auto * row_offsets = source.row_offsets.data.host_pointer;
   const auto * col_indices = source.col_indices.data.host_pointer;

   BSRMatrix<> moved( std::move( source ) );
   bool success =
      Check(
         source.block_rows == 0 &&
         source.block_cols == 0 &&
         source.num_blocks == 0 &&
         IsNull( source.values ) &&
         IsNull( source.row_offsets ) &&
         IsNull( source.col_indices ),
         "Moved-from BSRMatrix is not empty." );
   success =
      Check(
         moved.values.data.host_pointer == values &&
         moved.row_offsets.data.host_pointer == row_offsets &&
         moved.col_indices.data.host_pointer == col_indices,
         "BSRMatrix move construction did not transfer its storage." ) &&
      success;

   auto destination =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved );
   success =
      Check(
         moved.num_blocks == 0 &&
         IsNull( moved.values ) &&
         IsNull( moved.row_offsets ) &&
         IsNull( moved.col_indices ),
         "Move-assigned-from BSRMatrix is not empty." ) &&
      success;
   success =
      Check(
         destination.values.data.host_pointer == values &&
         destination.row_offsets.data.host_pointer == row_offsets &&
         destination.col_indices.data.host_pointer == col_indices,
         "BSRMatrix move assignment did not transfer its storage." ) &&
      success;
   success =
      Check(
         ReadHost( destination.values )[0] == 0.0,
         "BSRMatrix move assignment did not transfer validity state." ) &&
      success;
   return success;
}

bool TestCOOOwnershipTransfer()
{
   auto source = MakeCOOMatrix< Real, GlobalIndex >( 3, 4, 2 );
   auto source_data = GetHostWriteView( source );
   for ( GlobalIndex i = 0; i < source_data.nnz; ++i )
   {
      source_data.rows[i] = i;
      source_data.cols[i] = i;
      source_data.values[i] = Real( i + 1 );
   }
   const auto * rows = source.rows.data.host_pointer;
   const auto * cols = source.cols.data.host_pointer;
   const auto * values = source.values.data.host_pointer;

   COOMatrix<> moved( std::move( source ) );
   bool success =
      Check(
         source.num_rows == 0 &&
         source.num_cols == 0 &&
         source.nnz == 0 &&
         IsNull( source.rows ) &&
         IsNull( source.cols ) &&
         IsNull( source.values ),
         "Moved-from COOMatrix is not empty." );

   auto destination = MakeCOOMatrix< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved );
   success =
      Check(
         destination.rows.data.host_pointer == rows &&
         destination.cols.data.host_pointer == cols &&
         destination.values.data.host_pointer == values &&
         moved.nnz == 0 &&
         IsNull( moved.rows ) &&
         IsNull( moved.cols ) &&
         IsNull( moved.values ),
         "COOMatrix move assignment did not transfer and clear storage." ) &&
      success;
   success =
      Check(
         ReadHost( destination.values )[1] == 2.0,
         "COOMatrix move assignment did not transfer validity state." ) &&
      success;
   return success;
}

bool TestCSROwnershipTransfer()
{
   auto source = MakeCSRMatrix< Real, GlobalIndex >( 3, 4, 2 );
   auto source_data = GetHostWriteView( source );
   for ( GlobalIndex row = 0; row <= source_data.num_rows; ++row )
   {
      source_data.row_ptr[row] = row < 2 ? row : 2;
   }
   for ( GlobalIndex i = 0; i < source_data.nnz; ++i )
   {
      source_data.col_ind[i] = i;
      source_data.values[i] = Real( i + 1 );
   }
   const auto * row_ptr = source.row_ptr.data.host_pointer;
   const auto * col_ind = source.col_ind.data.host_pointer;
   const auto * values = source.values.data.host_pointer;

   CSRMatrix<> moved( std::move( source ) );
   bool success =
      Check(
         source.num_rows == 0 &&
         source.num_cols == 0 &&
         source.nnz == 0 &&
         IsNull( source.row_ptr ) &&
         IsNull( source.col_ind ) &&
         IsNull( source.values ),
         "Moved-from CSRMatrix is not empty." );

   auto destination = MakeCSRMatrix< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved );
   success =
      Check(
         destination.row_ptr.data.host_pointer == row_ptr &&
         destination.col_ind.data.host_pointer == col_ind &&
         destination.values.data.host_pointer == values &&
         moved.nnz == 0 &&
         IsNull( moved.row_ptr ) &&
         IsNull( moved.col_ind ) &&
         IsNull( moved.values ),
         "CSRMatrix move assignment did not transfer and clear storage." ) &&
      success;
   success =
      Check(
         ReadHost( destination.values )[1] == 2.0,
         "CSRMatrix move assignment did not transfer validity state." ) &&
      success;
   return success;
}

bool TestCSCOwnershipTransfer()
{
   auto source = MakeCSCMatrix< Real, GlobalIndex >( 3, 4, 2 );
   auto source_data = GetHostWriteView( source );
   for ( GlobalIndex col = 0; col <= source_data.num_cols; ++col )
   {
      source_data.col_ptr[col] = col < 2 ? col : 2;
   }
   for ( GlobalIndex i = 0; i < source_data.nnz; ++i )
   {
      source_data.row_ind[i] = i;
      source_data.values[i] = Real( i + 1 );
   }
   const auto * col_ptr = source.col_ptr.data.host_pointer;
   const auto * row_ind = source.row_ind.data.host_pointer;
   const auto * values = source.values.data.host_pointer;

   CSCMatrix<> moved( std::move( source ) );
   bool success =
      Check(
         source.num_rows == 0 &&
         source.num_cols == 0 &&
         source.nnz == 0 &&
         IsNull( source.col_ptr ) &&
         IsNull( source.row_ind ) &&
         IsNull( source.values ),
         "Moved-from CSCMatrix is not empty." );

   auto destination = MakeCSCMatrix< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved );
   success =
      Check(
         destination.col_ptr.data.host_pointer == col_ptr &&
         destination.row_ind.data.host_pointer == row_ind &&
         destination.values.data.host_pointer == values &&
         moved.nnz == 0 &&
         IsNull( moved.col_ptr ) &&
         IsNull( moved.row_ind ) &&
         IsNull( moved.values ),
         "CSCMatrix move assignment did not transfer and clear storage." ) &&
      success;
   success =
      Check(
         ReadHost( destination.values )[1] == 2.0,
         "CSCMatrix move assignment did not transfer validity state." ) &&
      success;
   return success;
}

bool TestRawCOOOwnershipTransfer()
{
   auto source =
      MakeRawCOOTripletBuffer< Real, GlobalIndex >( 3, 4, 2 );
   const auto * rows = source.rows.data.host_pointer;
   const auto * cols = source.cols.data.host_pointer;
   const auto * values = source.values.data.host_pointer;

   RawCOOTripletBuffer<> moved( std::move( source ) );
   bool success =
      Check(
         source.num_rows == 0 &&
         source.num_cols == 0 &&
         source.nnz_raw == 0 &&
         IsNull( source.rows ) &&
         IsNull( source.cols ) &&
         IsNull( source.values ),
         "Moved-from RawCOOTripletBuffer is not empty." );

   auto destination =
      MakeRawCOOTripletBuffer< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved );
   success =
      Check(
         destination.rows.data.host_pointer == rows &&
         destination.cols.data.host_pointer == cols &&
         destination.values.data.host_pointer == values &&
         moved.nnz_raw == 0 &&
         IsNull( moved.rows ) &&
         IsNull( moved.cols ) &&
         IsNull( moved.values ),
         "RawCOOTripletBuffer move assignment did not transfer storage." ) &&
      success;
   return success;
}

bool TestRawCOOLayoutAndTarget()
{
   RawCOOAssemblyLayout layout{};
   layout.num_elements = 2;
   layout.num_faces = 1;
   layout.block_entry_count = 4;
   layout.nnz_raw = 8;
   layout.diagonal_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 2 ) );
   layout.offdiag_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 2 ) );
   auto * diagonal = WriteHost( layout.diagonal_offsets );
   auto * offdiag = WriteHost( layout.offdiag_offsets );
   diagonal[0] = 3;
   diagonal[1] = 7;
   offdiag[0] = 11;
   offdiag[1] = 13;
   const auto * diagonal_offsets =
      layout.diagonal_offsets.data.host_pointer;
   const auto * offdiag_offsets =
      layout.offdiag_offsets.data.host_pointer;

   RawCOOAssemblyLayout moved( std::move( layout ) );
   bool success =
      Check(
         layout.num_elements == 0 &&
         layout.nnz_raw == 0 &&
         IsNull( layout.diagonal_offsets ) &&
         IsNull( layout.offdiag_offsets ),
         "Moved-from RawCOOAssemblyLayout is not empty." );

   RawCOOAssemblyLayout destination{};
   destination.diagonal_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 1 ) );
   destination.offdiag_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 1 ) );
   destination = std::move( moved );
   success =
      Check(
         destination.diagonal_offsets.data.host_pointer ==
            diagonal_offsets &&
         destination.offdiag_offsets.data.host_pointer ==
            offdiag_offsets &&
         IsNull( moved.diagonal_offsets ) &&
         IsNull( moved.offdiag_offsets ),
         "RawCOOAssemblyLayout move assignment did not transfer storage." ) &&
      success;
   const auto layout_view = GetHostReadView( destination );
   success =
      Check(
         layout_view.diagonal_offsets[1] == 7 &&
         layout_view.offdiag_offsets[1] == 13,
         "RawCOOAssemblyLayout move assignment did not transfer validity "
         "state." ) &&
      success;

   auto buffer =
      MakeRawCOOTripletBuffer< Real, GlobalIndex >( 3, 4, 8 );
   const auto target =
      MakeRawCOOAssemblyTarget(
         GetHostReadWriteView( buffer ),
         layout_view );
   const auto target_copy = target;
   success =
      Check(
         target_copy.values == ReadHost( buffer.values ) &&
         target_copy.diagonal_offsets ==
            ReadHost( destination.diagonal_offsets ),
         "RawCOOAssemblyTarget copy does not preserve borrowed pointers." ) &&
      success;
   return success;
}

bool TestVectorMoveAssignment()
{
   Vector source( 3 );
   Real * source_data = source.WriteHostData();
   source_data[0] = 1.0;
   source_data[1] = 2.0;
   source_data[2] = 3.0;

   Vector destination( 5 );
   destination = std::move( source );
   return
      Check(
         source.Size() == 0 &&
         destination.Size() == 3 &&
         destination.ReadHostData()[2] == 3.0,
         "Vector move assignment did not safely replace its storage." );
}

bool TestSyncHostDeviceArrayTransitions()
{
   auto array =
      MakeSyncHostDeviceArray< Real >( GlobalIndex( 2 ) );
   bool success =
      Check(
         GetSize( array ) == 2 &&
         !IsHostValid( array ) &&
         !IsDeviceValid( array ),
         "New SyncHostDeviceArray should be uninitialized." );

   Real * host = WriteHost( array );
   host[0] = 3.0;
   host[1] = 5.0;
   success =
      Check(
         IsHostValid( array ) && !IsDeviceValid( array ),
         "WriteHost did not make only the host copy authoritative." ) &&
      success;

   ReadDevice( array );
#if defined(GENDIL_USE_DEVICE)
   success =
      Check(
         IsHostValid( array ) && IsDeviceValid( array ),
         "ReadDevice did not synchronize a host-authoritative array." ) &&
      success;

   (void) ReadWriteDevice( array );
   success =
      Check(
         !IsHostValid( array ) && IsDeviceValid( array ),
         "ReadWriteDevice did not make only the device copy authoritative." ) &&
      success;
   ReadHost( array );
   success =
      Check(
         IsHostValid( array ) && IsDeviceValid( array ),
         "ReadHost did not synchronize a device-authoritative array." ) &&
      success;
#else
   success =
      Check(
         IsHostValid( array ) && !IsDeviceValid( array ),
         "CPU ReadDevice should use the host copy." ) &&
      success;
#endif

   ReadWriteHost( array )[0] = 7.0;
   success =
      Check(
         IsHostValid( array ) && !IsDeviceValid( array ),
         "ReadWriteHost did not invalidate the device copy." ) &&
      success;

   auto moved = std::move( array );
   success =
      Check(
         GetSize( array ) == 0 &&
         !IsHostValid( array ) &&
         !IsDeviceValid( array ) &&
         GetSize( moved ) == 2 &&
         IsHostValid( moved ),
         "Moving SyncHostDeviceArray did not transfer and clear its state." ) &&
      success;

   auto destination =
      MakeSyncHostDeviceArray< Real >( GlobalIndex( 5 ) );
   WriteHost( destination )[0] = -1.0;
   destination = std::move( moved );
   success =
      Check(
         GetSize( destination ) == 2 &&
         ReadHost( destination )[0] == 7.0 &&
         GetSize( moved ) == 0 &&
         IsNull( moved ),
         "SyncHostDeviceArray move assignment did not replace storage." ) &&
      success;

   auto empty =
      MakeSyncHostDeviceArray< Real >( GlobalIndex( 0 ) );
   ReadHost( empty );
   Sync( empty );
   success =
      Check(
         IsHostValid( empty )
#if defined(GENDIL_USE_DEVICE)
            && IsDeviceValid( empty )
#endif
         ,
         "A zero-sized synchronized array is not coherent." ) &&
      success;

   using FixedSize = std::integral_constant< std::size_t, 3 >;
   auto fixed = MakeSyncHostDeviceArray< Real >( FixedSize{} );
   WriteHost( fixed )[2] = 11.0;
   success =
      Check(
         GetSize( fixed ) == 3 &&
         ReadHost( fixed )[2] == 11.0,
         "Fixed-extent SyncHostDeviceArray does not expose its extent." ) &&
      success;
   return success;
}

bool TestSparseValidityMovesWithStorage()
{
   auto source =
      MakeRawCOOTripletBuffer< Real, GlobalIndex >( 1, 1, 1 );
   ReadWriteHost( source.values )[0] = 9.0;

   auto destination = std::move( source );
   bool success =
      Check(
         ReadHost( destination.values )[0] == 9.0,
         "RawCOO validity state did not move with host-authoritative data." );

   Sync( destination );
   success =
      Check(
         GetHostReadView( destination ).values[0] == 9.0,
         "RawCOO synchronization after a move lost the authoritative value." ) &&
      success;
   return success;
}

bool TestPerArrayValidityIsIndependent()
{
   auto matrix = MakeCOOMatrix< Real, GlobalIndex >( 1, 1, 1 );
   auto data = GetHostWriteView( matrix );
   data.rows[0] = 0;
   data.cols[0] = 0;
   data.values[0] = 2.0;
   Sync( matrix );

   ReadWriteHost( matrix.values )[0] = 3.0;
   bool success =
      Check(
         IsHostValid( matrix.rows ) &&
         IsHostValid( matrix.cols ) &&
         IsHostValid( matrix.values ),
         "Changing COO values altered another array's host validity." );
#if defined(GENDIL_USE_DEVICE)
   success =
      Check(
         IsDeviceValid( matrix.rows ) &&
         IsDeviceValid( matrix.cols ) &&
         !IsDeviceValid( matrix.values ),
         "Changing COO values altered another array's device validity." ) &&
      success;
#endif
   return success;
}

} // namespace

static_assert(
   !std::is_copy_constructible_v<
      SyncHostDeviceArray< Real, GlobalIndex > > );
static_assert(
   std::is_move_constructible_v<
      SyncHostDeviceArray< Real, GlobalIndex > > );
static_assert(
   std::is_move_assignable_v<
      SyncHostDeviceArray< Real, GlobalIndex > > );
static_assert( !std::is_copy_constructible_v< BSRMatrix<> > );
static_assert( !std::is_copy_assignable_v< BSRMatrix<> > );
static_assert( std::is_move_constructible_v< BSRMatrix<> > );
static_assert( std::is_move_assignable_v< BSRMatrix<> > );
static_assert( !HasDataType< BSRMatrix<> > );
static_assert( !std::is_copy_constructible_v< COOMatrix<> > );
static_assert( std::is_move_constructible_v< COOMatrix<> > );
static_assert( std::is_move_assignable_v< COOMatrix<> > );
static_assert( !HasDataType< COOMatrix<> > );
static_assert( !std::is_copy_constructible_v< CSRMatrix<> > );
static_assert( std::is_move_constructible_v< CSRMatrix<> > );
static_assert( std::is_move_assignable_v< CSRMatrix<> > );
static_assert( !HasDataType< CSRMatrix<> > );
static_assert( !std::is_copy_constructible_v< CSCMatrix<> > );
static_assert( std::is_move_constructible_v< CSCMatrix<> > );
static_assert( std::is_move_assignable_v< CSCMatrix<> > );
static_assert( !HasDataType< CSCMatrix<> > );
static_assert(
   !std::is_copy_constructible_v< RawCOOTripletBuffer<> > );
static_assert(
   std::is_move_constructible_v< RawCOOTripletBuffer<> > );
static_assert(
   std::is_move_assignable_v< RawCOOTripletBuffer<> > );
static_assert( !HasDataType< RawCOOTripletBuffer<> > );
static_assert( !std::is_copy_constructible_v< RawCOOAssemblyLayout > );
static_assert( std::is_move_constructible_v< RawCOOAssemblyLayout > );
static_assert( std::is_move_assignable_v< RawCOOAssemblyLayout > );
static_assert( std::is_copy_constructible_v< HostDevicePointer< Real > > );
static_assert(
   std::is_copy_constructible_v< RawCOOAssemblyTarget<> > );

static_assert( std::is_trivially_copyable_v< BSRMatrixView<> > );
static_assert( std::is_trivially_destructible_v< BSRMatrixView<> > );
static_assert( !HasBackendType< BSRMatrixView<> > );
static_assert( std::is_trivially_copyable_v< COOMatrixView<> > );
static_assert( std::is_trivially_destructible_v< COOMatrixView<> > );
static_assert( !HasBackendType< COOMatrixView<> > );
static_assert( std::is_trivially_copyable_v< CSRMatrixView<> > );
static_assert( std::is_trivially_destructible_v< CSRMatrixView<> > );
static_assert( !HasBackendType< CSRMatrixView<> > );
static_assert( std::is_trivially_copyable_v< CSCMatrixView<> > );
static_assert( std::is_trivially_destructible_v< CSCMatrixView<> > );
static_assert( !HasBackendType< CSCMatrixView<> > );
static_assert( std::is_trivially_copyable_v< RawCOOTripletView<> > );
static_assert( std::is_trivially_destructible_v< RawCOOTripletView<> > );
static_assert( !HasBackendType< RawCOOTripletView<> > );
static_assert(
   std::is_trivially_copyable_v< RawCOOAssemblyLayoutView<> > );
static_assert(
   std::is_trivially_destructible_v< RawCOOAssemblyLayoutView<> > );

using BSRReadView =
   decltype( GetHostReadView( std::declval< const BSRMatrix<> & >() ) );
using BSRValuesWriteView =
   decltype( GetHostValuesWriteView( std::declval< BSRMatrix<> & >() ) );
using BSRWriteView =
   decltype( GetHostWriteView( std::declval< BSRMatrix<> & >() ) );
using BSRKernelReadView =
   decltype(
      GetKernelReadView< false >(
         std::declval< const BSRMatrix<> & >() ) );
using BSRKernelValuesWriteView =
   decltype(
      GetKernelValuesWriteView< true >(
         std::declval< BSRMatrix<> & >() ) );
static_assert(
   std::is_same_v< decltype(BSRReadView::values), const Real * > );
static_assert(
   std::is_same_v< decltype(BSRValuesWriteView::values), Real * > );
static_assert(
   std::is_same_v<
      decltype(BSRValuesWriteView::row_offsets),
      const GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(BSRWriteView::row_offsets), GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(BSRKernelReadView::values), const Real * > );
static_assert(
   std::is_same_v<
      decltype(BSRKernelValuesWriteView::row_offsets),
      const GlobalIndex * > );

using COOReadView =
   decltype( GetDeviceReadView( std::declval< const COOMatrix<> & >() ) );
using COOValuesWriteView =
   decltype( GetDeviceValuesWriteView( std::declval< COOMatrix<> & >() ) );
using COOWriteView =
   decltype( GetDeviceWriteView( std::declval< COOMatrix<> & >() ) );
using COOKernelWriteView =
   decltype(
      GetKernelWriteView< false >(
         std::declval< COOMatrix<> & >() ) );
static_assert(
   std::is_same_v< decltype(COOReadView::values), const Real * > );
static_assert(
   std::is_same_v< decltype(COOValuesWriteView::values), Real * > );
static_assert(
   std::is_same_v<
      decltype(COOValuesWriteView::rows),
      const GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(COOWriteView::rows), GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(COOKernelWriteView::rows), GlobalIndex * > );

using CSRReadView =
   decltype( GetHostReadView( std::declval< const CSRMatrix<> & >() ) );
using CSRValuesWriteView =
   decltype( GetHostValuesWriteView( std::declval< CSRMatrix<> & >() ) );
using CSRWriteView =
   decltype( GetHostWriteView( std::declval< CSRMatrix<> & >() ) );
using CSRKernelReadWriteView =
   decltype(
      GetKernelReadWriteView< true >(
         std::declval< CSRMatrix<> & >() ) );
static_assert(
   std::is_same_v< decltype(CSRReadView::values), const Real * > );
static_assert(
   std::is_same_v< decltype(CSRValuesWriteView::values), Real * > );
static_assert(
   std::is_same_v<
      decltype(CSRValuesWriteView::row_ptr),
      const GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(CSRWriteView::row_ptr), GlobalIndex * > );
static_assert(
   std::is_same_v<
      decltype(CSRKernelReadWriteView::row_ptr),
      GlobalIndex * > );

using CSCReadView =
   decltype( GetHostReadView( std::declval< const CSCMatrix<> & >() ) );
using CSCValuesWriteView =
   decltype( GetHostValuesWriteView( std::declval< CSCMatrix<> & >() ) );
using CSCWriteView =
   decltype( GetHostWriteView( std::declval< CSCMatrix<> & >() ) );
using CSCKernelReadView =
   decltype(
      GetKernelReadView< true >(
         std::declval< const CSCMatrix<> & >() ) );
static_assert(
   std::is_same_v< decltype(CSCReadView::values), const Real * > );
static_assert(
   std::is_same_v< decltype(CSCValuesWriteView::values), Real * > );
static_assert(
   std::is_same_v<
      decltype(CSCValuesWriteView::col_ptr),
      const GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(CSCWriteView::col_ptr), GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(CSCKernelReadView::values), const Real * > );

using RawReadView =
   decltype(
      GetDeviceReadView(
         std::declval< const RawCOOTripletBuffer<> & >() ) );
using RawWriteView =
   decltype(
      GetDeviceWriteView(
         std::declval< RawCOOTripletBuffer<> & >() ) );
using RawKernelReadView =
   decltype(
      GetKernelReadView< false >(
         std::declval< const RawCOOTripletBuffer<> & >() ) );
static_assert(
   std::is_same_v< decltype(RawReadView::values), const Real * > );
static_assert(
   std::is_same_v< decltype(RawReadView::rows), const GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(RawWriteView::values), Real * > );
static_assert(
   std::is_same_v< decltype(RawWriteView::rows), GlobalIndex * > );
static_assert(
   std::is_same_v< decltype(RawKernelReadView::rows), const GlobalIndex * > );

using RawLayoutReadView =
   decltype(
      GetHostReadView(
         std::declval< const RawCOOAssemblyLayout & >() ) );
using RawLayoutWriteView =
   decltype(
      GetHostWriteView(
         std::declval< RawCOOAssemblyLayout & >() ) );
using RawKernelLayoutReadView =
   decltype(
      GetKernelReadView< true >(
         std::declval< const RawCOOAssemblyLayout & >() ) );
using RawKernelTarget =
   decltype(
      MakeRawCOOAssemblyTarget< false >(
         std::declval< RawCOOTripletBuffer<> & >(),
         std::declval< const RawCOOAssemblyLayout & >() ) );
static_assert(
   std::is_same_v<
      decltype(RawLayoutReadView::diagonal_offsets),
      const GlobalIndex * > );
static_assert(
   std::is_same_v<
      decltype(RawLayoutWriteView::diagonal_offsets),
      GlobalIndex * > );
static_assert(
   std::is_same_v<
      decltype(RawKernelLayoutReadView::diagonal_offsets),
      const GlobalIndex * > );
static_assert( std::is_same_v< RawKernelTarget, RawCOOAssemblyTarget<> > );

int main()
{
   bool success = true;
   success = TestBSROwnershipTransfer() && success;
   success = TestCOOOwnershipTransfer() && success;
   success = TestCSROwnershipTransfer() && success;
   success = TestCSCOwnershipTransfer() && success;
   success = TestRawCOOOwnershipTransfer() && success;
   success = TestRawCOOLayoutAndTarget() && success;
   success = TestVectorMoveAssignment() && success;
   success = TestSyncHostDeviceArrayTransitions() && success;
   success = TestSparseValidityMovesWithStorage() && success;
   success = TestPerArrayValidityIsIndependent() && success;
   return success ? 0 : 1;
}
