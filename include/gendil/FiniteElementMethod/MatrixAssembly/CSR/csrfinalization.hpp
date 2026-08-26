// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

#include <limits>

namespace gendil {

/**
 * Host-only RawCOO-to-CSR finalization.
 *
 * This function reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(row, col)`, and additively reduces exact duplicate coordinates before
 * building canonical CSR row storage. Exact reduced zeros are retained.
 */
template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSRHost(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCSRMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   const StoredIndexType row_ptr_size =
      matrix_data.num_rows + StoredIndexType( 1 );
   for ( StoredIndexType row = 0; row < row_ptr_size; ++row )
   {
      matrix_data.row_ptr[row] = StoredIndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix_data.num_rows,
         "FinalizeRawCOOToCSR received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix_data.num_cols,
         "FinalizeRawCOOToCSR received a column outside the matrix dimensions." );
      ++matrix_data.row_ptr[triplet.row + StoredIndexType( 1 )];
   }

   for ( StoredIndexType row = 0; row < matrix_data.num_rows; ++row )
   {
      matrix_data.row_ptr[row + StoredIndexType( 1 )] +=
         matrix_data.row_ptr[row];
   }

   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.col_ind[i] = triplet.col;
      matrix_data.values[i] = triplet.value;
   }

   return matrix;
}

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

/** GPU RawCOO-to-CSR finalization using CUB or rocPRIM primitives. */
template < typename IndexType >
bool DeviceCompressedPointerCountCanProcess( const IndexType major_extent )
{
   if ( major_extent == std::numeric_limits< IndexType >::max() )
   {
      return false;
   }
   return details::DevicePrimitiveCanProcess(
      major_extent + IndexType( 1 ) );
}

template < typename ValueType, typename IndexType, typename Backend >
CSRMatrix< ValueType, IndexType, Backend >
FinalizeRawCOOToCSRDevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   if ( !details::DevicePrimitiveCanProcess( raw.nnz_raw ) ||
        !DeviceCompressedPointerCountCanProcess( raw.num_rows ) )
   {
      details::WarnDeviceSparseFinalizationItemLimit();
      return FinalizeRawCOOToCSRHost(
         GetHostReadView( raw ),
         std::move( backend ) );
   }

   auto reduced =
      details::MakeDeviceSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( GetDeviceReadView( raw ) );
   auto matrix = MakeCSRMatrix< ValueType, IndexType, Backend >(
      raw.num_rows,
      raw.num_cols,
      reduced.nnz,
      std::move( backend ) );
   auto output = GetDeviceWriteView( matrix );
   const IndexType pointer_count = raw.num_rows + IndexType( 1 );
   details::DeviceMemset(
      output.row_ptr,
      0,
      static_cast< size_t >( pointer_count ) * sizeof( IndexType ) );

   const auto * coordinates = reduced.coordinates.data();
   const auto * values = reduced.values.data();
   const IndexType nnz = reduced.nnz;
   DeviceLoop(
      nnz,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         const auto coordinate = coordinates[i];
         output.col_ind[i] = coordinate.minor;
         output.values[i] = values[i];
         if ( i + IndexType( 1 ) == nnz ||
              coordinates[i + IndexType( 1 )].major != coordinate.major )
         {
            output.row_ptr[coordinate.major + IndexType( 1 )] =
               i + IndexType( 1 );
         }
      } );

   details::DeviceOnlyBuffer< IndexType > scanned_pointers(
      static_cast< size_t >( pointer_count ) );
   details::DeviceInclusiveScan(
      output.row_ptr,
      scanned_pointers.data(),
      pointer_count,
      details::DeviceMaximum< IndexType >{} );
   details::DeviceCopyToDevice(
      output.row_ptr,
      scanned_pointers.data(),
      static_cast< size_t >( pointer_count ) * sizeof( IndexType ) );
   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSRDevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCSRDevice(
      raw,
      DefaultCSRBackend{} );
}

#endif // GENDIL_HAS_DEVICE_SPARSE_FINALIZATION

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSRHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCSRHost(
      GetHostReadView( raw ),
      DefaultCSRBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSRHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   return FinalizeRawCOOToCSRHost(
      GetHostReadView( raw ),
      std::move( backend ) );
}

template < class KernelPolicy, typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSR(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
   if constexpr ( is_device_configuration_v< KernelPolicy > )
   {
      return FinalizeRawCOOToCSRDevice(
         raw,
         std::move( backend ) );
   }
#endif
   return FinalizeRawCOOToCSRHost(
      raw,
      std::move( backend ) );
}

} // namespace gendil
