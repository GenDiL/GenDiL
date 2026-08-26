// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

#include <limits>

namespace gendil {

/**
 * Host-only RawCOO-to-CSC finalization.
 *
 * This function reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(col, row)`, and additively reduces exact duplicate coordinates before
 * building canonical CSC column storage. Exact reduced zeros are retained.
 */
template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSCHost(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::ColumnThenRow >( raw );

   auto matrix =
      MakeCSCMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   const StoredIndexType col_ptr_size =
      matrix_data.num_cols + StoredIndexType( 1 );
   for ( StoredIndexType col = 0; col < col_ptr_size; ++col )
   {
      matrix_data.col_ptr[col] = StoredIndexType( 0 );
   }

   for ( const auto & triplet : reduced )
   {
      GENDIL_VERIFY(
         triplet.row < matrix_data.num_rows,
         "FinalizeRawCOOToCSC received a row outside the matrix dimensions." );
      GENDIL_VERIFY(
         triplet.col < matrix_data.num_cols,
         "FinalizeRawCOOToCSC received a column outside the matrix dimensions." );
      ++matrix_data.col_ptr[triplet.col + StoredIndexType( 1 )];
   }

   for ( StoredIndexType col = 0; col < matrix_data.num_cols; ++col )
   {
      matrix_data.col_ptr[col + StoredIndexType( 1 )] +=
         matrix_data.col_ptr[col];
   }

   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.row_ind[i] = triplet.row;
      matrix_data.values[i] = triplet.value;
   }

   return matrix;
}

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

/** GPU RawCOO-to-CSC finalization using CUB or rocPRIM primitives. */
template < typename IndexType >
bool DeviceCSCCompressedPointerCountCanProcess(
   const IndexType major_extent )
{
   if ( major_extent == std::numeric_limits< IndexType >::max() )
   {
      return false;
   }
   return details::DevicePrimitiveCanProcess(
      major_extent + IndexType( 1 ) );
}

template < typename ValueType, typename IndexType, typename Backend >
CSCMatrix< ValueType, IndexType, Backend >
FinalizeRawCOOToCSCDevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   if ( !details::DevicePrimitiveCanProcess( raw.nnz_raw ) ||
        !DeviceCSCCompressedPointerCountCanProcess( raw.num_cols ) )
   {
      details::WarnDeviceSparseFinalizationItemLimit();
      return FinalizeRawCOOToCSCHost(
         GetHostReadView( raw ),
         std::move( backend ) );
   }

   auto reduced =
      details::MakeDeviceSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::ColumnThenRow >( GetDeviceReadView( raw ) );
   auto matrix = MakeCSCMatrix< ValueType, IndexType, Backend >(
      raw.num_rows,
      raw.num_cols,
      reduced.nnz,
      std::move( backend ) );
   auto output = GetDeviceWriteView( matrix );
   const IndexType pointer_count = raw.num_cols + IndexType( 1 );
   details::DeviceMemset(
      output.col_ptr,
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
         output.row_ind[i] = coordinate.minor;
         output.values[i] = values[i];
         if ( i + IndexType( 1 ) == nnz ||
              coordinates[i + IndexType( 1 )].major != coordinate.major )
         {
            output.col_ptr[coordinate.major + IndexType( 1 )] =
               i + IndexType( 1 );
         }
      } );

   details::DeviceOnlyBuffer< IndexType > scanned_pointers(
      static_cast< size_t >( pointer_count ) );
   details::DeviceInclusiveScan(
      output.col_ptr,
      scanned_pointers.data(),
      pointer_count,
      details::DeviceMaximum< IndexType >{} );
   details::DeviceCopyToDevice(
      output.col_ptr,
      scanned_pointers.data(),
      static_cast< size_t >( pointer_count ) * sizeof( IndexType ) );
   return matrix;
}

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSCDevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCSCDevice(
      raw,
      DefaultCSCBackend{} );
}

#endif // GENDIL_HAS_DEVICE_SPARSE_FINALIZATION

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCSCHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCSCHost(
      GetHostReadView( raw ),
      DefaultCSCBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSCHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   return FinalizeRawCOOToCSCHost(
      GetHostReadView( raw ),
      std::move( backend ) );
}

template < class KernelPolicy, typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCSC(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
   if constexpr ( is_device_configuration_v< KernelPolicy > )
   {
      return FinalizeRawCOOToCSCDevice(
         raw,
         std::move( backend ) );
   }
#endif
   return FinalizeRawCOOToCSCHost(
      raw,
      std::move( backend ) );
}

} // namespace gendil
