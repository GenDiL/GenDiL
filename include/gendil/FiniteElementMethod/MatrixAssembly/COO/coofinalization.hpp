// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrix.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcoo.hpp"
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/rawcoosortreduce.hpp"
#include "gendil/Utilities/KernelContext/kernelplacementtraits.hpp"

namespace gendil {

/**
 * Host-only RawCOO finalization.
 *
 * This function reads the current host copy of `RawCOOTripletBuffer`, sorts by
 * `(row, col)`, and additively reduces exact duplicate coordinates. Callers are
 * responsible for ensuring a raw host view is current before calling this
 * overload. The buffer overload obtains a host read view and therefore performs
 * any required device-to-host synchronization lazily.
 */
template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOOHost(
   const RawCOOTripletView< ValueType, IndexType > & raw,
   Backend backend )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   const auto reduced =
      details::MakeSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( raw );

   auto matrix =
      MakeCOOMatrix< StoredValueType, StoredIndexType, Backend >(
         raw.num_rows,
         raw.num_cols,
         static_cast< StoredIndexType >( reduced.size() ),
         std::move( backend ) );

   auto matrix_data = GetHostWriteView( matrix );
   for ( StoredIndexType i = 0; i < matrix_data.nnz; ++i )
   {
      const auto & triplet = reduced[static_cast< size_t >( i )];
      matrix_data.rows[i] = triplet.row;
      matrix_data.cols[i] = triplet.col;
      matrix_data.values[i] = triplet.value;
   }

   return matrix;
}

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

/** GPU RawCOO finalization using CUB or rocPRIM vendor primitives. */
template < typename ValueType, typename IndexType, typename Backend >
COOMatrix< ValueType, IndexType, Backend >
FinalizeRawCOOToCOODevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   if ( !details::DevicePrimitiveCanProcess( raw.nnz_raw ) )
   {
      details::WarnDeviceSparseFinalizationItemLimit();
      return FinalizeRawCOOToCOOHost(
         GetHostReadView( raw ),
         std::move( backend ) );
   }

   auto reduced =
      details::MakeDeviceSortedReducedRawCOOTriplets<
         SparseCoordinateOrder::RowThenColumn >( GetDeviceReadView( raw ) );
   auto matrix = MakeCOOMatrix< ValueType, IndexType, Backend >(
      raw.num_rows,
      raw.num_cols,
      reduced.nnz,
      std::move( backend ) );
   auto output = GetDeviceWriteView( matrix );
   const auto * coordinates = reduced.coordinates.data();
   const auto * values = reduced.values.data();
   DeviceLoop(
      reduced.nnz,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         output.rows[i] = coordinates[i].major;
         output.cols[i] = coordinates[i].minor;
         output.values[i] = values[i];
      } );
   return matrix;
}

template < typename ValueType, typename IndexType >
COOMatrix< ValueType, IndexType, DefaultCOOBackend >
FinalizeRawCOOToCOODevice(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCOODevice(
      raw,
      DefaultCOOBackend{} );
}

#endif // GENDIL_HAS_DEVICE_SPARSE_FINALIZATION

template < typename ValueType, typename IndexType >
auto FinalizeRawCOOToCOOHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw )
{
   return FinalizeRawCOOToCOOHost(
      GetHostReadView( raw ),
      DefaultCOOBackend{} );
}

template < typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOOHost(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
   return FinalizeRawCOOToCOOHost(
      GetHostReadView( raw ),
      std::move( backend ) );
}

template < class KernelPolicy, typename ValueType, typename IndexType, typename Backend >
auto FinalizeRawCOOToCOO(
   const RawCOOTripletBuffer< ValueType, IndexType > & raw,
   Backend backend )
{
#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
   if constexpr ( is_device_configuration_v< KernelPolicy > )
   {
      return FinalizeRawCOOToCOODevice(
         raw,
         std::move( backend ) );
   }
#endif
   return FinalizeRawCOOToCOOHost(
      raw,
      std::move( backend ) );
}

} // namespace gendil
