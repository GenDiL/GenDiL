// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/SparseMatrixTypes/COO/rawcooview.hpp"

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
#include "gendil/FiniteElementMethod/MatrixAssembly/COO/deviceprimitives.hpp"
#include "gendil/Utilities/Loop/deviceloop.hpp"
#endif

#include <algorithm>
#include <type_traits>
#include <vector>

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
#include <iostream>
#endif

namespace gendil {

/** Supported canonical coordinate orders for sparse finalization. */
enum class SparseCoordinateOrder
{
   RowThenColumn,             ///< Row-major lexicographic order.
   RowThenDiagonalThenColumn, ///< Row-major with each diagonal entry first.
   ColumnThenRow              ///< Column-major lexicographic order.
};

namespace details
{

/** Host staging representation of one raw COO entry. */
template < typename ValueType, typename IndexType >
struct RawCOOTriplet
{
   IndexType row;
   IndexType col;
   ValueType value;
};

/** Compare two COO triplets using the requested canonical order. */
template < SparseCoordinateOrder Order, typename Triplet >
bool RawCOOTripletLess( const Triplet & lhs, const Triplet & rhs )
{
   if constexpr ( Order == SparseCoordinateOrder::RowThenColumn )
   {
      if ( lhs.row != rhs.row )
      {
         return lhs.row < rhs.row;
      }
      return lhs.col < rhs.col;
   }
   else if constexpr ( Order == SparseCoordinateOrder::RowThenDiagonalThenColumn )
   {
      if ( lhs.row != rhs.row )
      {
         return lhs.row < rhs.row;
      }

      const bool lhs_is_diagonal = lhs.row == lhs.col;
      const bool rhs_is_diagonal = rhs.row == rhs.col;
      if ( lhs_is_diagonal != rhs_is_diagonal )
      {
         return lhs_is_diagonal;
      }

      return lhs.col < rhs.col;
   }
   else
   {
      if ( lhs.col != rhs.col )
      {
         return lhs.col < rhs.col;
      }
      return lhs.row < rhs.row;
   }
}

/**
 * Host-sort raw COO entries and additively reduce duplicate coordinates.
 * Exact reduced zeros are retained in the returned canonical triplet vector.
 */
template <
   SparseCoordinateOrder Order,
   typename ValueType,
   typename IndexType >
auto MakeSortedReducedRawCOOTriplets(
   const RawCOOTripletView< ValueType, IndexType > & raw )
{
   using StoredValueType = std::remove_const_t< ValueType >;
   using StoredIndexType = std::remove_const_t< IndexType >;
   using Triplet = RawCOOTriplet< StoredValueType, StoredIndexType >;

   std::vector< Triplet > triplets;
   triplets.reserve( static_cast< size_t >( raw.nnz_raw ) );

   for ( StoredIndexType i = 0; i < raw.nnz_raw; ++i )
   {
      triplets.push_back(
         Triplet{ raw.rows[i], raw.cols[i], raw.values[i] } );
   }

   std::sort(
      triplets.begin(),
      triplets.end(),
      [] ( const Triplet & lhs, const Triplet & rhs )
      {
         return RawCOOTripletLess< Order >( lhs, rhs );
      } );

   std::vector< Triplet > reduced;
   reduced.reserve( triplets.size() );

   for ( const auto & triplet : triplets )
   {
      if ( !reduced.empty() &&
           reduced.back().row == triplet.row &&
           reduced.back().col == triplet.col )
      {
         reduced.back().value += triplet.value;
      }
      else
      {
         reduced.push_back( triplet );
      }
   }

   return reduced;
}

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

/** Device coordinate stored as order-dependent major and minor components. */
template < typename IndexType >
struct DeviceSparseCoordinate
{
   IndexType major{};
   IndexType minor{};

   /** Return whether both coordinate components are equal. */
   GENDIL_HOST_DEVICE
   bool operator==( const DeviceSparseCoordinate & other ) const
   {
      return major == other.major && minor == other.minor;
   }
};

/** Coordinate and matrix value carried alongside each device radix-sort key. */
template < typename IndexType, typename ValueType >
struct DeviceSparseSortData
{
   IndexType other_coordinate{};
   ValueType value{};
};

/** Device-callable equality operation for sparse coordinate keys. */
template < typename Coordinate >
struct DeviceCoordinateEqual
{
   /** Return whether two sparse coordinate keys are equal. */
   GENDIL_HOST_DEVICE
   bool operator()( const Coordinate & lhs, const Coordinate & rhs ) const
   {
      return lhs.major == rhs.major && lhs.minor == rhs.minor;
   }
};

/** Device-callable additive reduction operation. */
template < typename ValueType >
struct DeviceAdd
{
   /** Add two reduction values. */
   GENDIL_HOST_DEVICE
   ValueType operator()( const ValueType & lhs, const ValueType & rhs ) const
   {
      return lhs + rhs;
   }
};

/** Device-callable maximum operation used to propagate compressed pointers. */
template < typename T >
struct DeviceMaximum
{
   /** Return the greater of two values. */
   GENDIL_HOST_DEVICE
   T operator()( const T lhs, const T rhs ) const
   {
      return lhs < rhs ? rhs : lhs;
   }
};

/** Device-callable minimum operation used for missing-diagonal metadata. */
template < typename T >
struct DeviceMinimum
{
   /** Return the lesser of two values. */
   GENDIL_HOST_DEVICE
   T operator()( const T lhs, const T rhs ) const
   {
      return rhs < lhs ? rhs : lhs;
   }
};

/** Record coordinate validation failure, atomically when running on-device. */
GENDIL_HOST_DEVICE inline void RecordInvalidSparseCoordinate( int * status )
{
#if defined(GENDIL_DEVICE_CODE)
   atomicExch( status, 1 );
#else
   *status = 1;
#endif
}

/** Owning device result of coordinate sorting and duplicate reduction. */
template < typename ValueType, typename IndexType >
struct DeviceSortedReducedRawCOO
{
   using coordinate_type = DeviceSparseCoordinate< IndexType >;

   IndexType num_rows{};
   IndexType num_cols{};
   IndexType nnz{};
   DeviceOnlyBuffer< coordinate_type > coordinates{};
   DeviceOnlyBuffer< ValueType > values{};
};

/** Emit the device primitive item-limit fallback warning at most once. */
inline void WarnDeviceSparseFinalizationItemLimit()
{
   [[maybe_unused]] static const bool warning_emitted =
      []
      {
         std::cerr
            << "GenDiL warning: sparse assembly exceeds the GPU primitive "
               "item-count limit; raw triplets will move to host for "
               "finalization and the canonical matrix may later move back "
               "to device.\n";
         return true;
      }();
}

/** Return whether a possibly signed coordinate lies outside an extent. */
template < typename IndexType >
GENDIL_HOST_DEVICE bool SparseCoordinateOutOfBounds(
   const IndexType coordinate,
   const IndexType extent )
{
   if constexpr ( std::is_signed_v< IndexType > )
   {
      return coordinate < IndexType( 0 ) || coordinate >= extent;
   }
   else
   {
      return coordinate >= extent;
   }
}

/**
 * Validate, lexicographically sort, and additively reduce raw COO on-device.
 * Only validation status and the reduced entry count are copied to the host;
 * canonical coordinates and values remain in device-only storage.
 */
template <
   SparseCoordinateOrder Order,
   typename ValueType,
   typename IndexType >
DeviceSortedReducedRawCOO< ValueType, IndexType >
MakeDeviceSortedReducedRawCOOTriplets(
   const RawCOOTripletView< const ValueType, const IndexType > & raw )
{
   static_assert(
      Order == SparseCoordinateOrder::RowThenColumn ||
         Order == SparseCoordinateOrder::ColumnThenRow,
      "Device finalization sorts lexicographically; Hypre diagonal-first "
      "ordering is applied after row-major reduction." );

   using Coordinate = DeviceSparseCoordinate< IndexType >;
   using SortData = DeviceSparseSortData< IndexType, ValueType >;
   const IndexType count = raw.nnz_raw;

   DeviceSortedReducedRawCOO< ValueType, IndexType > reduced{};
   reduced.num_rows = raw.num_rows;
   reduced.num_cols = raw.num_cols;
   reduced.coordinates =
      DeviceOnlyBuffer< Coordinate >( static_cast< size_t >( count ) );
   reduced.values =
      DeviceOnlyBuffer< ValueType >( static_cast< size_t >( count ) );

   if ( count == IndexType( 0 ) )
   {
      return reduced;
   }

   DeviceOnlyBuffer< IndexType > keys_a( static_cast< size_t >( count ) );
   DeviceOnlyBuffer< IndexType > keys_b( static_cast< size_t >( count ) );
   DeviceOnlyBuffer< SortData > sort_data_a( static_cast< size_t >( count ) );
   DeviceOnlyBuffer< SortData > sort_data_b( static_cast< size_t >( count ) );
   DeviceOnlyBuffer< int > validation_status( 1 );
   DeviceMemset( validation_status.data(), 0, sizeof( int ) );

   auto * minor_keys = keys_a.data();
   auto * input_sort_data = sort_data_a.data();
   auto * status = validation_status.data();
   const auto rows = raw.rows;
   const auto cols = raw.cols;
   const auto values = raw.values;
   const IndexType num_rows = raw.num_rows;
   const IndexType num_cols = raw.num_cols;

   DeviceLoop(
      count,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         const IndexType row = rows[i];
         const IndexType col = cols[i];
         const ValueType value = values[i];
         const IndexType major_key =
            Order == SparseCoordinateOrder::RowThenColumn ? row : col;
         const IndexType minor_key =
            Order == SparseCoordinateOrder::RowThenColumn ? col : row;

         if ( SparseCoordinateOutOfBounds( row, num_rows ) ||
              SparseCoordinateOutOfBounds( col, num_cols ) )
         {
            RecordInvalidSparseCoordinate( status );
         }

         minor_keys[i] = minor_key;
         input_sort_data[i] = SortData{ major_key, value };
      } );

   DeviceStableRadixSortPairs(
      keys_a.data(),
      keys_b.data(),
      sort_data_a.data(),
      sort_data_b.data(),
      count );

   auto * major_keys = keys_a.data();
   auto * major_sort_data = sort_data_a.data();
   const auto * sorted_minor_keys = keys_b.data();
   const auto * sorted_minor_data = sort_data_b.data();
   DeviceLoop(
      count,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         major_keys[i] = sorted_minor_data[i].other_coordinate;
         major_sort_data[i] =
            SortData{ sorted_minor_keys[i], sorted_minor_data[i].value };
      } );

   DeviceStableRadixSortPairs(
      keys_a.data(),
      keys_b.data(),
      sort_data_a.data(),
      sort_data_b.data(),
      count );

   DeviceOnlyBuffer< Coordinate > sorted_coordinates(
      static_cast< size_t >( count ) );
   DeviceOnlyBuffer< ValueType > sorted_values(
      static_cast< size_t >( count ) );
   auto * coordinate_output = sorted_coordinates.data();
   auto * value_output = sorted_values.data();
   const auto * sorted_major_keys = keys_b.data();
   const auto * sorted_major_data = sort_data_b.data();
   DeviceLoop(
      count,
      [=] GENDIL_HOST_DEVICE ( const IndexType i )
      {
         coordinate_output[i] = Coordinate{
            sorted_major_keys[i],
            sorted_major_data[i].other_coordinate };
         value_output[i] = sorted_major_data[i].value;
      } );

   DeviceOnlyBuffer< IndexType > unique_count( 1 );
   DeviceReduceByKey(
      sorted_coordinates.data(),
      reduced.coordinates.data(),
      sorted_values.data(),
      reduced.values.data(),
      unique_count.data(),
      count,
      DeviceAdd< ValueType >{},
      DeviceCoordinateEqual< Coordinate >{} );

   int host_status = 0;
   DeviceCopyToHost(
      &host_status,
      validation_status.data(),
      sizeof( host_status ) );
   GENDIL_VERIFY(
      host_status == 0,
      "GPU sparse finalization received a coordinate outside the matrix "
      "dimensions." );

   DeviceCopyToHost(
      &reduced.nnz,
      unique_count.data(),
      sizeof( reduced.nnz ) );
   return reduced;
}

#endif // GENDIL_HAS_DEVICE_SPARSE_FINALIZATION

} // namespace details

} // namespace gendil
