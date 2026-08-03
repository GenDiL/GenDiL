// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/prelude.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace gendil
{

/**
 * Last execution path used by a vendor backend.
 *
 * This is primarily useful for diagnostics and tests. Vendor backends are not
 * thread-safe for concurrent calls through the same backend object.
 */
enum class VendorSparseExecutionPath
{
   Uninitialized,
   Vendor,
   Trivial
};

namespace details
{

enum class VendorSparseFormat
{
   BSR,
   COO,
   CSR,
   CSC
};

struct VendorSpMVConfig
{
   VendorSparseFormat format = VendorSparseFormat::CSR;
   std::int64_t rows = 0;
   std::int64_t cols = 0;
   std::int64_t nnz = 0;
   std::int64_t block_rows = 1;
   std::int64_t block_cols = 1;
   const void * offsets = nullptr;
   const void * indices = nullptr;
   const void * secondary_indices = nullptr;
   const void * values = nullptr;
   int index_type = 0;
   int value_type = 0;
   int input_type = 0;
   int output_type = 0;
   int compute_type = 0;
   int layout = 0;
   int algorithm = 0;

   std::int64_t InputSize() const
   {
      return cols * block_cols;
   }

   std::int64_t OutputSize() const
   {
      return rows * block_rows;
   }

   bool SameSparsePlan( const VendorSpMVConfig & other ) const
   {
      return
         format == other.format &&
         rows == other.rows &&
         cols == other.cols &&
         nnz == other.nnz &&
         block_rows == other.block_rows &&
         block_cols == other.block_cols &&
         offsets == other.offsets &&
         indices == other.indices &&
         secondary_indices == other.secondary_indices &&
         values == other.values &&
         index_type == other.index_type &&
         value_type == other.value_type &&
         input_type == other.input_type &&
         output_type == other.output_type &&
         compute_type == other.compute_type &&
         layout == other.layout &&
         algorithm == other.algorithm;
   }
};

template < typename ComputeType >
class InactiveVendorSparseBackendBase : public DeviceMatVecBackend
{
public:
   using compute_type = ComputeType;
   using accumulator_type = ComputeType;

   VendorSparseExecutionPath LastExecutionPath() const
   {
      return last_execution_path_;
   }

   bool HasCachedPlan() const
   {
      return false;
   }

   void ResetState() const
   { }

   void MarkExecutionPath( const VendorSparseExecutionPath path ) const
   {
      last_execution_path_ = path;
   }

private:
   mutable VendorSparseExecutionPath last_execution_path_ =
      VendorSparseExecutionPath::Uninitialized;
};

template < typename IndexType >
constexpr void CheckVendorSparseIndexType()
{
   static_assert(
      std::is_integral_v< IndexType > &&
      !std::is_same_v< std::remove_cv_t< IndexType >, bool >,
      "Vendor sparse backends require an integral matrix index type." );
   static_assert(
      sizeof( IndexType ) == sizeof( std::int32_t ) ||
      sizeof( IndexType ) == sizeof( std::int64_t ),
      "Vendor sparse backends support only bit-compatible 32-bit or 64-bit "
      "matrix index arrays." );
}

template < typename IndexType >
std::int64_t CheckedVendorSparseExtent(
   const IndexType value,
   const char * message )
{
   CheckVendorSparseIndexType< IndexType >();

   if constexpr ( std::is_signed_v< IndexType > )
   {
      GENDIL_VERIFY( value >= 0, message );
   }

   using UnsignedIndex = std::make_unsigned_t< IndexType >;
   const auto unsigned_value = static_cast< UnsignedIndex >( value );
   const auto signed_max =
      static_cast< UnsignedIndex >(
         std::numeric_limits<
            std::make_signed_t< IndexType > >::max() );
   GENDIL_VERIFY( unsigned_value <= signed_max, message );
   return static_cast< std::int64_t >( unsigned_value );
}

template < typename IndexType >
bool VendorSparseExtentFits( const IndexType value )
{
   CheckVendorSparseIndexType< IndexType >();
   if constexpr ( std::is_signed_v< IndexType > )
   {
      if ( value < 0 )
      {
         return false;
      }
   }

   using UnsignedIndex = std::make_unsigned_t< IndexType >;
   const auto signed_max =
      static_cast< UnsignedIndex >(
         std::numeric_limits<
            std::make_signed_t< IndexType > >::max() );
   return static_cast< UnsignedIndex >( value ) <= signed_max;
}

template < typename IndexType >
bool VendorSparseProductFits(
   const IndexType lhs,
   const IndexType rhs )
{
   if ( !VendorSparseExtentFits( lhs ) ||
        !VendorSparseExtentFits( rhs ) )
   {
      return false;
   }

   const auto lhs64 = static_cast< std::uint64_t >( lhs );
   const auto rhs64 = static_cast< std::uint64_t >( rhs );
   const auto maximum =
      static_cast< std::uint64_t >(
         std::numeric_limits< std::int64_t >::max() );
   return lhs64 == 0 || rhs64 <= maximum / lhs64;
}

template < typename IndexType >
bool IsVendorBSRStorageEligible(
   const IndexType block_rows,
   const IndexType block_cols,
   const IndexType num_row_blocks,
   const IndexType num_col_blocks,
   const IndexType num_blocks )
{
   CheckVendorSparseIndexType< IndexType >();

   return
      block_rows > 0 &&
      block_cols > 0 &&
      num_blocks > 0 &&
      VendorSparseExtentFits( block_rows ) &&
      VendorSparseExtentFits( block_cols ) &&
      VendorSparseExtentFits( num_row_blocks ) &&
      VendorSparseExtentFits( num_col_blocks ) &&
      VendorSparseExtentFits( num_blocks ) &&
      VendorSparseProductFits( num_row_blocks, block_rows ) &&
      VendorSparseProductFits( num_col_blocks, block_cols );
}

template <
   typename ValueType,
   typename InputType,
   typename OutputType,
   typename ComputeType >
inline constexpr bool vendor_sparse_float_matrix_double_vector_v =
   std::is_same_v< std::remove_cv_t< ValueType >, float > &&
   std::is_same_v< std::remove_cv_t< InputType >, double > &&
   std::is_same_v< std::remove_cv_t< OutputType >, double > &&
   std::is_same_v< std::remove_cv_t< ComputeType >, double >;

template <
   typename ValueType,
   typename InputType,
   typename OutputType,
   typename ComputeType >
inline constexpr bool vendor_sparse_arithmetic_supported_v =
   (
      std::is_same_v< ValueType, float > &&
      std::is_same_v< InputType, float > &&
      std::is_same_v< OutputType, float > &&
      std::is_same_v< ComputeType, float >
   ) ||
   (
      std::is_same_v< ValueType, double > &&
      std::is_same_v< InputType, double > &&
      std::is_same_v< OutputType, double > &&
      std::is_same_v< ComputeType, double >
   ) ||
   vendor_sparse_float_matrix_double_vector_v<
      ValueType,
      InputType,
      OutputType,
      ComputeType >;

template <
   typename ValueType,
   typename InputType,
   typename OutputType,
   typename ComputeType >
constexpr void CheckVendorSparseArithmetic()
{
   static_assert(
      vendor_sparse_arithmetic_supported_v<
         std::remove_cv_t< ValueType >,
         std::remove_cv_t< InputType >,
         std::remove_cv_t< OutputType >,
         std::remove_cv_t< ComputeType > >,
      "Vendor sparse SpMV supports float or double uniform precision and "
      "float-matrix/double-vector computation when supported by the selected "
      "vendor and toolkit. Select a NativeDevice sparse backend for other "
      "arithmetic combinations." );
}

/**
 * Execute an empty additive SpMV by preserving the output and clearing any
 * vendor plan that can no longer describe the matrix.
 */
template < typename Backend >
void ExecuteEmptySparseSpMV( const Backend & backend )
{
   backend.ResetState();
   backend.MarkExecutionPath( VendorSparseExecutionPath::Trivial );
}

/**
 * Execute an empty overwrite SpMV by zeroing the device output and clearing
 * any vendor plan that can no longer describe the matrix.
 */
template < typename Backend, typename OutputValue, typename IndexType >
void ExecuteEmptySparseSpMV(
   const Backend & backend,
   OutputValue * output,
   const IndexType output_size )
{
   backend.ResetState();
   ScaleSparseDeviceOutput(
      output,
      output_size,
      std::remove_cv_t< OutputValue >( 0 ) );
   backend.MarkExecutionPath( VendorSparseExecutionPath::Trivial );
}

} // namespace details

} // namespace gendil
