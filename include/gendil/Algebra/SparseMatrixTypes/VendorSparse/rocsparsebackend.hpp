// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/vendorsparsecommon.hpp"

#include <cstdint>
#include <type_traits>

#if defined(GENDIL_USE_HIP)

#include <rocsparse/rocsparse.h>

#include <cstdlib>
#include <iostream>
#include <memory>

namespace gendil::details
{

inline void CheckRocSparse(
   const rocsparse_status status,
   const char * expression,
   const char * context )
{
   if ( status != rocsparse_status_success )
   {
      std::cerr
         << "GenDiL rocSPARSE error\n"
         << "  context: " << context << '\n'
         << "  expression: " << expression << '\n'
         << "  status: " << static_cast< int >( status )
         << std::endl;
      std::abort();
   }
}

struct RocSparseSpMVState
{
   rocsparse_handle handle = nullptr;
   rocsparse_spmat_descr matrix = nullptr;
   rocsparse_dnvec_descr x = nullptr;
   rocsparse_dnvec_descr y = nullptr;
   void * workspace = nullptr;
   size_t workspace_size = 0;
   const void * x_values = nullptr;
   void * y_values = nullptr;
   VendorSpMVConfig config{};
   bool initialized = false;
   bool preprocessed = false;

   RocSparseSpMVState() = default;
   RocSparseSpMVState( const RocSparseSpMVState & ) = delete;
   RocSparseSpMVState & operator=( const RocSparseSpMVState & ) = delete;

   ~RocSparseSpMVState() noexcept
   {
      if ( initialized )
      {
         (void) hipStreamSynchronize( nullptr );
      }
      if ( workspace != nullptr )
      {
         (void) hipFree( workspace );
      }
      if ( y != nullptr )
      {
         (void) rocsparse_destroy_dnvec_descr( y );
      }
      if ( x != nullptr )
      {
         (void) rocsparse_destroy_dnvec_descr( x );
      }
      if ( matrix != nullptr )
      {
         (void) rocsparse_destroy_spmat_descr( matrix );
      }
      if ( handle != nullptr )
      {
         (void) rocsparse_destroy_handle( handle );
      }
   }
};

template < typename ComputeType >
class RocSparseBackendBase : public DeviceMatVecBackend
{
public:
   using compute_type = ComputeType;
   using accumulator_type = ComputeType;

   RocSparseBackendBase() = default;

   RocSparseBackendBase( const RocSparseBackendBase & )
   { }

   RocSparseBackendBase & operator=( const RocSparseBackendBase & )
   {
      state_.reset();
      last_execution_path_ = VendorSparseExecutionPath::Uninitialized;
      return *this;
   }

   RocSparseBackendBase( RocSparseBackendBase && ) noexcept = default;
   RocSparseBackendBase & operator=( RocSparseBackendBase && ) noexcept = default;

   RocSparseSpMVState & State() const
   {
      if ( state_ == nullptr )
      {
         state_ = std::make_unique< RocSparseSpMVState >();
      }
      return *state_;
   }

   /// Reset cached rocSPARSE state.
   ///
   /// Clears descriptors, preprocessing, and workspace allocations that
   /// reference a specific sparse matrix's storage.
   void ClearCachedState() const noexcept
   {
      state_.reset();
   }

   VendorSparseExecutionPath LastExecutionPath() const
   {
      return last_execution_path_;
   }

   bool HasCachedPlan() const
   {
      return state_ != nullptr && state_->initialized;
   }

   void MarkExecutionPath( const VendorSparseExecutionPath path ) const
   {
      last_execution_path_ = path;
   }

private:
   mutable std::unique_ptr< RocSparseSpMVState > state_;
   mutable VendorSparseExecutionPath last_execution_path_ =
      VendorSparseExecutionPath::Uninitialized;
};

} // namespace gendil::details

#endif

namespace gendil
{

#if defined(GENDIL_USE_HIP)
/// Clear cached state owned by a rocSPARSE backend.
template < typename ComputeType >
inline void ResetState(
   const details::RocSparseBackendBase< ComputeType > & backend ) noexcept
{
   backend.ClearCachedState();
}

template < typename ComputeType = void >
using RocSparseBackendBaseFor = details::RocSparseBackendBase< ComputeType >;
#else
template < typename ComputeType = void >
using RocSparseBackendBaseFor =
   details::InactiveVendorSparseBackendBase< ComputeType >;
#endif

template < typename ComputeType = void >
class RocSparseBSRBackend : public RocSparseBackendBaseFor< ComputeType >
{
public:
   void ConfigureShape(
      const std::uint64_t block_rows,
      const std::uint64_t block_cols )
   {
      assembled_square_blocks_ = block_rows == block_cols;
#if defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
      assembled_vendor_eligible_ = assembled_square_blocks_;
#else
      assembled_vendor_eligible_ = false;
#endif
   }

   bool AssembledSquareBlocks() const
   {
      return assembled_square_blocks_;
   }

   bool AssembledVendorEligible() const
   {
      return assembled_vendor_eligible_;
   }

   void ConfigureStorage(
      const std::uint64_t block_rows,
      const std::uint64_t block_cols,
      const bool storage_eligible )
   {
      ConfigureShape( block_rows, block_cols );
      assembled_vendor_eligible_ =
         assembled_vendor_eligible_ && storage_eligible;
   }

private:
   bool assembled_square_blocks_ = false;
   bool assembled_vendor_eligible_ = false;
};

template < typename ComputeType = void >
class RocSparseCOOBackend : public RocSparseBackendBaseFor< ComputeType >
{ };

template < typename ComputeType = void >
class RocSparseCSRBackend : public RocSparseBackendBaseFor< ComputeType >
{ };

template < typename ComputeType = void >
class RocSparseCSCBackend : public RocSparseBackendBaseFor< ComputeType >
{ };

static_assert(
   std::is_copy_constructible_v< RocSparseBSRBackend<> > &&
   std::is_copy_assignable_v< RocSparseBSRBackend<> > &&
   std::is_move_constructible_v< RocSparseBSRBackend<> > &&
   std::is_move_assignable_v< RocSparseBSRBackend<> >,
   "rocSPARSE backends must copy configuration without sharing state and "
   "transfer cached state on moves." );

} // namespace gendil
