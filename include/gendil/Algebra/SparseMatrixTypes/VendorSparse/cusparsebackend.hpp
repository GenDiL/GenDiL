// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/vendorsparsecommon.hpp"

#include <cstdint>
#include <type_traits>

#if defined(GENDIL_USE_CUDA)

#include <cusparse.h>

#include <cstdlib>
#include <iostream>
#include <memory>

namespace gendil::details
{

inline void CheckCuSparse(
   const cusparseStatus_t status,
   const char * expression,
   const char * context )
{
   if ( status != CUSPARSE_STATUS_SUCCESS )
   {
      std::cerr
         << "GenDiL cuSPARSE error\n"
         << "  context: " << context << '\n'
         << "  expression: " << expression << '\n'
         << "  status: " << cusparseGetErrorString( status )
         << std::endl;
      std::abort();
   }
}

struct CuSparseSpMVState
{
   cusparseHandle_t handle = nullptr;
   cusparseSpMatDescr_t matrix = nullptr;
   cusparseDnVecDescr_t x = nullptr;
   cusparseDnVecDescr_t y = nullptr;
   void * workspace = nullptr;
   size_t workspace_size = 0;
   const void * x_values = nullptr;
   void * y_values = nullptr;
   VendorSpMVConfig config{};
   bool initialized = false;
   bool preprocessed = false;

   CuSparseSpMVState() = default;
   CuSparseSpMVState( const CuSparseSpMVState & ) = delete;
   CuSparseSpMVState & operator=( const CuSparseSpMVState & ) = delete;

   ~CuSparseSpMVState() noexcept
   {
      if ( initialized )
      {
         (void) cudaStreamSynchronize( nullptr );
      }
      if ( workspace != nullptr )
      {
         (void) cudaFree( workspace );
      }
      if ( y != nullptr )
      {
         (void) cusparseDestroyDnVec( y );
      }
      if ( x != nullptr )
      {
         (void) cusparseDestroyDnVec( x );
      }
      if ( matrix != nullptr )
      {
         (void) cusparseDestroySpMat( matrix );
      }
      if ( handle != nullptr )
      {
         (void) cusparseDestroy( handle );
      }
   }
};

template < typename ComputeType >
class CuSparseBackendBase : public DeviceMatVecBackend
{
public:
   using compute_type = ComputeType;
   using accumulator_type = ComputeType;

   CuSparseBackendBase() = default;

   CuSparseBackendBase( const CuSparseBackendBase & )
   { }

   CuSparseBackendBase & operator=( const CuSparseBackendBase & )
   {
      state_.reset();
      last_execution_path_ = VendorSparseExecutionPath::Uninitialized;
      return *this;
   }

   CuSparseBackendBase( CuSparseBackendBase && ) noexcept = default;
   CuSparseBackendBase & operator=( CuSparseBackendBase && ) noexcept = default;

   CuSparseSpMVState & State() const
   {
      if ( state_ == nullptr )
      {
         state_ = std::make_unique< CuSparseSpMVState >();
      }
      return *state_;
   }

   /// Clear cached cuSPARSE descriptors, preprocessing, and workspace.
   ///
   /// Call this on every backend previously applied to a matrix before
   /// modifying that matrix's sparse structure or replacing its storage.
   void ResetState() const
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
   mutable std::unique_ptr< CuSparseSpMVState > state_;
   mutable VendorSparseExecutionPath last_execution_path_ =
      VendorSparseExecutionPath::Uninitialized;
};

} // namespace gendil::details

#endif

namespace gendil
{

#if defined(GENDIL_USE_CUDA)
template < typename ComputeType = void >
using CuSparseBackendBaseFor = details::CuSparseBackendBase< ComputeType >;
#else
template < typename ComputeType = void >
using CuSparseBackendBaseFor =
   details::InactiveVendorSparseBackendBase< ComputeType >;
#endif

template < typename ComputeType = void >
class CuSparseBSRBackend : public CuSparseBackendBaseFor< ComputeType >
{
public:
   void ConfigureShape(
      const std::uint64_t block_rows,
      const std::uint64_t block_cols )
   {
      assembled_square_blocks_ = block_rows == block_cols;
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR)
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
class CuSparseCOOBackend : public CuSparseBackendBaseFor< ComputeType >
{ };

template < typename ComputeType = void >
class CuSparseCSRBackend : public CuSparseBackendBaseFor< ComputeType >
{ };

template < typename ComputeType = void >
class CuSparseCSCBackend : public CuSparseBackendBaseFor< ComputeType >
{ };

static_assert(
   std::is_copy_constructible_v< CuSparseBSRBackend<> > &&
   std::is_copy_assignable_v< CuSparseBSRBackend<> > &&
   std::is_move_constructible_v< CuSparseBSRBackend<> > &&
   std::is_move_assignable_v< CuSparseBSRBackend<> >,
   "cuSPARSE backends must copy configuration without sharing state and "
   "transfer cached state on moves." );

} // namespace gendil
