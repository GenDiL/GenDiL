// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/matvecbackend.hpp"
#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/vendorsparsebackend.hpp"
#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/synchostdevicearray.hpp"

#include <type_traits>
#include <utility>

namespace gendil
{

template < typename ComputeType = void >
struct HostCOOBackend : HostMatVecBackend
{
   using compute_type = ComputeType;
};

template < typename ComputeType = void >
struct NativeDeviceCOOBackend : DeviceMatVecBackend
{
   using compute_type = ComputeType;
};

#if defined(GENDIL_USE_DEVICE)
using DefaultCOOBackend = VendorDeviceCOOBackend<>;
#else
using DefaultCOOBackend = HostCOOBackend<>;
#endif

/**
 * Canonical, move-only coordinate-list sparse matrix owner.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCOOBackend >
struct COOMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   SyncHostDeviceArray< IndexType, IndexType > rows{};
   SyncHostDeviceArray< IndexType, IndexType > cols{};
   SyncHostDeviceArray< ValueType, IndexType > values{};

   // Keep the backend last so cached descriptors are destroyed before arrays.
   Backend backend{};

   COOMatrix() = default;
   COOMatrix( const COOMatrix & ) = delete;
   COOMatrix & operator=( const COOMatrix & ) = delete;

   COOMatrix( COOMatrix && other )
      noexcept( std::is_nothrow_move_constructible_v< Backend > )
   : num_rows( std::exchange( other.num_rows, IndexType( 0 ) ) ),
     num_cols( std::exchange( other.num_cols, IndexType( 0 ) ) ),
     nnz( std::exchange( other.nnz, IndexType( 0 ) ) ),
     rows( std::move( other.rows ) ),
     cols( std::move( other.cols ) ),
     values( std::move( other.values ) ),
     backend( std::move( other.backend ) )
   { }

   COOMatrix & operator=( COOMatrix && other )
      noexcept( std::is_nothrow_move_assignable_v< Backend > )
   {
      if ( this != &other )
      {
         if constexpr ( requires { backend.ResetState(); } )
         {
            backend.ResetState();
         }

         num_rows = std::exchange( other.num_rows, IndexType( 0 ) );
         num_cols = std::exchange( other.num_cols, IndexType( 0 ) );
         nnz = std::exchange( other.nnz, IndexType( 0 ) );
         rows = std::move( other.rows );
         cols = std::move( other.cols );
         values = std::move( other.values );
         backend = std::move( other.backend );
      }
      return *this;
   }

   ~COOMatrix() = default;

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const;
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCOOBackend >
auto MakeCOOMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz,
   Backend backend = Backend{} )
{
   COOMatrix< ValueType, IndexType, Backend > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;
   matrix.rows = MakeSyncHostDeviceArray< IndexType >( nnz );
   matrix.cols = MakeSyncHostDeviceArray< IndexType >( nnz );
   matrix.values = MakeSyncHostDeviceArray< ValueType >( nnz );
   matrix.backend = std::move( backend );
   return matrix;
}

} // namespace gendil
