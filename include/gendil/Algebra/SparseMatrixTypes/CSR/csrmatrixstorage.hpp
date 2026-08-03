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

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct HostCSRBackend : HostMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

template <
   typename ComputeType = void,
   typename AccumulatorType = void >
struct NativeDeviceCSRBackend : DeviceMatVecBackend
{
   using compute_type = ComputeType;
   using accumulator_type = AccumulatorType;
};

#if defined(GENDIL_USE_DEVICE)
using DefaultCSRBackend = VendorDeviceCSRBackend<>;
#else
using DefaultCSRBackend = HostCSRBackend<>;
#endif

/**
 * Canonical, move-only compressed sparse row matrix owner.
 */
template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSRBackend >
struct CSRMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   SyncHostDeviceArray< IndexType, IndexType > row_ptr{};
   SyncHostDeviceArray< IndexType, IndexType > col_ind{};
   SyncHostDeviceArray< ValueType, IndexType > values{};

   // Keep the backend last so cached descriptors are destroyed before arrays.
   Backend backend{};

   CSRMatrix() = default;
   CSRMatrix( const CSRMatrix & ) = delete;
   CSRMatrix & operator=( const CSRMatrix & ) = delete;

   CSRMatrix( CSRMatrix && other )
      noexcept( std::is_nothrow_move_constructible_v< Backend > )
   : num_rows( std::exchange( other.num_rows, IndexType( 0 ) ) ),
     num_cols( std::exchange( other.num_cols, IndexType( 0 ) ) ),
     nnz( std::exchange( other.nnz, IndexType( 0 ) ) ),
     row_ptr( std::move( other.row_ptr ) ),
     col_ind( std::move( other.col_ind ) ),
     values( std::move( other.values ) ),
     backend( std::move( other.backend ) )
   { }

   CSRMatrix & operator=( CSRMatrix && other )
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
         row_ptr = std::move( other.row_ptr );
         col_ind = std::move( other.col_ind );
         values = std::move( other.values );
         backend = std::move( other.backend );
      }
      return *this;
   }

   ~CSRMatrix() = default;

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const;
};

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCSRBackend >
auto MakeCSRMatrix(
   const IndexType num_rows,
   const IndexType num_cols,
   const IndexType nnz,
   Backend backend = Backend{} )
{
   CSRMatrix< ValueType, IndexType, Backend > matrix{};
   matrix.num_rows = num_rows;
   matrix.num_cols = num_cols;
   matrix.nnz = nnz;
   matrix.row_ptr =
      MakeSyncHostDeviceArray< IndexType >( num_rows + IndexType( 1 ) );
   matrix.col_ind = MakeSyncHostDeviceArray< IndexType >( nnz );
   matrix.values = MakeSyncHostDeviceArray< ValueType >( nnz );
   matrix.backend = std::move( backend );
   return matrix;
}

} // namespace gendil
