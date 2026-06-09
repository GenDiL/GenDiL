// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Algebra/vector.hpp"
#include "gendil/Algebra/SparseMatrixTypes/sparseapplyarithmetic.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

#include <limits>

namespace gendil {

template < typename ComputeType = void >
struct HostCOOBackend
{
   using compute_type = ComputeType;
};

template < typename ComputeType = void >
struct NativeDeviceCOOBackend
{
   using compute_type = ComputeType;
};

using DefaultCOOBackend = HostCOOBackend<>;

template <
   typename ValueType = Real,
   typename IndexType = GlobalIndex,
   typename Backend = DefaultCOOBackend >
struct COOMatrix;

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const HostCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y );

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const HostCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y );
#endif

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HostCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y );

/**
 * Canonical coordinate-list sparse matrix.
 *
 * Entries are sorted lexicographically by `(row, col)`, and each coordinate
 * appears at most once. Duplicate raw assembly triplets are finalized by
 * additive reduction before they are stored here. Exact zero values produced by
 * reduction are retained.
 */
template <
   typename ValueType,
   typename IndexType,
   typename Backend >
struct COOMatrix
{
   using value_type = ValueType;
   using index_type = IndexType;
   using backend_type = Backend;

   IndexType num_rows = 0;
   IndexType num_cols = 0;
   IndexType nnz = 0;

   HostDevicePointer< IndexType > rows;
   HostDevicePointer< IndexType > cols;
   HostDevicePointer< ValueType > values;

   // Backend controls operator() dispatch. HostCOOBackend reads/writes host
   // data; NativeDeviceCOOBackend reads/writes device data.
   Backend backend{};

   template < typename InputVector, typename OutputVector >
   void operator()( const InputVector & x, OutputVector & y ) const
   {
      Apply( backend, *this, x, y );
   }
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
   matrix.backend = backend;

   AllocateHostPointer( nnz, matrix.rows );
   AllocateDevicePointer( nnz, matrix.rows );
   AllocateHostPointer( nnz, matrix.cols );
   AllocateDevicePointer( nnz, matrix.cols );
   AllocateHostPointer( nnz, matrix.values );
   AllocateDevicePointer( nnz, matrix.values );

   return matrix;
}

template < typename Matrix >
void FreeCOOMatrix( Matrix & matrix )
{
   FreeHostPointer( matrix.rows );
   FreeDevicePointer( matrix.rows );
   FreeHostPointer( matrix.cols );
   FreeDevicePointer( matrix.cols );
   FreeHostPointer( matrix.values );
   FreeDevicePointer( matrix.values );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   typename InputVector,
   typename OutputVector >
void Apply(
   const HostCOOBackend< BackendComputeType > &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< InputVector >,
      "Apply(HostCOOBackend, COOMatrix, x, y) supports only explicit "
      "host vector overloads. Add an overload that acquires the correct "
      "read/write memory space before entering the COO loop." );
}

template <
   typename Backend,
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename InputVector,
   typename OutputVector >
void Apply(
   const Backend &,
   const COOMatrix< ValueType, IndexType, MatrixBackend > &,
   const InputVector &,
   OutputVector & )
{
   static_assert(
      dependent_false_v< Backend >,
      "Apply(backend, COOMatrix, x, y) currently supports only "
      "HostCOOBackend and NativeDeviceCOOBackend. OpenMP is enabled through "
      "HostCOOBackend when GENDIL_USE_OPENMP is defined; row-segmented, "
      "format-converting, and other COO backends are not implemented yet." );
}

namespace details
{

template < typename Matrix >
void CheckCOOApplyDimensions(
   const Matrix & matrix,
   const size_t x_size,
   const size_t y_size )
{
   GENDIL_VERIFY(
      x_size == static_cast< size_t >( matrix.num_cols ),
      "Apply(COO backend, ...) input vector has the wrong size." );
   GENDIL_VERIFY(
      y_size == static_cast< size_t >( matrix.num_rows ),
      "Apply(COO backend, ...) output vector has the wrong size." );
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyHostCOOToRawPointers(
   const Backend &,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;
   using InputValueType = std::remove_cv_t< InputValue >;
   using OutputValueType = std::remove_cv_t< OutputValue >;
   using ComputeType =
      ResolveSparseComputeType_t< Backend, ValueType, InputValueType >;

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();

   #pragma omp parallel for
   for ( IndexType row = 0; row < matrix.num_rows; ++row )
   {
      y_data[row] = OutputValue( 0 );
   }

   #pragma omp parallel for
   for ( IndexType entry = 0; entry < matrix.nnz; ++entry )
   {
      const IndexType row = matrix.rows[entry];
      const IndexType col = matrix.cols[entry];
      const ComputeType contribution =
         static_cast< ComputeType >( matrix.values[entry] ) *
         static_cast< ComputeType >( x_data[col] );
      #pragma omp atomic
      y_data[row] += static_cast< OutputValueType >( contribution );
   }
}

#if defined(GENDIL_USE_DEVICE)
template < typename ValueType, typename IndexType >
__global__
void COODeviceZeroKernel(
   ValueType * y,
   const IndexType num_rows )
{
   const IndexType stride =
      static_cast< IndexType >( blockDim.x ) *
      static_cast< IndexType >( gridDim.x );

   for ( IndexType row =
            static_cast< IndexType >( blockIdx.x ) *
            static_cast< IndexType >( blockDim.x ) +
            static_cast< IndexType >( threadIdx.x );
         row < num_rows;
         row += stride )
   {
      y[row] = ValueType( 0 );
   }
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
__global__
void COODeviceApplyKernel(
   const Backend backend,
   const Matrix matrix,
   const InputValue * x,
   OutputValue * y )
{
   using IndexType = typename Matrix::index_type;
   using ValueType = typename Matrix::value_type;
   using InputValueType = std::remove_cv_t< InputValue >;
   using OutputValueType = std::remove_cv_t< OutputValue >;
   using ComputeType =
      ResolveSparseComputeType_t< Backend, ValueType, InputValueType >;

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();

   const IndexType stride =
      static_cast< IndexType >( blockDim.x ) *
      static_cast< IndexType >( gridDim.x );

   for ( IndexType entry =
            static_cast< IndexType >( blockIdx.x ) *
            static_cast< IndexType >( blockDim.x ) +
            static_cast< IndexType >( threadIdx.x );
         entry < matrix.nnz;
         entry += stride )
   {
      const IndexType row = matrix.rows[entry];
      const IndexType col = matrix.cols[entry];
      const ComputeType contribution =
         static_cast< ComputeType >( matrix.values[entry] ) *
         static_cast< ComputeType >( x[col] );
      AtomicAdd(
         y[row],
         static_cast< OutputValueType >( contribution ) );
   }
}

inline dim3 MakeCOOApplyGrid(
   const GlobalIndex work_items,
   const char * error_message )
{
   constexpr unsigned int threads_per_block = 256;
   const GlobalIndex grid_x_size =
      ( work_items + threads_per_block - 1 ) / threads_per_block;

   GENDIL_VERIFY(
      grid_x_size <=
         static_cast< GlobalIndex >( std::numeric_limits< unsigned int >::max() ),
      error_message );

   return dim3( static_cast< unsigned int >( grid_x_size ) );
}

template <
   typename Backend,
   typename Matrix,
   typename InputValue,
   typename OutputValue >
void ApplyDeviceCOOToRawPointers(
   const Backend & backend,
   const Matrix & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   constexpr unsigned int threads_per_block = 256;
   const dim3 block_dim( threads_per_block );

   if ( matrix.num_rows > 0 )
   {
      const dim3 grid_dim =
         MakeCOOApplyGrid(
            static_cast< GlobalIndex >( matrix.num_rows ),
            "Apply(NativeDeviceCOOBackend, ...) zero launch grid is too large." );
      CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "Apply(NativeDeviceCOOBackend, ...): before zero launch" );
      COODeviceZeroKernel<<< grid_dim, block_dim >>>(
         y_data,
         matrix.num_rows );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "Apply(NativeDeviceCOOBackend, ...) zero" );
   }

   if ( matrix.nnz > 0 )
   {
      const dim3 grid_dim =
         MakeCOOApplyGrid(
            static_cast< GlobalIndex >( matrix.nnz ),
            "Apply(NativeDeviceCOOBackend, ...) apply launch grid is too large." );
      CheckDeviceLaunchConfiguration( grid_dim, block_dim, 0 );
      GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(
         "Apply(NativeDeviceCOOBackend, ...): before apply launch" );
      COODeviceApplyKernel<<< grid_dim, block_dim >>>(
         backend,
         matrix,
         x_data,
         y_data );
      GENDIL_CHECK_LAST_DEVICE_LAUNCH(
         "Apply(NativeDeviceCOOBackend, ...) apply" );
   }
}
#endif

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const HostCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
   details::CheckCOOApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadHostData();
   auto * y_data = y.WriteHostData();

   details::ApplyHostCOOToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const Vector & x,
   Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCOOApplyDimensions(
      matrix,
      x.Size(),
      y.Size() );

   const auto * x_data = x.ReadDeviceData();
   auto * y_data = y.WriteDeviceData();

   details::ApplyDeviceCOOToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< COOMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCOOBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use COOMatrix::operator() for CPU execution." );
#endif
}

#ifdef GENDIL_USE_MFEM
template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const HostCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
   details::CheckCOOApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.HostRead();
   auto * y_data = y.HostWrite();

   details::ApplyHostCOOToRawPointers( backend, matrix, x_data, y_data );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType >
void Apply(
   const NativeDeviceCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const mfem::Vector & x,
   mfem::Vector & y )
{
#if defined(GENDIL_USE_DEVICE)
   details::CheckCOOApplyDimensions(
      matrix,
      static_cast< size_t >( x.Size() ),
      static_cast< size_t >( y.Size() ) );

   const auto * x_data = x.Read();
   auto * y_data = y.Write();

   details::ApplyDeviceCOOToRawPointers( backend, matrix, x_data, y_data );
#else
   static_assert(
      dependent_false_v< COOMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCOOBackend, mfem::Vector, ...) requires "
      "GENDIL_USE_DEVICE (CUDA or HIP)." );
#endif
}
#endif

} // namespace gendil
