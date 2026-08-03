// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapplycommon.hpp"
#include "gendil/Utilities/MathHelperFunctions/atomicadd.hpp"

#include <limits>

namespace gendil
{

namespace details
{

#if defined(GENDIL_USE_DEVICE)
template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType >
__global__
void COODeviceApplyKernel(
   const Backend backend,
   const MatrixView matrix,
   const InputValue * x,
   OutputValue * y,
   const AlphaType alpha )
{
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using InputValueType = std::remove_cv_t< InputValue >;
   using OutputValueType = std::remove_cv_t< OutputValue >;
   using ComputeType =
      ResolveSparseComputeType_t< Backend, ValueType, InputValueType >;

   CheckScatterSparseApplyArithmetic<
      ValueType,
      InputValueType,
      OutputValueType,
      ComputeType >();
   const ComputeType compute_alpha =
      static_cast< ComputeType >( alpha );

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
         compute_alpha *
         static_cast< ComputeType >( matrix.values[entry] ) *
         static_cast< ComputeType >( x[col] );
      AtomicAddInPlace(
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
   typename MatrixView,
   typename InputValue,
   typename OutputValue,
   typename AlphaType,
   typename BetaType >
void ApplyDeviceCOOToRawPointers(
   const Backend & backend,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data,
   const AlphaType alpha,
   const BetaType beta )
{
   constexpr unsigned int threads_per_block = 256;
   const dim3 block_dim( threads_per_block );

   ScaleSparseDeviceOutput(
      y_data,
      matrix.num_rows,
      beta );

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
         y_data,
         alpha );
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
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const NativeDeviceCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCOOApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );

   details::ApplyDeviceCOOToRawPointers(
      backend,
      matrix_data,
      x_data,
      y_data,
      Real( 1 ),
      Real( 0 ) );
#else
   static_assert(
      dependent_false_v< COOMatrix< ValueType, IndexType, MatrixBackend > >,
      "Apply(NativeDeviceCOOBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP). Use COOMatrix::operator() for CPU execution." );
#endif
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const NativeDeviceCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
#if defined(GENDIL_USE_DEVICE)
   const auto matrix_data = GetDeviceReadView( matrix );
   details::CheckCOOApplyDimensions(
      matrix_data,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( matrix_data.nnz == 0 )
   {
      return;
   }

   details::ApplyDeviceCOOToRawPointers(
      backend,
      matrix_data,
      ReadDeviceVector( x ),
      ReadWriteDeviceVector( y ),
      Real( 1 ),
      Real( 1 ) );
#else
   static_assert(
      dependent_false_v< COOMatrix< ValueType, IndexType, MatrixBackend > >,
      "ApplyAdd(NativeDeviceCOOBackend, ...) requires GENDIL_USE_DEVICE "
      "(CUDA or HIP)." );
#endif
}

} // namespace gendil
