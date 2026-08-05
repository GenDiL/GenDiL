// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/cusparsebackend.hpp"

#if defined(GENDIL_USE_CUDA)

#include <cusparse.h>

#include <cstdint>
#include <type_traits>

namespace gendil::details
{

template < typename IndexType >
constexpr cusparseIndexType_t CuSparseIndexType()
{
   CheckVendorSparseIndexType< IndexType >();
   if constexpr ( sizeof( IndexType ) == sizeof( std::int32_t ) )
   {
      return CUSPARSE_INDEX_32I;
   }
   else
   {
      return CUSPARSE_INDEX_64I;
   }
}

template < typename ValueType >
constexpr cudaDataType CuSparseDataType()
{
   if constexpr ( std::is_same_v< std::remove_cv_t< ValueType >, float > )
   {
      return CUDA_R_32F;
   }
   else
   {
      static_assert(
         std::is_same_v< std::remove_cv_t< ValueType >, double >,
         "cuSPARSE SpMV supports only float and double GenDiL values." );
      return CUDA_R_64F;
   }
}

template <
   typename ValueType,
   typename IndexType,
   typename InputType,
   typename OutputType,
   typename ComputeType >
VendorSpMVConfig MakeCuSparseSpMVConfig(
   const VendorSparseFormat format,
   const IndexType rows,
   const IndexType cols,
   const IndexType nnz,
   const char * rows_error,
   const char * cols_error,
   const char * nnz_error )
{
   CheckVendorSparseArithmetic<
      ValueType,
      InputType,
      OutputType,
      ComputeType >();

#if !defined(GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV)
   static_assert(
      !vendor_sparse_float_matrix_double_vector_v<
         ValueType,
         InputType,
         OutputType,
         ComputeType >,
      "cuSPARSE float-matrix/double-vector SpMV is unavailable in this "
      "CUDA toolkit. Use a NativeDevice sparse backend." );
#endif

   VendorSpMVConfig config{};
   config.format = format;
   config.rows = CheckedVendorSparseExtent( rows, rows_error );
   config.cols = CheckedVendorSparseExtent( cols, cols_error );
   config.nnz = CheckedVendorSparseExtent( nnz, nnz_error );
   config.index_type =
      static_cast< int >( CuSparseIndexType< IndexType >() );
   config.value_type =
      static_cast< int >( CuSparseDataType< ValueType >() );
   config.input_type =
      static_cast< int >( CuSparseDataType< InputType >() );
   config.output_type =
      static_cast< int >( CuSparseDataType< OutputType >() );
   config.compute_type =
      static_cast< int >( CuSparseDataType< ComputeType >() );
   return config;
}

template <
   typename Backend,
   typename InputType,
   typename OutputType,
   typename ComputeType,
   typename CreateMatrix >
CuSparseSpMVState & InitializeCuSparseSpMV(
   const Backend & backend,
   VendorSpMVConfig config,
   const InputType * x_data,
   OutputType * y_data,
   const cusparseSpMVAlg_t algorithm,
   CreateMatrix create_matrix,
   const char * context )
{
   const ComputeType initialization_alpha = ComputeType( 1 );
   const ComputeType initialization_beta = ComputeType( 0 );
   config.algorithm = static_cast< int >( algorithm );
   auto & current = backend.State();
   if ( current.initialized &&
        !current.config.SameSparsePlan( config ) )
   {
      ResetState( backend );
   }

   auto & state = backend.State();
   if ( !state.initialized )
   {
      CheckCuSparse(
         cusparseCreate( &state.handle ),
         "cusparseCreate",
         context );
      CheckCuSparse(
         cusparseSetStream( state.handle, nullptr ),
         "cusparseSetStream",
         context );
      CheckCuSparse(
         cusparseSetPointerMode(
            state.handle,
            CUSPARSE_POINTER_MODE_HOST ),
         "cusparseSetPointerMode",
         context );

      create_matrix( &state.matrix );

      CheckCuSparse(
         cusparseCreateDnVec(
            &state.x,
            config.InputSize(),
            const_cast< InputType * >( x_data ),
            CuSparseDataType< InputType >() ),
         "cusparseCreateDnVec(x)",
         context );
      CheckCuSparse(
         cusparseCreateDnVec(
            &state.y,
            config.OutputSize(),
            y_data,
            CuSparseDataType< OutputType >() ),
         "cusparseCreateDnVec(y)",
         context );

      CheckCuSparse(
         cusparseSpMV_bufferSize(
            state.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &initialization_alpha,
            state.matrix,
            state.x,
            &initialization_beta,
            state.y,
            CuSparseDataType< ComputeType >(),
            algorithm,
            &state.workspace_size ),
         "cusparseSpMV_bufferSize",
         context );

      if ( state.workspace_size > 0 )
      {
         GENDIL_DEVICE_CHECK(
            cudaMalloc(
               &state.workspace,
               state.workspace_size ) );
      }

#if defined(GENDIL_CUSPARSE_HAS_SPMV_PREPROCESS)
      CheckCuSparse(
         cusparseSpMV_preprocess(
            state.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &initialization_alpha,
            state.matrix,
            state.x,
            &initialization_beta,
            state.y,
            CuSparseDataType< ComputeType >(),
            algorithm,
            state.workspace ),
         "cusparseSpMV_preprocess",
         context );
      state.preprocessed = true;
#endif

      state.x_values = x_data;
      state.y_values = y_data;
      state.config = config;
      state.initialized = true;
   }
   else
   {
      if ( state.x_values != x_data )
      {
         CheckCuSparse(
            cusparseDnVecSetValues(
               state.x,
               const_cast< InputType * >( x_data ) ),
            "cusparseDnVecSetValues(x)",
            context );
         state.x_values = x_data;
      }
      if ( state.y_values != y_data )
      {
         CheckCuSparse(
            cusparseDnVecSetValues( state.y, y_data ),
            "cusparseDnVecSetValues(y)",
            context );
         state.y_values = y_data;
      }
   }

   return state;
}

template < typename Backend, typename ComputeType >
void ExecuteCuSparseSpMV(
   const Backend & backend,
   CuSparseSpMVState & state,
   const ComputeType alpha,
   const ComputeType beta,
   const cusparseSpMVAlg_t algorithm,
   const char * context )
{
   GENDIL_VERIFY(
      state.initialized,
      "ExecuteCuSparseSpMV requires an initialized SpMV plan." );
   GENDIL_VERIFY(
      state.config.algorithm == static_cast< int >( algorithm ),
      "ExecuteCuSparseSpMV algorithm does not match the initialized plan." );

   CheckCuSparse(
      cusparseSpMV(
         state.handle,
         CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha,
         state.matrix,
         state.x,
         &beta,
         state.y,
         CuSparseDataType< ComputeType >(),
         algorithm,
         state.workspace ),
      "cusparseSpMV",
      context );

   backend.MarkExecutionPath( VendorSparseExecutionPath::Vendor );
}

} // namespace gendil::details

#endif
