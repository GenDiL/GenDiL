// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/rocsparsebackend.hpp"

#if defined(GENDIL_USE_HIP)

#include <rocsparse/rocsparse.h>

#include <cstdint>
#include <type_traits>

namespace gendil::details
{

template < typename IndexType >
constexpr rocsparse_indextype RocSparseIndexType()
{
   CheckVendorSparseIndexType< IndexType >();
   if constexpr ( sizeof( IndexType ) == sizeof( std::int32_t ) )
   {
      return rocsparse_indextype_i32;
   }
   else
   {
      return rocsparse_indextype_i64;
   }
}

template < typename ValueType >
constexpr rocsparse_datatype RocSparseDataType()
{
   if constexpr ( std::is_same_v< std::remove_cv_t< ValueType >, float > )
   {
      return rocsparse_datatype_f32_r;
   }
   else
   {
      static_assert(
         std::is_same_v< std::remove_cv_t< ValueType >, double >,
         "rocSPARSE SpMV supports only float and double GenDiL values." );
      return rocsparse_datatype_f64_r;
   }
}

template <
   typename ValueType,
   typename IndexType,
   typename InputType,
   typename OutputType,
   typename ComputeType >
VendorSpMVConfig MakeRocSparseSpMVConfig(
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

   static_assert(
      !vendor_sparse_float_matrix_double_vector_v<
         ValueType,
         InputType,
         OutputType,
         ComputeType >,
      "rocSPARSE does not support float-matrix/double-vector SpMV. "
      "Use a NativeDevice sparse backend." );

   VendorSpMVConfig config{};
   config.format = format;
   config.rows = CheckedVendorSparseExtent( rows, rows_error );
   config.cols = CheckedVendorSparseExtent( cols, cols_error );
   config.nnz = CheckedVendorSparseExtent( nnz, nnz_error );
   config.index_type =
      static_cast< int >( RocSparseIndexType< IndexType >() );
   config.value_type =
      static_cast< int >( RocSparseDataType< ValueType >() );
   config.input_type =
      static_cast< int >( RocSparseDataType< InputType >() );
   config.output_type =
      static_cast< int >( RocSparseDataType< OutputType >() );
   config.compute_type =
      static_cast< int >( RocSparseDataType< ComputeType >() );
   return config;
}

template <
   typename Backend,
   typename InputType,
   typename OutputType,
   typename ComputeType,
   typename CreateMatrix >
RocSparseSpMVState & InitializeRocSparseSpMV(
   const Backend & backend,
   VendorSpMVConfig config,
   const InputType * x_data,
   OutputType * y_data,
   const rocsparse_spmv_alg algorithm,
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
      CheckRocSparse(
         rocsparse_create_handle( &state.handle ),
         "rocsparse_create_handle",
         context );
      CheckRocSparse(
         rocsparse_set_stream( state.handle, nullptr ),
         "rocsparse_set_stream",
         context );
      CheckRocSparse(
         rocsparse_set_pointer_mode(
            state.handle,
            rocsparse_pointer_mode_host ),
         "rocsparse_set_pointer_mode",
         context );

      create_matrix( &state.matrix );

      CheckRocSparse(
         rocsparse_create_dnvec_descr(
            &state.x,
            config.InputSize(),
            const_cast< InputType * >( x_data ),
            RocSparseDataType< InputType >() ),
         "rocsparse_create_dnvec_descr(x)",
         context );
      CheckRocSparse(
         rocsparse_create_dnvec_descr(
            &state.y,
            config.OutputSize(),
            y_data,
            RocSparseDataType< OutputType >() ),
         "rocsparse_create_dnvec_descr(y)",
         context );

      CheckRocSparse(
         rocsparse_spmv(
            state.handle,
            rocsparse_operation_none,
            &initialization_alpha,
            state.matrix,
            state.x,
            &initialization_beta,
            state.y,
            RocSparseDataType< ComputeType >(),
            algorithm,
            rocsparse_spmv_stage_buffer_size,
            &state.workspace_size,
            nullptr ),
         "rocsparse_spmv(buffer_size)",
         context );

      if ( state.workspace_size > 0 )
      {
         GENDIL_DEVICE_CHECK(
            hipMalloc(
               &state.workspace,
               state.workspace_size ) );
      }

      CheckRocSparse(
         rocsparse_spmv(
            state.handle,
            rocsparse_operation_none,
            &initialization_alpha,
            state.matrix,
            state.x,
            &initialization_beta,
            state.y,
            RocSparseDataType< ComputeType >(),
            algorithm,
            rocsparse_spmv_stage_preprocess,
            &state.workspace_size,
            state.workspace ),
         "rocsparse_spmv(preprocess)",
         context );
      state.preprocessed = true;

      state.x_values = x_data;
      state.y_values = y_data;
      state.config = config;
      state.initialized = true;
   }
   else
   {
      if ( state.x_values != x_data )
      {
         CheckRocSparse(
            rocsparse_dnvec_set_values(
               state.x,
               const_cast< InputType * >( x_data ) ),
            "rocsparse_dnvec_set_values(x)",
            context );
         state.x_values = x_data;
      }
      if ( state.y_values != y_data )
      {
         CheckRocSparse(
            rocsparse_dnvec_set_values( state.y, y_data ),
            "rocsparse_dnvec_set_values(y)",
            context );
         state.y_values = y_data;
      }
   }

   return state;
}

template < typename Backend, typename ComputeType >
void ExecuteRocSparseSpMV(
   const Backend & backend,
   RocSparseSpMVState & state,
   const ComputeType alpha,
   const ComputeType beta,
   const rocsparse_spmv_alg algorithm,
   const char * context )
{
   GENDIL_VERIFY(
      state.initialized,
      "ExecuteRocSparseSpMV requires an initialized SpMV plan." );
   GENDIL_VERIFY(
      state.config.algorithm == static_cast< int >( algorithm ),
      "ExecuteRocSparseSpMV algorithm does not match the initialized plan." );

   CheckRocSparse(
      rocsparse_spmv(
         state.handle,
         rocsparse_operation_none,
         &alpha,
         state.matrix,
         state.x,
         &beta,
         state.y,
         RocSparseDataType< ComputeType >(),
         algorithm,
         rocsparse_spmv_stage_compute,
         &state.workspace_size,
         state.workspace ),
      "rocsparse_spmv(compute)",
      context );

   backend.MarkExecutionPath( VendorSparseExecutionPath::Vendor );
}

} // namespace gendil::details

#endif
