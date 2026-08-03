// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapplycommon.hpp"
#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/cusparsespmv.hpp"

#if defined(GENDIL_USE_CUDA)

namespace gendil
{

namespace details
{

template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue >
CuSparseSpMVState & InitializeCuSparseCOOSpMV(
   const Backend & backend,
   const MatrixView & matrix,
   const InputValue * x_data,
   OutputValue * y_data )
{
   using ValueType = typename MatrixView::value_type;
   using IndexType = typename MatrixView::index_type;
   using ComputeType =
      ResolveSparseComputeType_t<
         Backend,
         ValueType,
         std::remove_cv_t< InputValue > >;

   auto config = MakeCuSparseSpMVConfig<
      ValueType,
      IndexType,
      InputValue,
      OutputValue,
      ComputeType >(
         VendorSparseFormat::COO,
         matrix.num_rows,
         matrix.num_cols,
         matrix.nnz,
         "cuSPARSE COO row count exceeds the signed vendor range.",
         "cuSPARSE COO column count exceeds the signed vendor range.",
         "cuSPARSE COO nnz exceeds the signed vendor range." );
   config.indices = matrix.rows;
   config.secondary_indices = matrix.cols;
   config.values = matrix.values;

   return InitializeCuSparseSpMV<
      Backend,
      InputValue,
      OutputValue,
      ComputeType >(
         backend,
         config,
         x_data,
         y_data,
         CUSPARSE_SPMV_ALG_DEFAULT,
         [&] ( cusparseSpMatDescr_t * descriptor )
         {
            CheckCuSparse(
               cusparseCreateCoo(
                  descriptor,
                  config.rows,
                  config.cols,
                  config.nnz,
                  const_cast< IndexType * >(
                     matrix.rows ),
                  const_cast< IndexType * >(
                     matrix.cols ),
                  const_cast< ValueType * >(
                     matrix.values ),
                  CuSparseIndexType< IndexType >(),
                  CUSPARSE_INDEX_BASE_ZERO,
                  CuSparseDataType< ValueType >() ),
               "cusparseCreateCoo",
               "Apply(CuSparseCOOBackend, ...)" );
         },
         "Apply(CuSparseCOOBackend, ...)" );
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const CuSparseCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCOOApplyDimensions(
      view,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( details::IsEmpty( view ) )
   {
      details::ExecuteEmptySparseSpMV(
         backend,
         WriteDeviceVector( y ),
         view.num_rows );
      return;
   }

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );
   using InputValue = SparsePointerValueType_t< decltype( x_data ) >;
   using ComputeType =
      ResolveSparseComputeType_t<
         CuSparseCOOBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCOOSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteCuSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 0 ),
      CUSPARSE_SPMV_ALG_DEFAULT,
      "Apply(CuSparseCOOBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const CuSparseCOOBackend< BackendComputeType > & backend,
   const COOMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCOOApplyDimensions(
      view,
      GetVectorSize( x ),
      GetVectorSize( y ) );
   if ( details::IsEmpty( view ) )
   {
      details::ExecuteEmptySparseSpMV( backend );
      return;
   }

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = ReadWriteDeviceVector( y );
   using InputValue = SparsePointerValueType_t< decltype( x_data ) >;
   using ComputeType =
      ResolveSparseComputeType_t<
         CuSparseCOOBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCOOSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteCuSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 1 ),
      CUSPARSE_SPMV_ALG_DEFAULT,
      "ApplyAdd(CuSparseCOOBackend, ...)" );
}

} // namespace gendil

#endif
