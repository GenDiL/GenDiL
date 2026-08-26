// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSR/csrmatrixapplycommon.hpp"
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
CuSparseSpMVState & InitializeCuSparseCSRSpMV(
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
         VendorSparseFormat::CSR,
         matrix.num_rows,
         matrix.num_cols,
         matrix.nnz,
         "cuSPARSE CSR row count exceeds the signed vendor range.",
         "cuSPARSE CSR column count exceeds the signed vendor range.",
         "cuSPARSE CSR nnz exceeds the signed vendor range." );
   config.offsets = matrix.row_ptr;
   config.indices = matrix.col_ind;
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
               cusparseCreateCsr(
                  descriptor,
                  config.rows,
                  config.cols,
                  config.nnz,
                  const_cast< IndexType * >(
                     matrix.row_ptr ),
                  const_cast< IndexType * >(
                     matrix.col_ind ),
                  const_cast< ValueType * >(
                     matrix.values ),
                  CuSparseIndexType< IndexType >(),
                  CuSparseIndexType< IndexType >(),
                  CUSPARSE_INDEX_BASE_ZERO,
                  CuSparseDataType< ValueType >() ),
               "cusparseCreateCsr",
               "Apply(CuSparseCSRBackend, ...)" );
         },
         "Apply(CuSparseCSRBackend, ...)" );
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
   const CuSparseCSRBackend< BackendComputeType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCSRApplyDimensions(
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
         CuSparseCSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCSRSpMV(
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
      "Apply(CuSparseCSRBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const CuSparseCSRBackend< BackendComputeType > & backend,
   const CSRMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCSRApplyDimensions(
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
         CuSparseCSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCSRSpMV(
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
      "ApplyAdd(CuSparseCSRBackend, ...)" );
}

} // namespace gendil

#endif
