// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixapplycommon.hpp"
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
CuSparseSpMVState & InitializeCuSparseCSCSpMV(
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
         VendorSparseFormat::CSC,
         matrix.num_rows,
         matrix.num_cols,
         matrix.nnz,
         "cuSPARSE CSC row count exceeds the signed vendor range.",
         "cuSPARSE CSC column count exceeds the signed vendor range.",
         "cuSPARSE CSC nnz exceeds the signed vendor range." );
   config.offsets = matrix.col_ptr;
   config.indices = matrix.row_ind;
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
               cusparseCreateCsc(
                  descriptor,
                  config.rows,
                  config.cols,
                  config.nnz,
                  const_cast< IndexType * >(
                     matrix.col_ptr ),
                  const_cast< IndexType * >(
                     matrix.row_ind ),
                  const_cast< ValueType * >(
                     matrix.values ),
                  CuSparseIndexType< IndexType >(),
                  CuSparseIndexType< IndexType >(),
                  CUSPARSE_INDEX_BASE_ZERO,
                  CuSparseDataType< ValueType >() ),
               "cusparseCreateCsc",
               "Apply(CuSparseCSCBackend, ...)" );
         },
         "Apply(CuSparseCSCBackend, ...)" );
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
   const CuSparseCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCSCApplyDimensions(
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
         CuSparseCSCBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCSCSpMV(
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
      "Apply(CuSparseCSCBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const CuSparseCSCBackend< BackendComputeType > & backend,
   const CSCMatrix< ValueType, IndexType, MatrixBackend > & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckCSCApplyDimensions(
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
         CuSparseCSCBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseCSCSpMV(
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
      "ApplyAdd(CuSparseCSCBackend, ...)" );
}

} // namespace gendil

#endif
