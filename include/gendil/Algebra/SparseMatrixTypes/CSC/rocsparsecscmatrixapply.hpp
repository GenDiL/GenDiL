// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/CSC/cscmatrixapplycommon.hpp"
#include "gendil/Algebra/SparseMatrixTypes/VendorSparse/rocsparsespmv.hpp"

#if defined(GENDIL_USE_HIP)

namespace gendil
{

namespace details
{

template <
   typename Backend,
   typename MatrixView,
   typename InputValue,
   typename OutputValue >
RocSparseSpMVState & InitializeRocSparseCSCSpMV(
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

   auto config = MakeRocSparseSpMVConfig<
      ValueType,
      IndexType,
      InputValue,
      OutputValue,
      ComputeType >(
         VendorSparseFormat::CSC,
         matrix.num_rows,
         matrix.num_cols,
         matrix.nnz,
         "rocSPARSE CSC row count exceeds the signed vendor range.",
         "rocSPARSE CSC column count exceeds the signed vendor range.",
         "rocSPARSE CSC nnz exceeds the signed vendor range." );
   config.offsets = matrix.col_ptr;
   config.indices = matrix.row_ind;
   config.values = matrix.values;

   return InitializeRocSparseSpMV<
      Backend,
      InputValue,
      OutputValue,
      ComputeType >(
         backend,
         config,
         x_data,
         y_data,
         rocsparse_spmv_alg_default,
         [&] ( rocsparse_spmat_descr * descriptor )
         {
            CheckRocSparse(
               rocsparse_create_csc_descr(
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
                  RocSparseIndexType< IndexType >(),
                  RocSparseIndexType< IndexType >(),
                  rocsparse_index_base_zero,
                  RocSparseDataType< ValueType >() ),
               "rocsparse_create_csc_descr",
               "Apply(RocSparseCSCBackend, ...)" );
         },
         "Apply(RocSparseCSCBackend, ...)" );
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
   const RocSparseCSCBackend< BackendComputeType > & backend,
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
         RocSparseCSCBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseCSCSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteRocSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 0 ),
      rocsparse_spmv_alg_default,
      "Apply(RocSparseCSCBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const RocSparseCSCBackend< BackendComputeType > & backend,
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
         RocSparseCSCBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseCSCSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteRocSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 1 ),
      rocsparse_spmv_alg_default,
      "ApplyAdd(RocSparseCSCBackend, ...)" );
}

} // namespace gendil

#endif
