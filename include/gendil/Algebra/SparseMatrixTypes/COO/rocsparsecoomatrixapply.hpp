// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/COO/coomatrixapplycommon.hpp"
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
RocSparseSpMVState & InitializeRocSparseCOOSpMV(
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
         VendorSparseFormat::COO,
         matrix.num_rows,
         matrix.num_cols,
         matrix.nnz,
         "rocSPARSE COO row count exceeds the signed vendor range.",
         "rocSPARSE COO column count exceeds the signed vendor range.",
         "rocSPARSE COO nnz exceeds the signed vendor range." );
   config.indices = matrix.rows;
   config.secondary_indices = matrix.cols;
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
               rocsparse_create_coo_descr(
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
                  RocSparseIndexType< IndexType >(),
                  rocsparse_index_base_zero,
                  RocSparseDataType< ValueType >() ),
               "rocsparse_create_coo_descr",
               "Apply(RocSparseCOOBackend, ...)" );
         },
         "Apply(RocSparseCOOBackend, ...)" );
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
   const RocSparseCOOBackend< BackendComputeType > & backend,
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
         RocSparseCOOBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseCOOSpMV(
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
      "Apply(RocSparseCOOBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const RocSparseCOOBackend< BackendComputeType > & backend,
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
         RocSparseCOOBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseCOOSpMV(
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
      "ApplyAdd(RocSparseCOOBackend, ...)" );
}

} // namespace gendil

#endif
