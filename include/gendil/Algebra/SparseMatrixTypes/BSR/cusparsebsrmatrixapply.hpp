// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapplycommon.hpp"
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
CuSparseSpMVState & InitializeCuSparseBSRSpMV(
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

   GENDIL_VERIFY(
      matrix.block_rows == matrix.block_cols,
      "Apply(CuSparseBSRBackend, ...) requires square blocks. "
      "Use NativeDeviceBSRBackend for rectangular BSR matrices." );

   const bool vendor_representation_fits =
      VendorSparseExtentFits( matrix.block_rows ) &&
      VendorSparseExtentFits( matrix.block_cols ) &&
      VendorSparseExtentFits( matrix.num_row_blocks ) &&
      VendorSparseExtentFits( matrix.num_col_blocks ) &&
      VendorSparseExtentFits( matrix.num_blocks ) &&
      VendorSparseProductFits(
         matrix.num_row_blocks,
         matrix.block_rows ) &&
      VendorSparseProductFits(
         matrix.num_col_blocks,
         matrix.block_cols );
   GENDIL_VERIFY(
      vendor_representation_fits,
      "Apply(CuSparseBSRBackend, ...) matrix dimensions do not fit the "
      "cuSPARSE BSR representation. Use NativeDeviceBSRBackend." );

#if !defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR)
   static_assert(
      dependent_false_v< Backend >,
      "Apply(CuSparseBSRBackend, ...) requires generic cuSPARSE BSR SpMV, "
      "which is unavailable in this CUDA toolkit. "
      "Use NativeDeviceBSRBackend." );
   return backend.State();
#else
   auto config = MakeCuSparseSpMVConfig<
      ValueType,
      IndexType,
      InputValue,
      OutputValue,
      ComputeType >(
         VendorSparseFormat::BSR,
         matrix.num_row_blocks,
         matrix.num_col_blocks,
         matrix.num_blocks,
         "cuSPARSE BSR block row count exceeds the signed vendor range.",
         "cuSPARSE BSR block column count exceeds the signed vendor range.",
         "cuSPARSE BSR block count exceeds the signed vendor range." );
   config.block_rows = CheckedVendorSparseExtent(
      matrix.block_rows,
      "cuSPARSE BSR row block size exceeds the signed vendor range." );
   config.block_cols = CheckedVendorSparseExtent(
      matrix.block_cols,
      "cuSPARSE BSR column block size exceeds the signed vendor range." );
   config.offsets = matrix.row_offsets;
   config.indices = matrix.col_indices;
   config.values = matrix.values;
   config.layout =
      MatrixView::block_layout == BlockLayout::RowMajor
         ? static_cast< int >( CUSPARSE_ORDER_ROW )
         : static_cast< int >( CUSPARSE_ORDER_COL );

   return InitializeCuSparseSpMV<
      Backend,
      InputValue,
      OutputValue,
      ComputeType >(
         backend,
         config,
         x_data,
         y_data,
         CUSPARSE_SPMV_BSR_ALG1,
         [&] ( cusparseSpMatDescr_t * descriptor )
         {
            CheckCuSparse(
               cusparseCreateBsr(
                  descriptor,
                  config.rows,
                  config.cols,
                  config.nnz,
                  config.block_rows,
                  config.block_cols,
                  const_cast< IndexType * >(
                     matrix.row_offsets ),
                  const_cast< IndexType * >(
                     matrix.col_indices ),
                  const_cast< ValueType * >(
                     matrix.values ),
                  CuSparseIndexType< IndexType >(),
                  CuSparseIndexType< IndexType >(),
                  CUSPARSE_INDEX_BASE_ZERO,
                  CuSparseDataType< ValueType >(),
                  MatrixView::block_layout == BlockLayout::RowMajor
                     ? CUSPARSE_ORDER_ROW
                     : CUSPARSE_ORDER_COL ),
               "cusparseCreateBsr",
               "Apply(CuSparseBSRBackend, ...)" );
         },
         "Apply(CuSparseBSRBackend, ...)" );
#endif
}

} // namespace details

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void Apply(
   const CuSparseBSRBackend< BackendComputeType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   const GlobalIndex expected_y_size =
      details::CheckBSRApplyDimensions(
         view,
         GetVectorSize( x ),
         GetVectorSize( y ) );
   if ( details::IsEmpty( view ) )
   {
      details::ExecuteEmptySparseSpMV(
         backend,
         WriteDeviceVector( y ),
         expected_y_size );
      return;
   }

   const auto * x_data = ReadDeviceVector( x );
   auto * y_data = WriteDeviceVector( y );
   using InputValue = SparsePointerValueType_t< decltype( x_data ) >;
   using ComputeType =
      ResolveSparseComputeType_t<
         CuSparseBSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseBSRSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteCuSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 0 ),
      CUSPARSE_SPMV_BSR_ALG1,
      "Apply(CuSparseBSRBackend, ...)" );
}

template <
   typename ValueType,
   typename IndexType,
   BlockLayout Layout,
   typename MatrixBackend,
   typename BackendComputeType,
   DeviceAccessibleVector InputVector,
   DeviceAccessibleVector OutputVector >
void ApplyAdd(
   const CuSparseBSRBackend< BackendComputeType > & backend,
   const BSRMatrix<ValueType, IndexType, Layout, MatrixBackend> & matrix,
   const InputVector & x,
   OutputVector & y )
{
   const auto view = GetDeviceReadView( matrix );
   details::CheckBSRApplyDimensions(
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
         CuSparseBSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeCuSparseBSRSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteCuSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 1 ),
      CUSPARSE_SPMV_BSR_ALG1,
      "ApplyAdd(CuSparseBSRBackend, ...)" );
}

} // namespace gendil

#endif
