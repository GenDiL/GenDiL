// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/SparseMatrixTypes/BSR/bsrmatrixapplycommon.hpp"
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
RocSparseSpMVState & InitializeRocSparseBSRSpMV(
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
      "Apply(RocSparseBSRBackend, ...) requires square blocks. "
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
      "Apply(RocSparseBSRBackend, ...) matrix dimensions do not fit the "
      "rocSPARSE BSR representation. Use NativeDeviceBSRBackend." );

#if !defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   static_assert(
      dependent_false_v< Backend >,
      "Apply(RocSparseBSRBackend, ...) requires generic rocSPARSE BSR SpMV, "
      "which is unavailable in this ROCm toolkit. "
      "Use NativeDeviceBSRBackend." );
   return backend.State();
#else
   auto config = MakeRocSparseSpMVConfig<
      ValueType,
      IndexType,
      InputValue,
      OutputValue,
      ComputeType >(
         VendorSparseFormat::BSR,
         matrix.num_row_blocks,
         matrix.num_col_blocks,
         matrix.num_blocks,
         "rocSPARSE BSR block row count exceeds the signed vendor range.",
         "rocSPARSE BSR block column count exceeds the signed vendor range.",
         "rocSPARSE BSR block count exceeds the signed vendor range." );
   config.block_rows = CheckedVendorSparseExtent(
      matrix.block_rows,
      "rocSPARSE BSR block size exceeds the signed vendor range." );
   config.block_cols = CheckedVendorSparseExtent(
      matrix.block_cols,
      "rocSPARSE BSR block size exceeds the signed vendor range." );
   config.offsets = matrix.row_offsets;
   config.indices = matrix.col_indices;
   config.values = matrix.values;
   config.layout =
      MatrixView::block_layout == BlockLayout::RowMajor
         ? static_cast< int >( rocsparse_direction_row )
         : static_cast< int >( rocsparse_direction_column );

   return InitializeRocSparseSpMV<
      Backend,
      InputValue,
      OutputValue,
      ComputeType >(
         backend,
         config,
         x_data,
         y_data,
         rocsparse_spmv_alg_bsr,
         [&] ( rocsparse_spmat_descr * descriptor )
         {
            CheckRocSparse(
               rocsparse_create_bsr_descr(
                  descriptor,
                  config.rows,
                  config.cols,
                  config.nnz,
                  MatrixView::block_layout == BlockLayout::RowMajor
                     ? rocsparse_direction_row
                     : rocsparse_direction_column,
                  config.block_rows,
                  const_cast< IndexType * >(
                     matrix.row_offsets ),
                  const_cast< IndexType * >(
                     matrix.col_indices ),
                  const_cast< ValueType * >(
                     matrix.values ),
                  RocSparseIndexType< IndexType >(),
                  RocSparseIndexType< IndexType >(),
                  rocsparse_index_base_zero,
                  RocSparseDataType< ValueType >() ),
               "rocsparse_create_bsr_descr",
               "Apply(RocSparseBSRBackend, ...)" );
         },
         "Apply(RocSparseBSRBackend, ...)" );
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
   const RocSparseBSRBackend< BackendComputeType > & backend,
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
         RocSparseBSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseBSRSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteRocSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 0 ),
      rocsparse_spmv_alg_bsr,
      "Apply(RocSparseBSRBackend, ...)" );
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
   const RocSparseBSRBackend< BackendComputeType > & backend,
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
         RocSparseBSRBackend< BackendComputeType >,
         ValueType,
         InputValue >;
   auto & initialized = details::InitializeRocSparseBSRSpMV(
      backend,
      view,
      x_data,
      y_data );
   details::ExecuteRocSparseSpMV(
      backend,
      initialized,
      ComputeType( 1 ),
      ComputeType( 1 ),
      rocsparse_spmv_alg_bsr,
      "ApplyAdd(RocSparseBSRBackend, ...)" );
}

} // namespace gendil

#endif
