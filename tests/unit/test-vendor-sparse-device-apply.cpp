// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   return 0;
}

#else

using namespace gendil;

namespace
{

// BSR default is capability-aware
#if \
   defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
   defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
using ExpectedBSRBackend = VendorDeviceBSRBackend<>;
#else
using ExpectedBSRBackend = NativeDeviceBSRBackend<>;
#endif

static_assert(
   std::is_same_v< DefaultBSRBackend, ExpectedBSRBackend > );
static_assert(
   std::is_same_v< DefaultCOOBackend, VendorDeviceCOOBackend<> > );
static_assert(
   std::is_same_v< DefaultCSRBackend, VendorDeviceCSRBackend<> > );
static_assert(
   std::is_same_v< DefaultCSCBackend, VendorDeviceCSCBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::BSR >,
      ExpectedBSRBackend > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::SGBSR >,
      ExpectedBSRBackend > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::COO >,
      VendorDeviceCOOBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSR >,
      VendorDeviceCSRBackend<> > );
static_assert(
   std::is_same_v<
      DefaultBackendFor_t< MatrixAssemblyType::CSC >,
      VendorDeviceCSCBackend<> > );

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool VectorsNear( const Vector & lhs, const Vector & rhs )
{
   if ( lhs.Size() != rhs.Size() )
   {
      return false;
   }

   const Real * lhs_data = lhs.ReadHostData();
   const Real * rhs_data = rhs.ReadHostData();
   for ( size_t i = 0; i < lhs.Size(); ++i )
   {
      if ( std::abs( lhs_data[i] - rhs_data[i] ) > 1.0e-11 )
      {
         return false;
      }
   }
   return true;
}

void FillInput( Vector & x, const Real shift = 0.0 )
{
   Real * data = x.WriteHostData();
   for ( size_t i = 0; i < x.Size(); ++i )
   {
      data[i] = shift + Real( i + 1 );
   }
}

void FillOutput( Vector & y, const Real shift = 0.0 )
{
   Real * data = y.WriteHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      data[i] = shift + Real( 2 * i + 1 );
   }
}

#ifdef GENDIL_USE_MFEM
template < typename HostBackend, typename DeviceBackend, typename Matrix >
bool CheckMFEMApplyAdd(
   const HostBackend & host_backend,
   const DeviceBackend & device_backend,
   const Matrix & matrix,
   const int input_size,
   const int output_size,
   const char * message )
{
   mfem::Vector x( input_size );
   Real * x_data = x.HostWrite();
   for ( int i = 0; i < input_size; ++i )
   {
      x_data[i] = Real( i + 1 );
   }

   mfem::Vector host( output_size );
   mfem::Vector device( output_size );
   Real * host_data = host.HostWrite();
   Real * device_data = device.HostWrite();
   for ( int i = 0; i < output_size; ++i )
   {
      host_data[i] = Real( 2 * i + 1 );
      device_data[i] = Real( 2 * i + 1 );
   }

   ApplyAdd( host_backend, matrix, x, host );
   ApplyAdd( device_backend, matrix, x, device );
   host_data = host.HostReadWrite();
   device_data = device.HostReadWrite();

   bool success = true;
   for ( int i = 0; i < output_size; ++i )
   {
      success = Check(
         std::abs( host_data[i] - device_data[i] ) < 1.0e-11,
         message ) && success;
   }
   return success;
}

template <
   typename InputVector,
   typename OutputVector,
   typename HostBackend,
   typename DeviceBackend,
   typename Matrix >
bool CheckMixedVectorApply(
   const HostBackend & host_backend,
   const DeviceBackend & device_backend,
   const Matrix & matrix,
   const int input_size,
   const int output_size,
   const char * message )
{
   InputVector x( input_size );
   auto * x_data = WriteHostVector( x );
   for ( int i = 0; i < input_size; ++i )
   {
      x_data[i] = Real( i + 1 );
   }

   auto initialize_output = [=] ( OutputVector & y )
   {
      auto * data = WriteHostVector( y );
      for ( int i = 0; i < output_size; ++i )
      {
         data[i] = Real( 2 * i + 1 );
      }
   };
   auto compare = [&] ( const OutputVector & host, const OutputVector & device )
   {
      const auto * host_data = ReadHostVector( host );
      const auto * device_data = ReadHostVector( device );
      bool result = true;
      for ( int i = 0; i < output_size; ++i )
      {
         result = Check(
            std::abs( host_data[i] - device_data[i] ) < 1.0e-11,
            message ) && result;
      }
      return result;
   };

   OutputVector host( output_size );
   OutputVector device( output_size );
   Apply( host_backend, matrix, x, host );
   Apply( device_backend, matrix, x, device );
   bool success = compare( host, device );

   initialize_output( host );
   initialize_output( device );
   ApplyAdd( host_backend, matrix, x, host );
   ApplyAdd( device_backend, matrix, x, device );
   success = compare( host, device ) && success;
   return success;
}
#endif

bool TestCOO()
{
   auto matrix = MakeCOOMatrix< Real, GlobalIndex >( 4, 5, 6 );
   auto matrix_data = GetHostWriteView( matrix );
   const GlobalIndex rows[] = { 0, 0, 1, 1, 1, 3 };
   const GlobalIndex cols[] = { 0, 3, 1, 2, 4, 0 };
   const Real values[] = { 2.0, -1.0, 4.0, 0.0, 1.5, -3.0 };
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix_data.rows[i] = rows[i];
      matrix_data.cols[i] = cols[i];
      matrix_data.values[i] = values[i];
   }
   Sync( matrix );

   Vector x( 5 );
   Vector host( 4 );
   Vector native( 4 );
   Vector explicit_vendor( 4 );
   Vector stored_default( 4 );
   FillInput( x );

   VendorDeviceCOOBackend<> vendor;
   Apply( HostCOOBackend<>{}, matrix, x, host );
   Apply( NativeDeviceCOOBackend<>{}, matrix, x, native );
   Apply( vendor, matrix, x, explicit_vendor );
   matrix( x, stored_default );

   bool success = true;
   success = Check(
      VectorsNear( host, native ),
      "Native COO result differs from host." ) && success;
   success = Check(
      VectorsNear( host, explicit_vendor ),
      "Vendor COO result differs from host." ) && success;
   success = Check(
      VectorsNear( host, stored_default ),
      "Default COO result differs from host." ) && success;
   success = Check(
      vendor.LastExecutionPath() == VendorSparseExecutionPath::Vendor,
      "Nonempty COO did not select the vendor path." ) && success;
   success = Check(
      vendor.HasCachedPlan(),
      "Vendor COO backend did not retain its warm plan." ) && success;

   const auto cached_matrix = vendor.State().matrix;
   Vector host_add( 4 );
   Vector vendor_add( 4 );
   FillOutput( host_add, 0.5 );
   FillOutput( vendor_add, 0.5 );
   ApplyAdd( HostCOOBackend<>{}, matrix, x, host_add );
   ApplyAdd( vendor, matrix, x, vendor_add );
   success = Check(
      VectorsNear( host_add, vendor_add ),
      "Vendor COO ApplyAdd differs from host." ) && success;
   success = Check(
      vendor.HasCachedPlan() &&
      vendor.State().matrix == cached_matrix,
      "Alternating COO Apply and ApplyAdd rebuilt the initialized plan." ) &&
      success;

#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      HostCOOBackend<>{},
      NativeDeviceCOOBackend<>{},
      matrix,
      5,
      4,
      "Native COO MFEM ApplyAdd differs from host." ) && success;
   success = CheckMFEMApplyAdd(
      HostCOOBackend<>{},
      vendor,
      matrix,
      5,
      4,
      "Vendor COO MFEM ApplyAdd differs from host." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      HostCOOBackend<>{},
      NativeDeviceCOOBackend<>{},
      matrix,
      5,
      4,
      "Native COO mixed-vector apply differs from host." ) && success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      HostCOOBackend<>{},
      vendor,
      matrix,
      5,
      4,
      "Vendor COO mixed-vector apply differs from host." ) && success;
#endif

   const auto copied_vendor = vendor;
   success = Check(
      !copied_vendor.HasCachedPlan(),
      "Copying a vendor backend unexpectedly shared its mutable plan." ) &&
      success;

   Vector x_second( 5 );
   Vector host_second( 4 );
   Vector vendor_second( 4 );
   FillInput( x_second, 0.25 );
   Apply( HostCOOBackend<>{}, matrix, x_second, host_second );
   Apply( vendor, matrix, x_second, vendor_second );
   success = Check(
      VectorsNear( host_second, vendor_second ),
      "Warm-cache COO apply with new vectors differs from host." ) && success;

   success = Check(
      matrix.backend.HasCachedPlan(),
      "Stored COO backend did not retain its warm plan before a move." ) &&
      success;
   auto moved_matrix = std::move( matrix );
   success = Check(
      moved_matrix.backend.HasCachedPlan() &&
      !matrix.backend.HasCachedPlan(),
      "COO move construction did not transfer the cached vendor plan." ) &&
      success;

   auto destination = MakeCOOMatrix< Real, GlobalIndex >( 1, 1, 1 );
   destination = std::move( moved_matrix );
   success = Check(
      destination.backend.HasCachedPlan() &&
      !moved_matrix.backend.HasCachedPlan(),
      "COO move assignment did not transfer the cached vendor plan." ) &&
      success;

   Vector moved_result( 4 );
   destination( x_second, moved_result );
   success = Check(
      VectorsNear( host_second, moved_result ),
      "Move-assigned COO matrix with a cached vendor plan is incorrect." ) &&
      success;

   vendor.ResetState();
   return success;
}

template < typename IndexType, typename ValueType = Real >
bool TestCSR()
{
   auto matrix =
      MakeCSRMatrix<
         ValueType,
         IndexType,
         VendorDeviceCSRBackend<> >( 4, 5, 6 );
   auto matrix_data = GetHostWriteView( matrix );
   const IndexType row_ptr[] = { 0, 2, 5, 5, 6 };
   const IndexType col_ind[] = { 0, 3, 1, 2, 4, 0 };
   const ValueType values[] = {
      ValueType( 2.0 ),
      ValueType( -1.0 ),
      ValueType( 4.0 ),
      ValueType( 0.0 ),
      ValueType( 1.5 ),
      ValueType( -3.0 ) };
   for ( IndexType i = 0; i < IndexType( 5 ); ++i )
   {
      matrix_data.row_ptr[i] = row_ptr[i];
   }
   for ( IndexType i = 0; i < matrix.nnz; ++i )
   {
      matrix_data.col_ind[i] = col_ind[i];
      matrix_data.values[i] = values[i];
   }
   Sync( matrix );

   Vector x( 5 );
   Vector host( 4 );
   Vector native( 4 );
   Vector vendor_result( 4 );
   Vector stored_default( 4 );
   FillInput( x );

   VendorDeviceCSRBackend<> vendor;
   Apply( HostCSRBackend<>{}, matrix, x, host );
   Apply( NativeDeviceCSRBackend<>{}, matrix, x, native );
   Apply( vendor, matrix, x, vendor_result );
   matrix( x, stored_default );

   bool success = true;
   success = Check(
      VectorsNear( host, native ),
      "Native CSR result differs from host." ) && success;
   success = Check(
      VectorsNear( host, vendor_result ),
      "Vendor CSR result differs from host." ) && success;
   success = Check(
      VectorsNear( host, stored_default ),
      "Default CSR result differs from host." ) && success;

   const auto cached_matrix = vendor.State().matrix;
   Vector host_add( 4 );
   Vector vendor_add( 4 );
   FillOutput( host_add, -0.25 );
   FillOutput( vendor_add, -0.25 );
   ApplyAdd( HostCSRBackend<>{}, matrix, x, host_add );
   ApplyAdd( vendor, matrix, x, vendor_add );
   success = Check(
      VectorsNear( host_add, vendor_add ),
      "Vendor CSR ApplyAdd differs from host." ) && success;
   success = Check(
      vendor.HasCachedPlan() &&
      vendor.State().matrix == cached_matrix,
      "Alternating CSR Apply and ApplyAdd rebuilt the initialized plan." ) &&
      success;

#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      HostCSRBackend<>{},
      NativeDeviceCSRBackend<>{},
      matrix,
      5,
      4,
      "Native CSR MFEM ApplyAdd differs from host." ) && success;
   success = CheckMFEMApplyAdd(
      HostCSRBackend<>{},
      vendor,
      matrix,
      5,
      4,
      "Vendor CSR MFEM ApplyAdd differs from host." ) && success;
   if constexpr ( std::is_same_v< ValueType, Real > )
   {
      success = CheckMixedVectorApply< Vector, mfem::Vector >(
         HostCSRBackend<>{},
         NativeDeviceCSRBackend<>{},
         matrix,
         5,
         4,
         "Native CSR mixed-vector apply differs from host." ) && success;
      success = CheckMixedVectorApply< mfem::Vector, Vector >(
         HostCSRBackend<>{},
         vendor,
         matrix,
         5,
         4,
         "Vendor CSR mixed-vector apply differs from host." ) && success;
   }
#endif

   GetHostValuesReadWriteView( matrix ).values[0] = ValueType( 3.5 );
   success = Check(
      matrix.backend.HasCachedPlan(),
      "A CSR values-only mutation invalidated the stored vendor plan." ) &&
      success;
   Apply( HostCSRBackend<>{}, matrix, x, host );
   Apply( vendor, matrix, x, vendor_result );
   success = Check(
      VectorsNear( host, vendor_result ),
      "Warm-cache CSR apply after a values update differs from host." ) &&
      success;

   (void) GetHostReadWriteView( matrix ).col_ind;
   success = Check(
      matrix.backend.HasCachedPlan() && vendor.HasCachedPlan(),
      "Acquiring a mutable CSR view invalidated a vendor plan." ) &&
      success;

   ResetState( matrix.backend );
   ResetState( vendor );
   success = Check(
      !matrix.backend.HasCachedPlan() && !vendor.HasCachedPlan(),
      "Explicit CSR vendor backend reset retained a cached plan." ) &&
      success;

   GetHostReadWriteView( matrix ).col_ind[0] = IndexType( 1 );
   Apply( HostCSRBackend<>{}, matrix, x, host );
   matrix( x, stored_default );
   Apply( vendor, matrix, x, vendor_result );
   success = Check(
      VectorsNear( host, stored_default ) &&
      VectorsNear( host, vendor_result ),
      "CSR apply after explicit structure-cache invalidation differs from host." ) &&
      success;
   success = Check(
      matrix.backend.HasCachedPlan() && vendor.HasCachedPlan(),
      "CSR apply did not rebuild explicitly invalidated vendor plans." ) &&
      success;

   vendor.ResetState();
   return success;
}

bool TestCSC()
{
   auto matrix = MakeCSCMatrix< Real, GlobalIndex >( 4, 5, 6 );
   auto matrix_data = GetHostWriteView( matrix );
   const GlobalIndex col_ptr[] = { 0, 2, 3, 4, 5, 6 };
   const GlobalIndex row_ind[] = { 0, 3, 1, 1, 0, 1 };
   const Real values[] = { 2.0, -3.0, 4.0, 0.0, -1.0, 1.5 };
   for ( GlobalIndex i = 0; i < matrix.num_cols + 1; ++i )
   {
      matrix_data.col_ptr[i] = col_ptr[i];
   }
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      matrix_data.row_ind[i] = row_ind[i];
      matrix_data.values[i] = values[i];
   }
   Sync( matrix );

   Vector x( 5 );
   Vector host( 4 );
   Vector native( 4 );
   Vector vendor_result( 4 );
   Vector stored_default( 4 );
   FillInput( x );

   VendorDeviceCSCBackend<> vendor;
   Apply( HostCSCBackend<>{}, matrix, x, host );
   Apply( NativeDeviceCSCBackend<>{}, matrix, x, native );
   Apply( vendor, matrix, x, vendor_result );
   matrix( x, stored_default );

   bool success =
      Check(
         VectorsNear( host, native ),
         "Native CSC result differs from host." ) &&
      Check(
         VectorsNear( host, vendor_result ),
         "Vendor CSC result differs from host." ) &&
      Check(
         VectorsNear( host, stored_default ),
         "Default CSC result differs from host." ) &&
      Check(
         vendor.LastExecutionPath() == VendorSparseExecutionPath::Vendor,
         "Nonempty CSC did not select the vendor path." );

   const auto cached_matrix = vendor.State().matrix;
   Vector host_add( 4 );
   Vector vendor_add( 4 );
   FillOutput( host_add, 1.25 );
   FillOutput( vendor_add, 1.25 );
   ApplyAdd( HostCSCBackend<>{}, matrix, x, host_add );
   ApplyAdd( vendor, matrix, x, vendor_add );
   success = Check(
      VectorsNear( host_add, vendor_add ),
      "Vendor CSC ApplyAdd differs from host." ) && success;
   success = Check(
      vendor.HasCachedPlan() &&
      vendor.State().matrix == cached_matrix,
      "Alternating CSC Apply and ApplyAdd rebuilt the initialized plan." ) &&
      success;

#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      HostCSCBackend<>{},
      NativeDeviceCSCBackend<>{},
      matrix,
      5,
      4,
      "Native CSC MFEM ApplyAdd differs from host." ) && success;
   success = CheckMFEMApplyAdd(
      HostCSCBackend<>{},
      vendor,
      matrix,
      5,
      4,
      "Vendor CSC MFEM ApplyAdd differs from host." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      HostCSCBackend<>{},
      NativeDeviceCSCBackend<>{},
      matrix,
      5,
      4,
      "Native CSC mixed-vector apply differs from host." ) && success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      HostCSCBackend<>{},
      vendor,
      matrix,
      5,
      4,
      "Vendor CSC mixed-vector apply differs from host." ) && success;
#endif

   vendor.ResetState();
   return success;
}

template <
   BlockLayout Layout,
   typename Backend = VendorDeviceBSRBackend<> >
auto MakeBSR(
   const GlobalIndex block_rows,
   const GlobalIndex block_cols,
   Backend backend = Backend{} )
{
   BSRMatrix<
      Real,
      GlobalIndex,
      Layout,
      Backend > matrix{};
   matrix.backend = std::move( backend );
   matrix.block_rows = block_rows;
   matrix.block_cols = block_cols;
   matrix.num_row_blocks = 2;
   matrix.num_col_blocks = 2;
   matrix.num_blocks = 2;
   ConfigureBSRBackend(
      matrix.backend,
      matrix.block_rows,
      matrix.block_cols,
      matrix.num_row_blocks,
      matrix.num_col_blocks,
      matrix.num_blocks );

   matrix.row_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 3 ) );
   matrix.col_indices =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 2 ) );
   matrix.values =
      MakeSyncHostDeviceArray< Real >(
         2 * block_rows * block_cols );

   auto matrix_data = GetHostWriteView( matrix );
   matrix_data.row_offsets[0] = 0;
   matrix_data.row_offsets[1] = 1;
   matrix_data.row_offsets[2] = 2;
   matrix_data.col_indices[0] = 0;
   matrix_data.col_indices[1] = 1;
   for ( GlobalIndex i = 0;
         i < 2 * block_rows * block_cols;
         ++i )
   {
      matrix_data.values[i] = Real( i + 1 ) * 0.25;
   }

   Sync( matrix );
   return matrix;
}

template < BlockLayout Layout >
bool TestSquareBSR()
{
   auto matrix = MakeBSR< Layout >( 2, 2 );
   Vector x( 4 );
   Vector host( 4 );
   Vector native( 4 );
   Vector stored_default( 4 );
   FillInput( x );

   Apply( HostBSRBackend<>{}, matrix, x, host );
   Apply( NativeDeviceBSRBackend<>{}, matrix, x, native );
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
    defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   matrix( x, stored_default );
#else
   Apply(
      NativeDeviceBSRBackend<>{},
      matrix,
      x,
      stored_default );
#endif

   bool success = true;
   success = Check(
      matrix.backend.AssembledSquareBlocks(),
      "Square BSR shape was not recorded during storage configuration." ) &&
      success;
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
    defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   success = Check(
      matrix.backend.AssembledVendorEligible(),
      "Representable square BSR storage was not marked vendor-eligible." ) &&
      success;
#else
   success = Check(
      !matrix.backend.AssembledVendorEligible(),
      "BSR storage was marked vendor-eligible without toolkit support." ) &&
      success;
#endif
   success = Check(
      VectorsNear( host, native ),
      "Native square BSR result differs from host." ) && success;
   success = Check(
      VectorsNear( host, stored_default ),
      "Default square BSR result differs from host." ) && success;

#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
    defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   success = Check(
      matrix.backend.LastExecutionPath() ==
         VendorSparseExecutionPath::Vendor,
      "Vendor-compatible square BSR did not select the vendor path." ) &&
      success;

   const auto cached_matrix = matrix.backend.State().matrix;
   Vector host_add( 4 );
   Vector vendor_add( 4 );
   FillOutput( host_add, -1.5 );
   FillOutput( vendor_add, -1.5 );
   ApplyAdd( HostBSRBackend<>{}, matrix, x, host_add );
   ApplyAdd( matrix, x, vendor_add );
   success = Check(
      VectorsNear( host_add, vendor_add ),
      "Vendor BSR ApplyAdd differs from host." ) && success;
   success = Check(
      matrix.backend.HasCachedPlan() &&
      matrix.backend.State().matrix == cached_matrix,
      "Alternating BSR Apply and ApplyAdd rebuilt the initialized plan." ) &&
      success;
#ifdef GENDIL_USE_MFEM
   success = CheckMFEMApplyAdd(
      HostBSRBackend<>{},
      NativeDeviceBSRBackend<>{},
      matrix,
      4,
      4,
      "Native BSR MFEM ApplyAdd differs from host." ) && success;
   success = CheckMFEMApplyAdd(
      HostBSRBackend<>{},
      matrix.backend,
      matrix,
      4,
      4,
      "Vendor BSR MFEM ApplyAdd differs from host." ) && success;
   success = CheckMixedVectorApply< Vector, mfem::Vector >(
      HostBSRBackend<>{},
      NativeDeviceBSRBackend<>{},
      matrix,
      4,
      4,
      "Native BSR mixed-vector apply differs from host." ) && success;
   success = CheckMixedVectorApply< mfem::Vector, Vector >(
      HostBSRBackend<>{},
      matrix.backend,
      matrix,
      4,
      4,
      "Vendor BSR mixed-vector apply differs from host." ) && success;
#endif
#endif

   return success;
}

bool TestEmptyBSR()
{
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
    defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   using EmptyBSRBackend = DefaultBSRBackend;
#else
   using EmptyBSRBackend = NativeDeviceBSRBackend<>;
#endif
   BSRMatrix<
      Real,
      GlobalIndex,
      BlockLayout::RowMajor,
      EmptyBSRBackend > matrix{};
   matrix.block_rows = 1;
   matrix.block_cols = 1;
   matrix.num_row_blocks = 2;
   matrix.num_col_blocks = 2;
   matrix.num_blocks = 0;
   matrix.row_offsets =
      MakeSyncHostDeviceArray< GlobalIndex >( GlobalIndex( 3 ) );
   auto matrix_data = GetHostWriteView( matrix );
   matrix_data.row_offsets[0] = 0;
   matrix_data.row_offsets[1] = 0;
   matrix_data.row_offsets[2] = 0;

   Vector x( 2 );
   Vector y( 2 );
   FillInput( x );
   y = 9.0;
   matrix( x, y );

   bool success = true;
   const Real * data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 0.0,
         "Empty vendor BSR did not overwrite the output with zero." ) &&
         success;
   }
#if defined(GENDIL_CUSPARSE_HAS_GENERIC_BSR) || \
    defined(GENDIL_ROCSPARSE_HAS_GENERIC_BSR)
   success = Check(
      matrix.backend.LastExecutionPath() ==
         VendorSparseExecutionPath::Trivial,
      "Empty vendor BSR did not select the trivial zero path." ) &&
      success;
   y = 7.0;
   ApplyAdd( matrix, x, y );
   data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 7.0,
         "Empty vendor BSR ApplyAdd changed the output." ) && success;
   }
   success = Check(
      !matrix.backend.HasCachedPlan(),
      "Empty vendor BSR ApplyAdd initialized a vendor plan." ) && success;
#endif
   return success;
}

bool TestRectangularNativeBSR()
{
   auto matrix =
      MakeBSR<
         BlockLayout::RowMajor,
         NativeDeviceBSRBackend<> >( 2, 3 );
   Vector x( 6 );
   Vector host( 4 );
   Vector native( 4 );
   FillInput( x );
   Apply( HostBSRBackend<>{}, matrix, x, host );
   matrix( x, native );

   const bool success =
      Check(
         VectorsNear( host, native ),
         "Rectangular native BSR result differs from host." );
   return success;
}

bool TestEmptyCOO()
{
   auto matrix = MakeCOOMatrix< Real, GlobalIndex >( 3, 2, 0 );
   Vector x( 2 );
   Vector y( 3 );
   FillInput( x );
   y = 9.0;
   matrix( x, y );

   bool success =
      Check(
         matrix.backend.LastExecutionPath() ==
            VendorSparseExecutionPath::Trivial,
         "Empty COO did not select the trivial zero path." );
   const Real * data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 0.0,
         "Empty COO did not overwrite the output with zero." ) &&
         success;
   }

   y = 7.0;
   ApplyAdd( matrix, x, y );
   data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 7.0,
         "Empty COO ApplyAdd changed the output." ) && success;
   }
   success = Check(
      !matrix.backend.HasCachedPlan(),
      "Empty COO ApplyAdd initialized a vendor plan." ) && success;

   return success;
}

bool TestEmptyCSR()
{
   auto matrix = MakeCSRMatrix< Real, GlobalIndex >( 3, 2, 0 );
   auto matrix_data = GetHostWriteView( matrix );
   for ( GlobalIndex row = 0; row < 4; ++row )
   {
      matrix_data.row_ptr[row] = 0;
   }

   Vector x( 2 );
   Vector y( 3 );
   FillInput( x );
   y = 9.0;
   matrix( x, y );

   bool success =
      Check(
         matrix.backend.LastExecutionPath() ==
            VendorSparseExecutionPath::Trivial,
         "Empty CSR did not select the trivial zero path." );
   const Real * data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 0.0,
         "Empty CSR did not overwrite the output with zero." ) && success;
   }

   y = 7.0;
   ApplyAdd( matrix, x, y );
   data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 7.0,
         "Empty CSR ApplyAdd changed the output." ) && success;
   }
   success = Check(
      !matrix.backend.HasCachedPlan(),
      "Empty CSR ApplyAdd initialized a vendor plan." ) && success;

   return success;
}

bool TestEmptyCSC()
{
   auto matrix = MakeCSCMatrix< Real, GlobalIndex >( 3, 2, 0 );
   auto matrix_data = GetHostWriteView( matrix );
   for ( GlobalIndex col = 0; col < 3; ++col )
   {
      matrix_data.col_ptr[col] = 0;
   }
   Vector x( 2 );
   Vector y( 3 );
   FillInput( x );
   y = 9.0;
   matrix( x, y );

   bool success =
      Check(
         matrix.backend.LastExecutionPath() ==
            VendorSparseExecutionPath::Trivial,
         "Empty CSC did not select the trivial zero path." );
   const Real * data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 0.0,
         "Empty CSC did not overwrite the output with zero." ) &&
         success;
   }

   y = 7.0;
   ApplyAdd( matrix, x, y );
   data = y.ReadHostData();
   for ( size_t i = 0; i < y.Size(); ++i )
   {
      success = Check(
         data[i] == 7.0,
         "Empty CSC ApplyAdd changed the output." ) && success;
   }
   success = Check(
      !matrix.backend.HasCachedPlan(),
      "Empty CSC ApplyAdd initialized a vendor plan." ) && success;

   return success;
}

} // namespace

int main()
{
#if defined(GENDIL_USE_MFEM)
  #if defined(GENDIL_USE_CUDA)
   mfem::Device device("cuda");
  #elif defined(GENDIL_USE_HIP)
   mfem::Device device("hip");
  #endif
#endif

   bool success = true;
   success = TestCOO() && success;
   success = TestCSR< GlobalIndex >() && success;
#if defined(GENDIL_CUSPARSE_HAS_FLOAT_DOUBLE_SPMV)
   success = TestCSR< std::uint32_t, float >() && success;
#endif
   success = TestCSC() && success;
   success = TestSquareBSR< BlockLayout::ColumnMajor >() && success;
   success = TestSquareBSR< BlockLayout::RowMajor >() && success;
   success = TestEmptyBSR() && success;
   success = TestRectangularNativeBSR() && success;
   success = TestEmptyCOO() && success;
   success = TestEmptyCSR() && success;
   success = TestEmptyCSC() && success;
   return success ? 0 : 1;
}

#endif
