// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

[[maybe_unused]] bool Near( const Real lhs, const Real rhs )
{
   return std::abs( lhs - rhs ) < tolerance;
}

auto MakeFixture()
{
   constexpr std::array< GlobalIndex, 14 > rows{
      3, 0, 1, 0, 3, 1, 4, 3, 0, 1, 3, 2, 4, 4 };
   constexpr std::array< GlobalIndex, 14 > cols{
      5, 4, 2, 4, 1, 2, 0, 3, 0, 0, 1, 2, 4, 4 };
   constexpr std::array< Real, 14 > values{
      2.0, 1.0, 3.0, -1.0, 4.0, 2.0, 0.0,
      7.0, 6.0, 1.0, -1.0, 8.0, 4.0, 5.0 };

   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >(
      6,
      7,
      rows.size() );
   auto view = GetHostWriteView( raw );
   for ( GlobalIndex i = 0; i < raw.nnz_raw; ++i )
   {
      view.rows[i] = rows[static_cast< size_t >( i )];
      view.cols[i] = cols[static_cast< size_t >( i )];
      view.values[i] = values[static_cast< size_t >( i )];
   }
   return raw;
}

template < typename T, typename SizeType >
bool HostOnly( const SyncHostDeviceArray< T, SizeType > & array )
{
   return IsHostValid( array ) && !IsDeviceValid( array );
}

#if defined(GENDIL_USE_DEVICE)
template < typename T, typename SizeType >
bool DeviceOnly( const SyncHostDeviceArray< T, SizeType > & array )
{
   return !IsHostValid( array ) && IsDeviceValid( array );
}
#endif

template < typename ActualBackend, typename ExpectedBackend >
bool CompareCOO(
   const COOMatrix< Real, GlobalIndex, ActualBackend > & actual,
   const COOMatrix< Real, GlobalIndex, ExpectedBackend > & expected )
{
   bool success = Check(
      actual.num_rows == expected.num_rows &&
         actual.num_cols == expected.num_cols && actual.nnz == expected.nnz,
      "Device COO dimensions or nnz differ from host finalization." );
   const auto actual_view = GetHostReadView( actual );
   const auto expected_view = GetHostReadView( expected );
   for ( GlobalIndex i = 0; i < actual.nnz; ++i )
   {
      success = Check(
         actual_view.rows[i] == expected_view.rows[i] &&
            actual_view.cols[i] == expected_view.cols[i] &&
            Near( actual_view.values[i], expected_view.values[i] ),
         "Device COO payload differs from host finalization." ) && success;
   }
   return success;
}

template < typename ActualBackend, typename ExpectedBackend >
bool CompareCSR(
   const CSRMatrix< Real, GlobalIndex, ActualBackend > & actual,
   const CSRMatrix< Real, GlobalIndex, ExpectedBackend > & expected )
{
   bool success = Check(
      actual.num_rows == expected.num_rows &&
         actual.num_cols == expected.num_cols && actual.nnz == expected.nnz,
      "Device CSR dimensions or nnz differ from host finalization." );
   const auto actual_view = GetHostReadView( actual );
   const auto expected_view = GetHostReadView( expected );
   for ( GlobalIndex row = 0; row <= actual.num_rows; ++row )
   {
      success = Check(
         actual_view.row_ptr[row] == expected_view.row_ptr[row],
         "Device CSR row pointer differs from host finalization." ) && success;
   }
   for ( GlobalIndex i = 0; i < actual.nnz; ++i )
   {
      success = Check(
         actual_view.col_ind[i] == expected_view.col_ind[i] &&
            Near( actual_view.values[i], expected_view.values[i] ),
         "Device CSR payload differs from host finalization." ) && success;
   }
   return success;
}

template < typename ActualBackend, typename ExpectedBackend >
bool CompareCSC(
   const CSCMatrix< Real, GlobalIndex, ActualBackend > & actual,
   const CSCMatrix< Real, GlobalIndex, ExpectedBackend > & expected )
{
   bool success = Check(
      actual.num_rows == expected.num_rows &&
         actual.num_cols == expected.num_cols && actual.nnz == expected.nnz,
      "Device CSC dimensions or nnz differ from host finalization." );
   const auto actual_view = GetHostReadView( actual );
   const auto expected_view = GetHostReadView( expected );
   for ( GlobalIndex col = 0; col <= actual.num_cols; ++col )
   {
      success = Check(
         actual_view.col_ptr[col] == expected_view.col_ptr[col],
         "Device CSC column pointer differs from host finalization." ) && success;
   }
   for ( GlobalIndex i = 0; i < actual.nnz; ++i )
   {
      success = Check(
         actual_view.row_ind[i] == expected_view.row_ind[i] &&
            Near( actual_view.values[i], expected_view.values[i] ),
         "Device CSC payload differs from host finalization." ) && success;
   }
   return success;
}

#if defined(GENDIL_USE_HYPRE)
template < typename ActualBackend, typename ExpectedBackend >
bool CompareHypreCSR(
   const HypreCSRMatrix< ActualBackend > & actual,
   const HypreCSRMatrix< ExpectedBackend > & expected )
{
   bool success = Check(
      actual.csr.num_rows == expected.csr.num_rows &&
         actual.csr.num_cols == expected.csr.num_cols &&
         actual.csr.nnz == expected.csr.nnz,
      "Device HypreCSR dimensions or nnz differ from host finalization." );
   success = Check(
      actual.metadata.explicit_diagonal_count ==
            expected.metadata.explicit_diagonal_count &&
         actual.metadata.missing_diagonal_count ==
            expected.metadata.missing_diagonal_count &&
         actual.metadata.first_missing_diagonal ==
            expected.metadata.first_missing_diagonal,
      "Device HypreCSR diagonal metadata differs from host finalization." ) &&
      success;
   const auto actual_view = GetHostReadView( actual.csr );
   const auto expected_view = GetHostReadView( expected.csr );
   for ( HYPRE_Int row = 0; row <= actual.csr.num_rows; ++row )
   {
      success = Check(
         actual_view.row_ptr[row] == expected_view.row_ptr[row],
         "Device HypreCSR row pointer differs from host finalization." ) &&
         success;
   }
   for ( HYPRE_Int i = 0; i < actual.csr.nnz; ++i )
   {
      success = Check(
         actual_view.col_ind[i] == expected_view.col_ind[i] &&
            Near(
               static_cast< Real >( actual_view.values[i] ),
               static_cast< Real >( expected_view.values[i] ) ),
         "Device HypreCSR payload or diagonal-first ordering differs from "
         "host finalization." ) && success;
   }
   return success;
}
#endif

bool TestHostLazyResidency()
{
   auto raw = MakeFixture();
   bool success = Check(
      HostOnly( raw.rows ) && HostOnly( raw.cols ) && HostOnly( raw.values ),
      "The public RawCOO factory should leave data host-authoritative." );

   auto coo = FinalizeRawCOOToCOOHost(
      raw,
      HostCOOBackend<>{} );
   auto csr = FinalizeRawCOOToCSRHost(
      raw,
      HostCSRBackend<>{} );
   auto csc = FinalizeRawCOOToCSCHost(
      raw,
      HostCSCBackend<>{} );
#if defined(GENDIL_USE_HYPRE)
   auto hypre = FinalizeRawCOOToHypreCSRHost(
      raw,
      HypreCSRHostBackend{} );
#endif
   success = Check(
      HostOnly( coo.rows ) && HostOnly( coo.cols ) && HostOnly( coo.values ),
      "Host COO finalization should not eagerly materialize device arrays." ) &&
      success;
   const auto coo_view = GetHostReadView( coo );
   success = Check(
      coo.nnz == 10 && coo_view.rows[1] == 0 && coo_view.cols[1] == 4 &&
         coo_view.values[1] == 0.0,
      "Sparse finalization should retain an exact reduced zero." ) && success;
   success = Check(
      HostOnly( csr.row_ptr ) && HostOnly( csr.col_ind ) &&
         HostOnly( csr.values ),
      "Host CSR finalization should not eagerly materialize device arrays." ) &&
      success;
   success = Check(
      HostOnly( csc.col_ptr ) && HostOnly( csc.row_ind ) &&
         HostOnly( csc.values ),
      "Host CSC finalization should not eagerly materialize device arrays." ) &&
      success;
#if defined(GENDIL_USE_HYPRE)
   success = Check(
      HostOnly( hypre.csr.row_ptr ) && HostOnly( hypre.csr.col_ind ) &&
         HostOnly( hypre.csr.values ),
      "Host HypreCSR finalization should not eagerly materialize device "
      "arrays." ) && success;
#endif
   return success;
}

#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

bool TestDeviceFinalization()
{
   auto initialized_raw =
      details::MakeAssemblyRawCOOTripletBuffer<
         true,
         Real,
         GlobalIndex >( 2, 3, 4 );
   GENDIL_DEVICE_SYNC;
   bool success = Check(
      DeviceOnly( initialized_raw.rows ) &&
         DeviceOnly( initialized_raw.cols ) &&
         DeviceOnly( initialized_raw.values ),
      "Device RawCOO initialization should not materialize host arrays." );
   (void) GetHostReadView( initialized_raw );
   success = Check(
      IsHostValid( initialized_raw.rows ) &&
         IsHostValid( initialized_raw.cols ) &&
         IsHostValid( initialized_raw.values ),
      "The first RawCOO host read should materialize host arrays." ) && success;

   auto host_raw = MakeFixture();
   auto device_raw = MakeFixture();
   (void) GetDeviceReadWriteView( device_raw );

   success = Check(
      DeviceOnly( device_raw.rows ) && DeviceOnly( device_raw.cols ) &&
         DeviceOnly( device_raw.values ),
      "The device RawCOO fixture should be device-authoritative." ) && success;

   auto host_coo = FinalizeRawCOOToCOOHost(
      host_raw,
      HostCOOBackend<>{} );
   auto host_csr = FinalizeRawCOOToCSRHost(
      host_raw,
      HostCSRBackend<>{} );
   auto host_csc = FinalizeRawCOOToCSCHost(
      host_raw,
      HostCSCBackend<>{} );

   auto device_coo = FinalizeRawCOOToCOODevice(
      device_raw,
      NativeDeviceCOOBackend<>{} );
   using DevicePolicy =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 1 >;
   auto device_csr = FinalizeRawCOOToCSR< DevicePolicy >(
      device_raw,
      NativeDeviceCSRBackend<>{} );
   auto device_csc = FinalizeRawCOOToCSCDevice(
      device_raw,
      NativeDeviceCSCBackend<>{} );
#if defined(GENDIL_USE_HYPRE)
   auto host_hypre = FinalizeRawCOOToHypreCSRHost(
      host_raw,
      HypreCSRHostBackend{} );
   auto device_hypre = FinalizeRawCOOToHypreCSRDevice(
      device_raw,
      HypreCSRHostBackend{} );
#endif

   success = Check(
      DeviceOnly( device_coo.rows ) && DeviceOnly( device_coo.cols ) &&
         DeviceOnly( device_coo.values ),
      "Device COO finalization should leave arrays device-authoritative." ) &&
      success;
#if defined(GENDIL_USE_HYPRE)
   success = Check(
      DeviceOnly( device_hypre.csr.row_ptr ) &&
         DeviceOnly( device_hypre.csr.col_ind ) &&
         DeviceOnly( device_hypre.csr.values ),
      "Device HypreCSR finalization should leave arrays device-authoritative." ) &&
      success;
#endif
   success = Check(
      DeviceOnly( device_csr.row_ptr ) && DeviceOnly( device_csr.col_ind ) &&
         DeviceOnly( device_csr.values ),
      "Device CSR finalization should leave arrays device-authoritative." ) &&
      success;
   success = Check(
      DeviceOnly( device_csc.col_ptr ) && DeviceOnly( device_csc.row_ind ) &&
         DeviceOnly( device_csc.values ),
      "Device CSC finalization should leave arrays device-authoritative." ) &&
      success;

   Vector x( device_csr.num_cols );
   auto * x_host = x.WriteHostData();
   for ( GlobalIndex i = 0; i < device_csr.num_cols; ++i )
   {
      x_host[i] = Real( i + 1 );
   }
   Vector y( device_csr.num_rows );
   Apply( NativeDeviceCSRBackend<>{}, device_csr, x, y );
   GENDIL_DEVICE_SYNC;
   success = Check(
      DeviceOnly( device_csr.row_ptr ) && DeviceOnly( device_csr.col_ind ) &&
         DeviceOnly( device_csr.values ),
      "Immediate device matvec should not materialize CSR arrays on host." ) &&
      success;

   success = CompareCOO( device_coo, host_coo ) && success;
   success = CompareCSR( device_csr, host_csr ) && success;
   success = CompareCSC( device_csc, host_csc ) && success;
#if defined(GENDIL_USE_HYPRE)
   success = CompareHypreCSR( device_hypre, host_hypre ) && success;
#endif

   auto empty = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 0, 0, 0 );
   auto empty_device = FinalizeRawCOOToCSRDevice(
      empty,
      NativeDeviceCSRBackend<>{} );
   auto empty_coo = FinalizeRawCOOToCOODevice(
      empty,
      NativeDeviceCOOBackend<>{} );
   auto empty_csc = FinalizeRawCOOToCSCDevice(
      empty,
      NativeDeviceCSCBackend<>{} );
   success = Check(
      empty_device.num_rows == 0 && empty_device.num_cols == 0 &&
         empty_device.nnz == 0 && ReadHost( empty_device.row_ptr )[0] == 0,
      "Zero-sized device CSR finalization is incorrect." ) && success;
   success = Check(
      empty_coo.num_rows == 0 && empty_coo.num_cols == 0 &&
         empty_coo.nnz == 0 && empty_csc.num_rows == 0 &&
         empty_csc.num_cols == 0 && empty_csc.nnz == 0 &&
         ReadHost( empty_csc.col_ptr )[0] == 0,
      "Zero-sized device COO/CSC finalization is incorrect." ) && success;
#if defined(GENDIL_USE_HYPRE)
   auto empty_hypre = FinalizeRawCOOToHypreCSRDevice(
      empty,
      HypreCSRHostBackend{} );
   success = Check(
      empty_hypre.csr.num_rows == 0 && empty_hypre.csr.num_cols == 0 &&
         empty_hypre.csr.nnz == 0 &&
         empty_hypre.metadata.diagonal_rows == 0 &&
         empty_hypre.metadata.has_explicit_diagonal,
      "Zero-sized device HypreCSR finalization is incorrect." ) && success;
#endif
   return success;
}

#endif

#if defined(GENDIL_USE_DEVICE) && \
   !defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)

bool TestDeviceBuildHostFallback()
{
   using DevicePolicy =
      DeviceKernelConfiguration< ThreadBlockLayout<>, 0, 1 >;
   auto raw = MakeFixture();
   (void) GetDeviceReadWriteView( raw );
   auto matrix = FinalizeRawCOOToCSR< DevicePolicy >(
      raw,
      NativeDeviceCSRBackend<>{} );
   return Check(
      HostOnly( matrix.row_ptr ) && HostOnly( matrix.col_ind ) &&
         HostOnly( matrix.values ),
      "A device build without primitives should use host finalization." );
}

#endif

} // namespace

int main()
{
   bool success = TestHostLazyResidency();
#if defined(GENDIL_HAS_DEVICE_SPARSE_FINALIZATION)
   success = TestDeviceFinalization() && success;
#elif defined(GENDIL_USE_DEVICE)
   success = TestDeviceBuildHostFallback() && success;
#endif
   return success ? 0 : 1;
}
