// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-sgbsr-device-gather-scatter skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-10;
using ScalarFE = GLFiniteElement< 2, 1 >;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

template < typename VectorType >
bool CheckVector(
   const VectorType & vector,
   const Real * expected,
   const size_t size,
   const char * message )
{
   const auto * data = ReadHostVector( vector );
   bool success = true;
   for ( size_t i = 0; i < size; ++i )
   {
      success = Check( Near( data[i], expected[i] ), message ) && success;
   }
   return success;
}

template < typename VectorType >
void FillVector( VectorType & vector, const Real offset )
{
   auto * data = WriteHostVector( vector );
   for ( size_t i = 0; i < GetVectorSize( vector ); ++i )
   {
      data[i] = offset + Real( 0.25 ) * static_cast< Real >( i );
   }
}

template < typename FiniteElementSpace >
auto MakeIdentityDeviceBSR( const FiniteElementSpace & finite_element_space )
{
   using ShapeFunctions =
      typename FiniteElementSpace::finite_element_type::shape_functions;
   constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();

   auto matrix =
      MakeBlockDiagonalDGBSRPattern<
         Real,
         GlobalIndex,
         BlockLayout::ColumnMajor >(
            finite_element_space.GetNumberOfFiniteElements(),
            block_size,
            block_size,
            NativeDeviceBSRBackend<>{} );

   auto view = GetHostValuesWriteView( matrix );
   for ( GlobalIndex block = 0; block < matrix.num_blocks; ++block )
   {
      for ( GlobalIndex local_row = 0;
            local_row < block_size;
            ++local_row )
      {
         for ( GlobalIndex local_col = 0;
               local_col < block_size;
               ++local_col )
         {
            view.GetBlockEntry( block, local_row, local_col ) =
               local_row == local_col ? Real( 1 ) : Real( 0 );
         }
      }
   }
   Sync( matrix );
   return matrix;
}

template <
   typename FiniteElementSpace,
   typename TrialGather,
   typename TestScatter >
bool CheckDGVariant(
   const FiniteElementSpace & finite_element_space,
   TrialGather trial_gather,
   TestScatter test_scatter,
   const char * apply_message,
   const char * apply_add_message )
{
   auto bsr_matrix = MakeIdentityDeviceBSR( finite_element_space );
   using BSRType = std::remove_cvref_t< decltype( bsr_matrix ) >;
   SGBSRMatrix< BSRType, TrialGather, TestScatter > matrix(
      std::move( bsr_matrix ),
      std::move( trial_gather ),
      std::move( test_scatter ) );

   const size_t size = static_cast< size_t >(
      finite_element_space.GetNumberOfFiniteElementDofs() );
   Vector x( size );
   FillVector( x, Real( 0.5 ) );

   Vector host_y( size );
   Vector device_y( size );
   Apply( HostBSRBackend<>{}, matrix, x, host_y );
   Apply( NativeDeviceBSRBackend<>{}, matrix, x, device_y );

   bool success = true;
   success = Check(
      device_y.IsDeviceValid() && !device_y.IsHostValid(),
      "Device SGBSR Apply staged its output through host memory." ) &&
      success;
   if constexpr ( !TrialGather::is_identity )
   {
      success = Check(
         matrix.x_bsr.IsDeviceValid() && !matrix.x_bsr.IsHostValid(),
         "Device SGBSR gather did not leave its workspace device-valid." ) &&
         success;
   }
   if constexpr ( !TestScatter::is_identity )
   {
      success = Check(
         matrix.y_bsr.IsDeviceValid() && !matrix.y_bsr.IsHostValid(),
         "Device SGBSR BSR output workspace was staged through host memory." ) &&
         success;
   }

   const auto * host_data = ReadHostVector( host_y );
   success =
      CheckVector( device_y, host_data, size, apply_message ) && success;

   host_y = Real( 3 );
   device_y = Real( 3 );
   ApplyAdd( HostBSRBackend<>{}, matrix, x, host_y );
   ApplyAdd( NativeDeviceBSRBackend<>{}, matrix, x, device_y );
   host_data = ReadHostVector( host_y );
   success =
      CheckVector( device_y, host_data, size, apply_add_message ) && success;

   Vector stored_y( size );
   Apply( matrix, x, stored_y );
   const auto * x_data = ReadHostVector( x );
   success = CheckVector(
      stored_y,
      x_data,
      size,
      "Stored native-device SGBSR Apply produced the wrong result." ) &&
      success;

   return success;
}

bool TestDGMappings()
{
   constexpr Integer num_elements = 16;
   Cartesian1DMesh mesh( Real( 1 ) / num_elements, num_elements );
   auto finite_element_space =
      MakeFiniteElementSpace( mesh, ScalarFE{}, L2Restriction{} );
   using Space = decltype( finite_element_space );

   bool success = true;
   success = CheckDGVariant(
      finite_element_space,
      IdentityBsrGather{},
      IdentityBsrScatter{},
      "Identity device SGBSR Apply disagrees with the host backend.",
      "Identity device SGBSR ApplyAdd disagrees with the host backend." ) &&
      success;
   success = CheckDGVariant(
      finite_element_space,
      IdentityBsrGather{},
      DGScatterFromBsr< Space >{ finite_element_space },
      "Device SGBSR scatter-only Apply disagrees with the host backend.",
      "Device SGBSR scatter-only ApplyAdd disagrees with the host backend." ) &&
      success;
   success = CheckDGVariant(
      finite_element_space,
      DGGatherToBsr< Space >{ finite_element_space },
      IdentityBsrScatter{},
      "Device SGBSR gather-only Apply disagrees with the host backend.",
      "Device SGBSR gather-only ApplyAdd disagrees with the host backend." ) &&
      success;
   success = CheckDGVariant(
      finite_element_space,
      DGGatherToBsr< Space >{ finite_element_space },
      DGScatterFromBsr< Space >{ finite_element_space },
      "Device SGBSR gather/scatter Apply disagrees with the host backend.",
      "Device SGBSR gather/scatter ApplyAdd disagrees with the host backend." ) &&
      success;
   return success;
}

HostDevicePointer< const int > MakeRestrictionView(
   const SyncHostDeviceArray< int > & indices )
{
   HostDevicePointer< const int > view{};
   view.host_pointer = indices.data.host_pointer;
   view.device_pointer = indices.data.device_pointer;
   return view;
}

bool TestScalarH1AtomicScatter()
{
   constexpr Integer num_elements = 128;
   constexpr GlobalIndex local_dofs = 2;
   Cartesian1DMesh mesh( Real( 1 ) / num_elements, num_elements );

   auto restriction_indices =
      MakeSyncHostDeviceArray< int >(
         GlobalIndex( num_elements ) * local_dofs );
   auto * host_indices = WriteHost( restriction_indices );
   for ( GlobalIndex i = 0;
         i < GlobalIndex( num_elements ) * local_dofs;
         ++i )
   {
      host_indices[i] = 0;
   }
   Sync( restriction_indices );

   H1Restriction restriction{
      MakeRestrictionView( restriction_indices ),
      1 };
   auto finite_element_space =
      MakeFiniteElementSpace( mesh, ScalarFE{}, restriction );
   using Space = decltype( finite_element_space );

   auto bsr_matrix = MakeIdentityDeviceBSR( finite_element_space );
   using BSRType = std::remove_cvref_t< decltype( bsr_matrix ) >;
   SGBSRMatrix<
      BSRType,
      CGGatherToBsr< Space >,
      CGScatterFromBsr< Space > > matrix(
         std::move( bsr_matrix ),
         CGGatherToBsr< Space >{ finite_element_space },
         CGScatterFromBsr< Space >{ finite_element_space } );

   Vector x( 1 );
   WriteHostVector( x )[0] = Real( 2 );
   Vector host_y( 1 );
   Vector device_y( 1 );
   Apply( HostBSRBackend<>{}, matrix, x, host_y );
   Apply( NativeDeviceBSRBackend<>{}, matrix, x, device_y );

   bool success = true;
   success = Check(
      device_y.IsDeviceValid() && !device_y.IsHostValid(),
      "Device scalar H1 scatter staged its output through host memory." ) &&
      success;
   const Real expected =
      Real( 2 ) * Real( num_elements ) * Real( local_dofs );
   success = CheckVector(
      device_y,
      &expected,
      1,
      "Atomic scalar H1 device scatter lost shared-DoF contributions." ) &&
      success;
   const Real * host_data = ReadHostVector( host_y );
   success = Check(
      Near( host_data[0], expected ),
      "Scalar H1 host reference produced the wrong shared-DoF sum." ) &&
      success;

   device_y = Real( 5 );
   ApplyAdd( matrix, x, device_y );
   const Real additive_expected = expected + Real( 5 );
   success = CheckVector(
      device_y,
      &additive_expected,
      1,
      "Atomic scalar H1 device ApplyAdd lost its initial output." ) &&
      success;
   return success;
}

bool TestVectorH1AtomicScatter()
{
   constexpr Integer num_elements = 128;
   constexpr GlobalIndex scalar_local_dofs = 2;
   Cartesian1DMesh mesh( Real( 1 ) / num_elements, num_elements );

   auto restriction_indices =
      MakeSyncHostDeviceArray< int >(
         GlobalIndex( num_elements ) * scalar_local_dofs );
   auto * host_indices = WriteHost( restriction_indices );
   for ( GlobalIndex i = 0;
         i < GlobalIndex( num_elements ) * scalar_local_dofs;
         ++i )
   {
      host_indices[i] = 0;
   }
   Sync( restriction_indices );

   H1Restriction scalar_restriction{
      MakeRestrictionView( restriction_indices ),
      1 };
   auto vector_restriction =
      MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_finite_element =
      MakeVectorFiniteElement( ScalarFE{}, ScalarFE{} );
   auto finite_element_space =
      MakeFiniteElementSpace(
         mesh,
         vector_finite_element,
         vector_restriction );
   using Space = decltype( finite_element_space );

   auto bsr_matrix = MakeIdentityDeviceBSR( finite_element_space );
   using BSRType = std::remove_cvref_t< decltype( bsr_matrix ) >;
   SGBSRMatrix<
      BSRType,
      VectorCGGatherToBsr< Space >,
      VectorCGScatterFromBsr< Space > > matrix(
         std::move( bsr_matrix ),
         VectorCGGatherToBsr< Space >{ finite_element_space },
         VectorCGScatterFromBsr< Space >{ finite_element_space } );

   Vector x( 2 );
   auto * x_data = WriteHostVector( x );
   x_data[0] = Real( 1.5 );
   x_data[1] = Real( -0.25 );

   Vector host_y( 2 );
   Vector device_y( 2 );
   Apply( HostBSRBackend<>{}, matrix, x, host_y );
   Apply( NativeDeviceBSRBackend<>{}, matrix, x, device_y );

   const Real multiplicity =
      Real( num_elements ) * Real( scalar_local_dofs );
   const Real expected[2]{
      x_data[0] * multiplicity,
      x_data[1] * multiplicity };
   bool success = CheckVector(
      device_y,
      expected,
      2,
      "Atomic vector H1 device scatter lost shared-DoF contributions." );
   success = CheckVector(
      host_y,
      expected,
      2,
      "Vector H1 host reference produced the wrong shared-DoF sum." ) &&
      success;

   device_y = Real( 4 );
   ApplyAdd(
      NativeDeviceBSRBackend<>{},
      matrix,
      x,
      device_y );
   const Real additive_expected[2]{
      expected[0] + Real( 4 ),
      expected[1] + Real( 4 ) };
   success = CheckVector(
      device_y,
      additive_expected,
      2,
      "Atomic vector H1 device ApplyAdd lost its initial output." ) &&
      success;
   return success;
}

#if defined(GENDIL_USE_MFEM)
bool TestMFEMAndMixedDeviceVectors()
{
   constexpr Integer num_elements = 8;
   Cartesian1DMesh mesh( Real( 1 ) / num_elements, num_elements );
   auto finite_element_space =
      MakeFiniteElementSpace( mesh, ScalarFE{}, L2Restriction{} );
   using Space = decltype( finite_element_space );

   auto bsr_matrix = MakeIdentityDeviceBSR( finite_element_space );
   using BSRType = std::remove_cvref_t< decltype( bsr_matrix ) >;
   SGBSRMatrix<
      BSRType,
      DGGatherToBsr< Space >,
      DGScatterFromBsr< Space > > matrix(
         std::move( bsr_matrix ),
         DGGatherToBsr< Space >{ finite_element_space },
         DGScatterFromBsr< Space >{ finite_element_space } );

   const int size = finite_element_space.GetNumberOfFiniteElementDofs();
   mfem::Vector mfem_x( size );
   FillVector( mfem_x, Real( 0.75 ) );
   const auto * expected = ReadHostVector( mfem_x );

   mfem::Vector mfem_y( size );
   Apply( NativeDeviceBSRBackend<>{}, matrix, mfem_x, mfem_y );
   bool success = CheckVector(
      mfem_y,
      expected,
      static_cast< size_t >( size ),
      "Device SGBSR MFEM-vector Apply produced the wrong result." );

   Vector gendil_x( static_cast< size_t >( size ) );
   FillVector( gendil_x, Real( 1.25 ) );
   const auto * gendil_expected = ReadHostVector( gendil_x );
   Apply( NativeDeviceBSRBackend<>{}, matrix, gendil_x, mfem_y );
   success = CheckVector(
      mfem_y,
      gendil_expected,
      static_cast< size_t >( size ),
      "Device SGBSR GenDiL-to-MFEM Apply produced the wrong result." ) &&
      success;

   Vector gendil_y( static_cast< size_t >( size ) );
   Apply( NativeDeviceBSRBackend<>{}, matrix, mfem_x, gendil_y );
   success = CheckVector(
      gendil_y,
      expected,
      static_cast< size_t >( size ),
      "Device SGBSR MFEM-to-GenDiL Apply produced the wrong result." ) &&
      success;

   auto * additive_data = WriteHostVector( mfem_y );
   for ( int i = 0; i < size; ++i )
   {
      additive_data[i] = Real( 2 );
   }
   ApplyAdd( matrix, mfem_x, mfem_y );
   const auto * result = ReadHostVector( mfem_y );
   for ( int i = 0; i < size; ++i )
   {
      success = Check(
         Near( result[i], expected[i] + Real( 2 ) ),
         "Stored device SGBSR MFEM ApplyAdd produced the wrong result." ) &&
         success;
   }
   return success;
}
#endif

} // namespace

int main()
{
#if defined(GENDIL_USE_MFEM)
   #if defined(GENDIL_USE_CUDA)
   mfem::Device device( "cuda" );
   #else
   mfem::Device device( "hip" );
   #endif
#endif

   static_assert(
      is_device_matvec_backend_v< NativeDeviceBSRBackend<> > );
   static_assert(
      is_device_matvec_backend_v< VendorDeviceBSRBackend<> > );

   bool success = true;
   success = TestDGMappings() && success;
   success = TestScalarH1AtomicScatter() && success;
   success = TestVectorH1AtomicScatter() && success;
#if defined(GENDIL_USE_MFEM)
   success = TestMFEMAndMixedDeviceVectors() && success;
#endif

   if ( !success )
   {
      return 1;
   }

   std::cout << "SGBSR device gather/scatter tests passed.\n";
   return 0;
}

#endif
