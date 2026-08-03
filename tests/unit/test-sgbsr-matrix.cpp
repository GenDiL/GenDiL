// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

using ScalarFE0 = GLFiniteElement< 1, 1 >;
using ScalarFE1 = GLFiniteElement< 2, 1 >;
using ScalarShape0 = typename ScalarFE0::shape_functions;
using VectorFE = decltype( MakeVectorFiniteElement( ScalarFE0{}, ScalarFE1{} ) );
using VectorShape = typename VectorFE::shape_functions;
using VectorSpace = FiniteElementSpace< Cartesian2DMesh, VectorFE, L2Restriction >;
using Component0Tag = std::integral_constant< size_t, 0 >;
using Component1Tag = std::integral_constant< size_t, 1 >;
using IdentitySGBSRMatrix =
   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      IdentityBsrGather,
      IdentityBsrScatter >;

static_assert( !std::is_copy_constructible_v< IdentitySGBSRMatrix > );
static_assert( !std::is_copy_assignable_v< IdentitySGBSRMatrix > );
static_assert( std::is_move_constructible_v< IdentitySGBSRMatrix > );
static_assert( std::is_move_assignable_v< IdentitySGBSRMatrix > );
static_assert(
   requires ( IdentitySGBSRMatrix & matrix )
   {
      matrix.bsr_matrix;
      matrix.trial_gather;
      matrix.test_scatter;
      matrix.x_bsr;
      matrix.y_bsr;
   } );

constexpr Real tolerance = 1.0e-12;

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

constexpr std::array< int, 8 > h1_q1_restriction_map{
   0, 1, 3, 4,
   1, 2, 4, 5
};

HostDevicePointer< const int > MakeManualH1RestrictionIndices()
{
   HostDevicePointer< const int > indices{};
   indices.host_pointer = h1_q1_restriction_map.data();
   return indices;
}

void FillVectorH1InputCase( Vector & x, const Integer case_id )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] = 0.0;
   }

   if ( case_id == 0 )
   {
      for ( GlobalIndex i = 0; i < x.Size(); ++i )
      {
         data[i] = 0.5 + 0.25 * static_cast< Real >( i );
      }
   }
   else if ( case_id == 1 )
   {
      data[0] = 1.0;
      data[1] = 2.0;
      data[2] = 3.0;
   }
   else
   {
      data[3] = -1.0;
      data[4] = 0.5;
      data[5] = 4.0;
   }
}

void ApplyTwoCellVectorH1P1MassReference(
   const Vector & x,
   Vector & y,
   const Real h )
{
   // Two uniform 1D p1 H1 elements have local scalar mass
   // (h / 6) * [[2, 1], [1, 2]]. Assembling the shared middle
   // node gives the component-major diag(M_scalar, M_scalar) action.
   const Real scale = h / 6.0;
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = 0.0;
   }

   for ( GlobalIndex component = 0; component < 2; ++component )
   {
      const GlobalIndex offset = 3 * component;
      y_data[offset + 0] =
         scale * ( 2.0 * x_data[offset + 0] + x_data[offset + 1] );
      y_data[offset + 1] =
         scale * ( x_data[offset + 0] +
                   4.0 * x_data[offset + 1] +
                   x_data[offset + 2] );
      y_data[offset + 2] =
         scale * ( x_data[offset + 1] + 2.0 * x_data[offset + 2] );
   }
}

bool CheckVectorNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < actual.Size(); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

void FillIdentityBlocks(
   BSRMatrix< Real, GlobalIndex > & matrix )
{
   auto * matrix_values = ReadWriteHost( matrix.values );
   const GlobalIndex block_size = matrix.block_rows;
   GENDIL_VERIFY(
      matrix.block_rows == matrix.block_cols,
      "FillIdentityBlocks requires square BSR blocks." );

   for ( GlobalIndex i = 0;
         i < matrix.num_blocks * matrix.block_rows * matrix.block_cols;
         ++i )
   {
      matrix_values[i] = 0.0;
   }

   for ( GlobalIndex block = 0; block < matrix.num_blocks; ++block )
   {
      const GlobalIndex block_offset = block * block_size * block_size;
      for ( GlobalIndex local = 0; local < block_size; ++local )
      {
         matrix_values[block_offset + local * block_size + local] = 1.0;
      }
   }
}

struct CopyBsrGather
{
   static constexpr bool is_identity = false;

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void operator()(
      const Backend &,
      const InputVector & input,
      OutputVector & output ) const
   {
      GENDIL_VERIFY(
         GetVectorSize( input ) == GetVectorSize( output ),
         "CopyBsrGather requires matching input and output sizes." );

      if constexpr ( is_host_matvec_backend_v< Backend > )
      {
         const auto * input_data = ReadHostVector( input );
         auto * output_data = WriteHostVector( output );
         for ( size_t i = 0; i < GetVectorSize( input ); ++i )
         {
            output_data[i] = input_data[i];
         }
      }
      else
      {
         static_assert( is_device_matvec_backend_v< Backend > );
         const auto * input_data = ReadDeviceVector( input );
         auto * output_data = WriteDeviceVector( output );
         DeviceLoop(
            GetVectorSize( input ),
            [=] GENDIL_HOST_DEVICE ( const size_t i )
            {
               output_data[i] = input_data[i];
            } );
      }
   }
};

struct CopyBsrScatter
{
   static constexpr bool is_identity = false;

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void operator()(
      const Backend & backend,
      const InputVector & input,
      OutputVector & output ) const
   {
      Scatter< false >( backend, input, output );
   }

   template <
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void ApplyAdd(
      const Backend & backend,
      const InputVector & input,
      OutputVector & output ) const
   {
      Scatter< true >( backend, input, output );
   }

   template <
      bool Add,
      typename Backend,
      typename InputVector,
      typename OutputVector >
   void Scatter(
      const Backend &,
      const InputVector & input,
      OutputVector & output ) const
   {
      GENDIL_VERIFY(
         GetVectorSize( input ) == GetVectorSize( output ),
         "CopyBsrScatter requires matching input and output sizes." );

      if constexpr ( is_host_matvec_backend_v< Backend > )
      {
         const auto * input_data = ReadHostVector( input );
         auto * output_data = [&] ()
         {
            if constexpr ( Add )
            {
               return ReadWriteHostVector( output );
            }
            else
            {
               return WriteHostVector( output );
            }
         }();
         for ( size_t i = 0; i < GetVectorSize( input ); ++i )
         {
            if constexpr ( Add )
            {
               output_data[i] += input_data[i];
            }
            else
            {
               output_data[i] = input_data[i];
            }
         }
      }
      else
      {
         static_assert( is_device_matvec_backend_v< Backend > );
         const auto * input_data = ReadDeviceVector( input );
         auto * output_data = [&] ()
         {
            if constexpr ( Add )
            {
               return ReadWriteDeviceVector( output );
            }
            else
            {
               return WriteDeviceVector( output );
            }
         }();
         DeviceLoop(
            GetVectorSize( input ),
            [=] GENDIL_HOST_DEVICE ( const size_t i )
            {
               if constexpr ( Add )
               {
                  output_data[i] += input_data[i];
               }
               else
               {
                  output_data[i] = input_data[i];
               }
            } );
      }
   }
};

template < typename TrialGather, typename TestScatter >
bool CheckSGBSRFreeApplyVariant(
   TrialGather trial_gather,
   TestScatter test_scatter )
{
   auto bsr_matrix =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 2 );
   FillIdentityBlocks( bsr_matrix );

   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      TrialGather,
      TestScatter > matrix(
         std::move( bsr_matrix ),
         std::move( trial_gather ),
         std::move( test_scatter ) );

   Vector x( 4 );
   auto * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.5 + static_cast< Real >( i );
   }

   Vector y_operator( 4 );
   Vector y_stored( 4 );
   Vector y_explicit( 4 );
   matrix( x, y_operator );
   Apply( matrix, x, y_stored );
   Apply( HostBSRBackend<>{}, matrix, x, y_explicit );

   bool success = true;
   success = Check(
      matrix.bsr_matrix.num_blocks == 2 &&
      matrix.x_bsr.Size() == 4 &&
      matrix.y_bsr.Size() == 4,
      "SGBSRMatrix public composition state has inconsistent dimensions." ) &&
      success;

   const auto * operator_data = y_operator.ReadHostData();
   const auto * stored_data = y_stored.ReadHostData();
   const auto * explicit_data = y_explicit.ReadHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      success = Check(
         Near( operator_data[i], x_data[i] ) &&
         Near( stored_data[i], x_data[i] ) &&
         Near( explicit_data[i], x_data[i] ),
         "SGBSRMatrix free Apply disagrees with operator() or the expected "
         "identity result." ) && success;
   }

   y_stored = 2.0;
   y_explicit = 3.0;
   ApplyAdd( matrix, x, y_stored );
   ApplyAdd( HostBSRBackend<>{}, matrix, x, y_explicit );
   stored_data = y_stored.ReadHostData();
   explicit_data = y_explicit.ReadHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      success = Check(
         Near( stored_data[i], x_data[i] + 2.0 ) &&
         Near( explicit_data[i], x_data[i] + 3.0 ),
         "SGBSRMatrix free ApplyAdd produced the wrong additive result." ) &&
         success;
   }

   return success;
}

bool TestSGBSRFreeApplyVariants()
{
   bool success = true;
   success = CheckSGBSRFreeApplyVariant(
      IdentityBsrGather{},
      IdentityBsrScatter{} ) && success;
   success = CheckSGBSRFreeApplyVariant(
      IdentityBsrGather{},
      CopyBsrScatter{} ) && success;
   success = CheckSGBSRFreeApplyVariant(
      CopyBsrGather{},
      IdentityBsrScatter{} ) && success;
   success = CheckSGBSRFreeApplyVariant(
      CopyBsrGather{},
      CopyBsrScatter{} ) && success;
   return success;
}

bool TestIdentityWrapperMatchesRawBsr()
{
   auto raw_matrix = MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 3 );
   auto * raw_values = ReadWriteHost( raw_matrix.values );

   for ( GlobalIndex i = 0;
         i < raw_matrix.num_blocks * raw_matrix.block_rows * raw_matrix.block_cols;
         ++i )
   {
      raw_values[i] = static_cast< Real >( 1 + i );
   }

   Vector x( 6 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.25 + static_cast< Real >( i );
   }

   Vector y_raw( 4 );
   Vector y_sg( 4 );
   y_raw = 0.0;
   y_sg = 0.0;

   raw_matrix( x, y_raw );

   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      IdentityBsrGather,
      IdentityBsrScatter > sg_matrix(
         std::move( raw_matrix ),
         IdentityBsrGather{},
         IdentityBsrScatter{} );

   sg_matrix( x, y_sg );

   bool success = true;
   success = Check(
      sg_matrix.TrialBsrSize() == 6,
      "Identity SGBSRMatrix trial BSR size is wrong." ) && success;
   success = Check(
      sg_matrix.TestBsrSize() == 4,
      "Identity SGBSRMatrix test BSR size is wrong." ) && success;

   const Real * y_raw_data = y_raw.ReadHostData();
   const Real * y_sg_data = y_sg.ReadHostData();
   for ( GlobalIndex i = 0; i < y_raw.Size(); ++i )
   {
      success = Check(
         Near( y_raw_data[i], y_sg_data[i] ),
         "Identity SGBSRMatrix apply disagrees with raw BSRMatrix." ) && success;
   }

   y_sg = 2.5;
   ApplyAdd( sg_matrix, x, y_sg );
   y_sg_data = y_sg.ReadHostData();
   for ( GlobalIndex i = 0; i < y_raw.Size(); ++i )
   {
      success = Check(
         Near( y_sg_data[i], y_raw_data[i] + 2.5 ),
         "Identity SGBSRMatrix ApplyAdd disagrees with raw BSRMatrix." ) &&
         success;
   }

   y_sg = 1.25;
   ApplyAdd( HostBSRBackend<>{}, sg_matrix, x, y_sg );
   y_sg_data = y_sg.ReadHostData();
   for ( GlobalIndex i = 0; i < y_raw.Size(); ++i )
   {
      success = Check(
         Near( y_sg_data[i], y_raw_data[i] + 1.25 ),
         "Explicit-backend SGBSRMatrix ApplyAdd produced the wrong result." ) &&
         success;
   }

   return success;
}

bool TestSGBSRMoveAssignment()
{
   using Matrix =
      SGBSRMatrix<
         BSRMatrix< Real, GlobalIndex >,
         IdentityBsrGather,
         IdentityBsrScatter >;

   auto source_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 2 );
   FillIdentityBlocks( source_bsr );
   Matrix source(
      std::move( source_bsr ),
      IdentityBsrGather{},
      IdentityBsrScatter{} );

   auto destination_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 1, 1, 1 );
   FillIdentityBlocks( destination_bsr );
   Matrix destination(
      std::move( destination_bsr ),
      IdentityBsrGather{},
      IdentityBsrScatter{} );

   Matrix moved_source( std::move( source ) );
   destination = std::move( moved_source );

   Vector x( 4 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 1.0 + static_cast< Real >( i );
   }
   Vector y( 4 );
   Apply( destination, x, y );

   bool success = true;
   success = Check(
      destination.bsr_matrix.num_blocks == 2 &&
      destination.x_bsr.Size() == 4 &&
      destination.y_bsr.Size() == 4,
      "Move-assigned SGBSRMatrix did not transfer its public storage and "
      "workspaces." ) && success;
   const Real * y_data = y.ReadHostData();
   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      success =
         Check(
            Near( y_data[i], x_data[i] ),
            "Move-assigned SGBSRMatrix lost its BSR storage or workspace." ) &&
         success;
   }
   return success;
}

bool TestRawBsrOperatorDelegatesToBackendApply()
{
   auto raw_matrix = MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 2, 3 );
   auto * raw_values = ReadWriteHost( raw_matrix.values );

   for ( GlobalIndex i = 0;
         i < raw_matrix.num_blocks * raw_matrix.block_rows * raw_matrix.block_cols;
         ++i )
   {
      raw_values[i] = static_cast< Real >( 2 + 3 * i );
   }

   Vector x( 6 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = -0.5 + static_cast< Real >( i );
   }

   Vector y_operator( 4 );
   Vector y_apply( 4 );
   y_operator = 0.0;
   y_apply = 0.0;

   raw_matrix( x, y_operator );
   Apply( raw_matrix.backend, raw_matrix, x, y_apply );

   bool success = true;
   const Real * y_operator_data = y_operator.ReadHostData();
   const Real * y_apply_data = y_apply.ReadHostData();
   for ( GlobalIndex i = 0; i < y_operator.Size(); ++i )
   {
      success = Check(
         Near( y_operator_data[i], y_apply_data[i] ),
         "Raw BSR operator() disagrees with backend Apply." ) && success;
   }

   return success;
}

bool TestScalarH1GatherScatterMapping()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   H1Restriction restriction{ MakeManualH1RestrictionIndices(), 6 };
   auto h1_space = MakeFiniteElementSpace( mesh, ScalarFE0{}, restriction );

   constexpr GlobalIndex block_size = LocalDofCount< ScalarShape0 >();
   const GlobalIndex num_elements = h1_space.GetNumberOfFiniteElements();

   Vector x_fe( h1_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 10.0 * static_cast< Real >( i + 1 );
   }

   CGGatherToBsr< decltype( h1_space ) > gather{ h1_space };
   Vector x_bsr( num_elements * block_size );
   gather( HostBSRBackend<>{}, x_fe, x_bsr );

   bool success = true;
   const Real expected_gather[] = {
      10.0, 20.0, 40.0, 50.0,
      20.0, 30.0, 50.0, 60.0
   };
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < x_bsr.Size(); ++i )
   {
      success = Check(
         Near( x_bsr_data[i], expected_gather[i] ),
         "Scalar H1 gather mapping is wrong." ) && success;
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = static_cast< Real >( i + 1 );
   }

   Vector y_fe( h1_space.GetNumberOfFiniteElementDofs() );
   Real * y_fe_data = y_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      y_fe_data[i] = 99.0;
   }

   CGScatterFromBsr< decltype( h1_space ) > scatter{ h1_space };
   scatter( HostBSRBackend<>{}, y_bsr, y_fe );

   const Real expected_scatter[] = {
      1.0, 7.0, 6.0, 3.0, 11.0, 8.0
   };
   y_fe_data = y_fe.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], expected_scatter[i] ),
         "Scalar H1 scatter-add mapping or Set semantics is wrong." ) && success;
   }

   y_fe = 4.0;
   scatter.ApplyAdd( HostBSRBackend<>{}, y_bsr, y_fe );
   y_fe_data = y_fe.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], expected_scatter[i] + 4.0 ),
         "Scalar H1 additive scatter did not preserve the initial output." ) &&
         success;
   }

   return success;
}

bool TestSharedScalarH1HostScatter()
{
   constexpr Integer num_elements = 4096;
   Cartesian1DMesh mesh(
      Real( 1 ) / static_cast< Real >( num_elements ),
      num_elements );

   FiniteElementOrders< 1 > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   std::vector< int > restriction_map(
      static_cast< size_t >( 2 * num_elements ) );
   for ( Integer element = 0; element < num_elements; ++element )
   {
      restriction_map[static_cast< size_t >( 2 * element )] = 0;
      restriction_map[static_cast< size_t >( 2 * element + 1 )] = 1;
   }

   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction restriction{ restriction_indices, 2 };
   auto finite_element_space =
      MakeFiniteElementSpace( mesh, finite_element, restriction );

   Vector element_values( 2 * num_elements );
   auto * element_data = WriteHostVector( element_values );
   for ( Integer i = 0; i < 2 * num_elements; ++i )
   {
      element_data[i] = Real( 1 );
   }

   CGScatterFromBsr< decltype( finite_element_space ) > scatter{
      finite_element_space };
   Vector output( 2 );
   scatter( HostBSRBackend<>{}, element_values, output );

   const auto * output_data = ReadHostVector( output );
   bool success = Check(
      output_data[0] == Real( num_elements ) &&
         output_data[1] == Real( num_elements ),
      "Parallel host H1 scatter lost shared-DoF contributions." );

   output = Real( 3 );
   scatter.ApplyAdd( HostBSRBackend<>{}, element_values, output );
   output_data = ReadHostVector( output );
   success = Check(
      output_data[0] == Real( num_elements + 3 ) &&
         output_data[1] == Real( num_elements + 3 ),
      "Parallel host H1 additive scatter lost shared-DoF contributions." ) &&
      success;
   return success;
}

bool TestVectorGatherScatterMapping()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr Component0Tag c0{};
   constexpr Component1Tag c1{};
   constexpr GlobalIndex block_size = LocalDofCount< VectorShape >();

   Vector x_fe( vector_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = -1.0;
   }

   const GlobalIndex num_elements = vector_space.GetNumberOfFiniteElements();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex global_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         x_fe_data[global_index] =
            100.0 * element +
            static_cast< Real >( FlattenLocalDof( vector_space, c0, indices ) );
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex global_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         x_fe_data[global_index] =
            100.0 * element +
            static_cast< Real >( FlattenLocalDof( vector_space, c1, indices ) );
      });
   }

   DGGatherToBsr< decltype( vector_space ) > gather{ vector_space };
   Vector x_bsr( num_elements * block_size );
   gather( HostBSRBackend<>{}, x_fe, x_bsr );

   bool success = true;
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c0, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         success = Check(
            Near( x_bsr_data[bsr_index], x_fe_data[fe_index] ),
            "Vector gather component 0 mapping is wrong." ) && success;
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c1, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         success = Check(
            Near( x_bsr_data[bsr_index], x_fe_data[fe_index] ),
            "Vector gather component 1 mapping is wrong." ) && success;
      });
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = 7.0 + static_cast< Real >( 3 * i );
   }

   DGScatterFromBsr< decltype( vector_space ) > scatter{ vector_space };
   Vector y_fe( vector_space.GetNumberOfFiniteElementDofs() );
   scatter( HostBSRBackend<>{}, y_bsr, y_fe );

   const Real * y_fe_data = y_fe.ReadHostData();
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      UnitLoop< component_dof_shape_t< VectorShape, 0 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c0, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c0, element, indices );
         success = Check(
            Near( y_fe_data[fe_index], y_bsr_data[bsr_index] ),
            "Vector scatter component 0 mapping is wrong." ) && success;
      });

      UnitLoop< component_dof_shape_t< VectorShape, 1 > >( [&] ( auto... k )
      {
         const std::array< GlobalIndex, sizeof...( k ) > indices{
            static_cast< GlobalIndex >( k )... };
         const GlobalIndex bsr_index =
            element * block_size + FlattenLocalDof( vector_space, c1, indices );
         const GlobalIndex fe_index =
            GlobalDofIndex( vector_space, c1, element, indices );
         success = Check(
            Near( y_fe_data[fe_index], y_bsr_data[bsr_index] ),
            "Vector scatter component 1 mapping is wrong." ) && success;
      });
   }

   return success;
}

bool TestVectorH1GatherScatterMapping()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_h1_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   using VectorH1Space = std::remove_cvref_t< decltype( vector_h1_space ) >;
   using ShapeFunctions = typename VectorH1Space::finite_element_type::shape_functions;
   constexpr GlobalIndex block_size = LocalDofCount< ShapeFunctions >();
   const GlobalIndex num_elements =
      vector_h1_space.GetNumberOfFiniteElements();

   Vector x_fe( vector_h1_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 10.0 * static_cast< Real >( i + 1 );
   }

   VectorCGGatherToBsr< decltype( vector_h1_space ) > gather{
      vector_h1_space };
   Vector x_bsr( num_elements * block_size );
   gather( HostBSRBackend<>{}, x_fe, x_bsr );

   bool success = true;
   const Real expected_gather[] = {
      10.0, 20.0, 40.0, 50.0,
      20.0, 30.0, 50.0, 60.0
   };
   const Real * x_bsr_data = x_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < x_bsr.Size(); ++i )
   {
      success = Check(
         Near( x_bsr_data[i], expected_gather[i] ),
         "Vector H1 gather mapping is wrong." ) && success;
   }

   Vector y_bsr( num_elements * block_size );
   Real * y_bsr_data = y_bsr.WriteHostData();
   for ( GlobalIndex i = 0; i < y_bsr.Size(); ++i )
   {
      y_bsr_data[i] = static_cast< Real >( i + 1 );
   }

   Vector y_fe( vector_h1_space.GetNumberOfFiniteElementDofs() );
   Real * y_fe_data = y_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      y_fe_data[i] = 99.0;
   }

   VectorCGScatterFromBsr< decltype( vector_h1_space ) > scatter{
      vector_h1_space };
   scatter( HostBSRBackend<>{}, y_bsr, y_fe );

   const Real expected_scatter[] = {
      1.0, 7.0, 6.0,
      3.0, 11.0, 8.0
   };
   y_fe_data = y_fe.ReadWriteHostData();
   for ( GlobalIndex i = 0; i < y_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], expected_scatter[i] ),
         "Vector H1 scatter-add mapping or Set semantics is wrong." ) && success;
   }

   return success;
}

bool TestVectorSGBSRPermutationApply()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );

   constexpr GlobalIndex block_size = LocalDofCount< VectorShape >();
   auto identity_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >(
         vector_space.GetNumberOfFiniteElements(),
         block_size,
         block_size );
   FillIdentityBlocks( identity_bsr );

   Vector x_fe( vector_space.GetNumberOfFiniteElementDofs() );
   Real * x_fe_data = x_fe.WriteHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      x_fe_data[i] = 0.5 + static_cast< Real >( i * i );
   }

   Vector y_fe( vector_space.GetNumberOfFiniteElementDofs() );

   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      DGGatherToBsr< decltype( vector_space ) >,
      DGScatterFromBsr< decltype( vector_space ) > > sg_matrix(
         std::move( identity_bsr ),
         DGGatherToBsr< decltype( vector_space ) >{ vector_space },
         DGScatterFromBsr< decltype( vector_space ) >{ vector_space } );
   sg_matrix( x_fe, y_fe );

   bool success = true;
   const Real * y_fe_data = y_fe.ReadHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], x_fe_data[i] ),
         "Vector SGBSR identity BSR apply did not preserve FE vector values." ) && success;
   }

   y_fe = 3.0;
   ApplyAdd( sg_matrix, x_fe, y_fe );
   y_fe_data = y_fe.ReadHostData();
   for ( GlobalIndex i = 0; i < x_fe.Size(); ++i )
   {
      success = Check(
         Near( y_fe_data[i], x_fe_data[i] + 3.0 ),
         "Vector SGBSR ApplyAdd did not preserve the initial FE output." ) &&
         success;
   }

   return success;
}

bool TestVectorH1SGBSRCellMass()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::array< int, 4 > restriction_map{
      0, 1,
      1, 2
   };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto vector_h1_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > cells;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( cells, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( vector_h1_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;
   auto sgbsr_matrix =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule,
         HostBSRBackend<>{} );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   for ( Integer case_id = 0; case_id < 3; ++case_id )
   {
      Vector x( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_sgbsr( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_operator( vector_h1_space.GetNumberOfFiniteElementDofs() );
      Vector y_expected( vector_h1_space.GetNumberOfFiniteElementDofs() );

      FillVectorH1InputCase( x, case_id );
      y_sgbsr = 0.0;
      y_operator = 0.0;

      sgbsr_matrix( x, y_sgbsr );
      generic_operator( x, y_operator );
      ApplyTwoCellVectorH1P1MassReference( x, y_expected, h );

      success = CheckVectorNear(
         y_sgbsr,
         y_expected,
         "Vector H1 SGBSR action disagrees with the manual p1 mass reference." ) && success;
      success = CheckVectorNear(
         y_sgbsr,
         y_operator,
         "Vector H1 SGBSR action disagrees with GenericOperator." ) && success;

      Vector y_add_expected( y_expected.Size() );
      Real * y_add_expected_data = y_add_expected.WriteHostData();
      const Real * y_expected_read = y_expected.ReadHostData();
      for ( GlobalIndex i = 0; i < y_expected.Size(); ++i )
      {
         y_add_expected_data[i] = y_expected_read[i] + 2.0;
      }
      y_sgbsr = 2.0;
      ApplyAdd( sgbsr_matrix, x, y_sgbsr );
      success = CheckVectorNear(
         y_sgbsr,
         y_add_expected,
         "Vector H1 SGBSR ApplyAdd did not preserve the initial output." ) &&
         success;

      const Real * y_data = y_sgbsr.ReadHostData();
      const Real * expected_data = y_expected.ReadHostData();
      if ( case_id == 1 )
      {
         success = Check(
            Near( y_data[1], expected_data[1] + 2.0 ),
            "Vector H1 SGBSR did not accumulate the component 0 shared middle node." ) && success;
         success = Check(
            Near( y_data[3], 2.0 ) &&
            Near( y_data[4], 2.0 ) &&
            Near( y_data[5], 2.0 ),
            "Vector H1 SGBSR aliased component 0 input into component 1 output." ) && success;
      }
      else if ( case_id == 2 )
      {
         success = Check(
            Near( y_data[4], expected_data[4] + 2.0 ),
            "Vector H1 SGBSR did not accumulate the component 1 shared middle node." ) && success;
         success = Check(
            Near( y_data[0], 2.0 ) &&
            Near( y_data[1], 2.0 ) &&
            Near( y_data[2], 2.0 ),
            "Vector H1 SGBSR aliased component 1 input into component 0 output." ) && success;
      }
   }

   return success;
}

#ifdef GENDIL_USE_MFEM
bool TestMFEMSGBSRApply()
{
   bool success = true;

   auto identity_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >( 2, 1, 1 );
   auto * identity_values = ReadWriteHost( identity_bsr.values );
   identity_values[0] = 2.0;
   identity_values[1] = 3.0;
   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      IdentityBsrGather,
      IdentityBsrScatter > identity_matrix(
         std::move( identity_bsr ),
         IdentityBsrGather{},
         IdentityBsrScatter{} );

   mfem::Vector identity_x( 2 );
   auto * identity_x_data = WriteHostVector( identity_x );
   identity_x_data[0] = 4.0;
   identity_x_data[1] = 5.0;
   mfem::Vector identity_y( 2 );
   Apply( identity_matrix, identity_x, identity_y );
   const auto * identity_y_data = ReadHostVector( identity_y );
   success = Check(
      Near( identity_y_data[0], 8.0 ) &&
      Near( identity_y_data[1], 15.0 ),
      "Identity SGBSR MFEM apply produced the wrong result." ) && success;

   auto * identity_add_data = WriteHostVector( identity_y );
   identity_add_data[0] = 10.0;
   identity_add_data[1] = 20.0;
   ApplyAdd( identity_matrix, identity_x, identity_y );
   identity_y_data = ReadHostVector( identity_y );
   success = Check(
      Near( identity_y_data[0], 18.0 ) &&
      Near( identity_y_data[1], 35.0 ),
      "Identity SGBSR MFEM ApplyAdd produced the wrong result." ) && success;

   Vector mixed_gendil_x( 2 );
   auto * mixed_gendil_x_data = WriteHostVector( mixed_gendil_x );
   mixed_gendil_x_data[0] = 4.0;
   mixed_gendil_x_data[1] = 5.0;
   mfem::Vector mixed_mfem_y( 2 );
   Apply( identity_matrix, mixed_gendil_x, mixed_mfem_y );
   const auto * mixed_mfem_y_data = ReadHostVector( mixed_mfem_y );
   success = Check(
      Near( mixed_mfem_y_data[0], 8.0 ) &&
      Near( mixed_mfem_y_data[1], 15.0 ),
      "Stored-backend SGBSR Apply failed for GenDiL input and MFEM output." ) &&
      success;

   Vector mixed_gendil_y( 2 );
   auto * mixed_gendil_y_data = WriteHostVector( mixed_gendil_y );
   mixed_gendil_y_data[0] = 1.0;
   mixed_gendil_y_data[1] = 2.0;
   ApplyAdd(
      HostBSRBackend<>{},
      identity_matrix,
      identity_x,
      mixed_gendil_y );
   const auto * mixed_gendil_y_read = ReadHostVector( mixed_gendil_y );
   success = Check(
      Near( mixed_gendil_y_read[0], 9.0 ) &&
      Near( mixed_gendil_y_read[1], 17.0 ),
      "Explicit-backend SGBSR ApplyAdd failed for MFEM input and GenDiL "
      "output." ) && success;

   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto vector_space = MakeFiniteElementSpace( mesh, VectorFE{} );
   constexpr GlobalIndex block_size = LocalDofCount< VectorShape >();
   auto gathered_bsr =
      MakeBlockDiagonalDGBSRPattern< Real, GlobalIndex >(
         vector_space.GetNumberOfFiniteElements(),
         block_size,
         block_size );
   FillIdentityBlocks( gathered_bsr );
   SGBSRMatrix<
      BSRMatrix< Real, GlobalIndex >,
      DGGatherToBsr< decltype( vector_space ) >,
      DGScatterFromBsr< decltype( vector_space ) > > gathered_matrix(
         std::move( gathered_bsr ),
         DGGatherToBsr< decltype( vector_space ) >{ vector_space },
         DGScatterFromBsr< decltype( vector_space ) >{ vector_space } );

   const auto fe_size = static_cast< int >(
      vector_space.GetNumberOfFiniteElementDofs() );
   mfem::Vector gathered_x( fe_size );
   auto * gathered_x_data = WriteHostVector( gathered_x );
   for ( int i = 0; i < fe_size; ++i )
   {
      gathered_x_data[i] = 0.5 + static_cast< Real >( i );
   }
   mfem::Vector gathered_y( fe_size );
   Apply( gathered_matrix, gathered_x, gathered_y );
   const auto * gathered_y_data = ReadHostVector( gathered_y );
   for ( int i = 0; i < fe_size; ++i )
   {
      success = Check(
         Near( gathered_y_data[i], gathered_x_data[i] ),
         "Gathered/scattered SGBSR MFEM apply changed an identity result." ) &&
         success;
   }

   auto * gathered_add_data = WriteHostVector( gathered_y );
   for ( int i = 0; i < fe_size; ++i )
   {
      gathered_add_data[i] = 2.0;
   }
   ApplyAdd( gathered_matrix, gathered_x, gathered_y );
   gathered_y_data = ReadHostVector( gathered_y );
   for ( int i = 0; i < fe_size; ++i )
   {
      success = Check(
         Near( gathered_y_data[i], gathered_x_data[i] + 2.0 ),
         "Gathered/scattered SGBSR MFEM ApplyAdd lost the initial output." ) &&
         success;
   }

   return success;
}
#endif

} // namespace

int main()
{
   bool success = true;
   success = TestSGBSRFreeApplyVariants() && success;
   success = TestIdentityWrapperMatchesRawBsr() && success;
   success = TestSGBSRMoveAssignment() && success;
   success = TestRawBsrOperatorDelegatesToBackendApply() && success;
   success = TestScalarH1GatherScatterMapping() && success;
   success = TestSharedScalarH1HostScatter() && success;
   success = TestVectorGatherScatterMapping() && success;
   success = TestVectorH1GatherScatterMapping() && success;
   success = TestVectorSGBSRPermutationApply() && success;
   success = TestVectorH1SGBSRCellMass() && success;
#ifdef GENDIL_USE_MFEM
   success = TestMFEMSGBSRApply() && success;
#endif

   return success ? 0 : 1;
}
