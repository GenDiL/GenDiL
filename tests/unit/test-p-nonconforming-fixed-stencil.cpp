// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

namespace {

std::atomic_size_t allocation_count{ 0 };

} // namespace

void * operator new( const std::size_t size )
{
   allocation_count.fetch_add( 1, std::memory_order_relaxed );
   if ( void * result = std::malloc( size ) )
   {
      return result;
   }
   throw std::bad_alloc{};
}

void * operator new[]( const std::size_t size )
{
   return ::operator new( size );
}

void operator delete( void * pointer ) noexcept { std::free( pointer ); }
void operator delete[]( void * pointer ) noexcept { std::free( pointer ); }
void operator delete( void * pointer, std::size_t ) noexcept
{
   std::free( pointer );
}
void operator delete[]( void * pointer, std::size_t ) noexcept
{
   std::free( pointer );
}

namespace {

using namespace gendil;

constexpr Real tolerance = 1.0e-12;

struct Q1Stencil
{
   inline static constexpr std::array< std::size_t, 4 >
      row_entry_counts{ 1, 1, 1, 1 };
   inline static constexpr std::array< Real, 4 >
      entry_weights{ 1.0, 1.0, 1.0, 1.0 };
};

/** Q2 cell whose constrained interface is its negative-x face. */
struct Q2PlusStencil
{
   inline static constexpr std::array< std::size_t, 9 >
      row_entry_counts{ 1, 1, 1, 2, 1, 1, 1, 1, 1 };
   inline static constexpr std::array< Real, 10 >
      entry_weights{ 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1 };
};

/** Q2 cell whose constrained interface is its positive-x face. */
struct Q2MinusStencil
{
   inline static constexpr std::array< std::size_t, 9 >
      row_entry_counts{ 1, 1, 1, 1, 1, 2, 1, 1, 1 };
   inline static constexpr std::array< Real, 10 >
      entry_weights{ 1, 1, 1, 1, 1, 0.5, 0.5, 1, 1, 1 };
};

template < class T, std::size_t N >
HostDevicePointer< const T > BorrowHost( const std::array< T, N > & values )
{
   HostDevicePointer< const T > result{};
   result.host_pointer = values.data();
   return result;
}

bool Near( const Real left, const Real right )
{
   return std::abs( left - right ) <= tolerance;
}

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cerr << message << '\n';
   }
   return condition;
}

void FillDeterministicInput( Vector & input )
{
   Real * values = input.WriteHostData();
   for ( GlobalIndex i = 0; i < input.Size(); ++i )
   {
      values[i] =
         0.35 + 0.13 * static_cast< Real >( i ) +
         0.011 * static_cast< Real >( i * i );
   }
}

template < class Left, class Right >
bool VectorsNear(
   const Left & left,
   const Right & right,
   const char * message )
{
   const Real * left_values = left.ReadHostData();
   const Real * right_values = right.ReadHostData();
   bool success = left.Size() == right.Size();
   for ( GlobalIndex i = 0; i < left.Size() && success; ++i )
   {
      success = Near( left_values[i], right_values[i] );
   }
   return Check( success, message );
}

template < class Map >
Vector ApplyMap( const Map & map, const Vector & input )
{
   Vector output( GetOutputSize( map ) );
   output = 9.0;
   Apply( map, input, output );
   return output;
}

template < class RawCOO >
bool CheckRawTriplets( const RawCOO & raw )
{
   const auto data = GetHostReadView( raw );
   bool success = true;
   for ( GlobalIndex entry = 0; entry < raw.nnz_raw; ++entry )
   {
      success = Check(
         data.rows[entry] < raw.num_rows &&
            data.cols[entry] < raw.num_cols &&
            std::isfinite( data.values[entry] ),
         "Fixed-stencil RawCOO contains an invalid triplet." ) && success;
   }
   return success;
}

template < class COO >
bool HasSortedUniqueCoordinates( const COO & coo )
{
   const GlobalIndex * rows = ReadHost( coo.rows );
   const GlobalIndex * cols = ReadHost( coo.cols );
   for ( GlobalIndex entry = 1; entry < coo.nnz; ++entry )
   {
      if (
         rows[entry] < rows[entry - 1] ||
         ( rows[entry] == rows[entry - 1] &&
           cols[entry] <= cols[entry - 1] ) )
      {
         return false;
      }
   }
   return true;
}

struct OracleRow
{
   std::size_t count{};
   std::array< GlobalIndex, 2 > index{};
   std::array< Real, 2 > weight{};
};

template < std::size_t NumRows, std::size_t NumDofs1D >
void AddTensorMassToOracle(
   std::array< Real, 100 > & matrix,
   const std::array< OracleRow, NumRows > & restriction,
   const std::array< std::array< Real, NumDofs1D >, NumDofs1D > & mass_1d )
{
   static_assert( NumRows == NumDofs1D * NumDofs1D );
   for ( std::size_t test = 0; test < NumRows; ++test )
   {
      const std::size_t test_x = test % NumDofs1D;
      const std::size_t test_y = test / NumDofs1D;
      for ( std::size_t trial = 0; trial < NumRows; ++trial )
      {
         const std::size_t trial_x = trial % NumDofs1D;
         const std::size_t trial_y = trial / NumDofs1D;
         const Real local_value =
            mass_1d[test_x][trial_x] * mass_1d[test_y][trial_y];
         for ( std::size_t test_entry = 0;
               test_entry < restriction[test].count;
               ++test_entry )
         {
            for ( std::size_t trial_entry = 0;
                  trial_entry < restriction[trial].count;
                  ++trial_entry )
            {
               const GlobalIndex row = restriction[test].index[test_entry];
               const GlobalIndex col = restriction[trial].index[trial_entry];
               matrix[row * 10 + col] +=
                  restriction[test].weight[test_entry] *
                  local_value *
                  restriction[trial].weight[trial_entry];
            }
         }
      }
   }
}

template < bool Q2IsMinus >
std::array< Real, 100 > MakeIndependentCellMassOracle()
{
   constexpr std::array< std::array< Real, 2 >, 2 > q1_mass{
      std::array< Real, 2 >{ 2.0 / 6.0, 1.0 / 6.0 },
      std::array< Real, 2 >{ 1.0 / 6.0, 2.0 / 6.0 } };
   constexpr std::array< std::array< Real, 3 >, 3 > q2_mass{
      std::array< Real, 3 >{ 4.0 / 30.0, 2.0 / 30.0, -1.0 / 30.0 },
      std::array< Real, 3 >{ 2.0 / 30.0, 16.0 / 30.0, 2.0 / 30.0 },
      std::array< Real, 3 >{ -1.0 / 30.0, 2.0 / 30.0, 4.0 / 30.0 } };

   std::array< OracleRow, 4 > q1{};
   std::array< OracleRow, 9 > q2{};
   if constexpr ( Q2IsMinus )
   {
      q1 = {
         OracleRow{ 1, { 1, 0 }, { 1, 0 } },
         OracleRow{ 1, { 4, 0 }, { 1, 0 } },
         OracleRow{ 1, { 3, 0 }, { 1, 0 } },
         OracleRow{ 1, { 5, 0 }, { 1, 0 } } };
      q2 = {
         OracleRow{ 1, { 6, 0 }, { 1, 0 } },
         OracleRow{ 1, { 7, 0 }, { 1, 0 } },
         OracleRow{ 1, { 1, 0 }, { 1, 0 } },
         OracleRow{ 1, { 8, 0 }, { 1, 0 } },
         OracleRow{ 1, { 9, 0 }, { 1, 0 } },
         OracleRow{ 2, { 1, 3 }, { 0.5, 0.5 } },
         OracleRow{ 1, { 0, 0 }, { 1, 0 } },
         OracleRow{ 1, { 2, 0 }, { 1, 0 } },
         OracleRow{ 1, { 3, 0 }, { 1, 0 } } };
   }
   else
   {
      q1 = {
         OracleRow{ 1, { 0, 0 }, { 1, 0 } },
         OracleRow{ 1, { 1, 0 }, { 1, 0 } },
         OracleRow{ 1, { 2, 0 }, { 1, 0 } },
         OracleRow{ 1, { 3, 0 }, { 1, 0 } } };
      q2 = {
         OracleRow{ 1, { 1, 0 }, { 1, 0 } },
         OracleRow{ 1, { 4, 0 }, { 1, 0 } },
         OracleRow{ 1, { 5, 0 }, { 1, 0 } },
         OracleRow{ 2, { 1, 3 }, { 0.5, 0.5 } },
         OracleRow{ 1, { 6, 0 }, { 1, 0 } },
         OracleRow{ 1, { 7, 0 }, { 1, 0 } },
         OracleRow{ 1, { 3, 0 }, { 1, 0 } },
         OracleRow{ 1, { 8, 0 }, { 1, 0 } },
         OracleRow{ 1, { 9, 0 }, { 1, 0 } } };
   }

   std::array< Real, 100 > matrix{};
   AddTensorMassToOracle( matrix, q1, q1_mass );
   AddTensorMassToOracle( matrix, q2, q2_mass );
   return matrix;
}

Vector ApplyDenseOracle(
   const std::array< Real, 100 > & matrix,
   const Vector & input )
{
   Vector output( 10 );
   const Real * x = input.ReadHostData();
   Real * y = output.WriteHostData();
   for ( GlobalIndex row = 0; row < 10; ++row )
   {
      y[row] = 0.0;
      for ( GlobalIndex col = 0; col < 10; ++col )
      {
         y[row] += matrix[row * 10 + col] * x[col];
      }
   }
   return output;
}

template < class Space, class Mesh, class Form, class Rule >
auto AssemblePartSGBSR(
   const Space & space,
   const Mesh & mesh,
   const Form & form,
   const Rule & rule )
{
   const auto context = MakeWeakFormContext(
      MakeTrialField< "u" >( space ),
      MakeIntegrationDomain< "domain" >( mesh ) );
   return GenericAssembly<
      MatrixAssemblyType::SGBSR,
      SerialKernelConfiguration >(
         form,
         context,
         rule,
         HostBSRBackend<>{} );
}

template < bool Q2IsMinus >
bool TestQ1Q2Case()
{
   const Cartesian2DMesh left_mesh(
      1.0, 1, 1, Point< 2 >{ 0.0, 0.0 } );
   const Cartesian2DMesh right_mesh(
      1.0, 1, 1, Point< 2 >{ 1.0, 0.0 } );
   const CartesianIntermeshFaceConnectivity< 2, 2 > interface_faces(
      { 1, 1 }, { 1, 1 } );
   const auto partition = MakePartition(
      MakeCellPart( left_mesh ),
      MakeCellPart( right_mesh ),
      MakeInteriorFacePart< 0, 1 >( interface_faces ) );

   const auto q1_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 1, 1 >{} );
   const auto q2_fe =
      MakeLobattoFiniteElement( FiniteElementOrders< 2, 2 >{} );

   const std::array< GlobalIndex, 4 > q1_left_indices{ 0, 1, 2, 3 };
   const std::array< GlobalIndex, 10 > q2_right_indices{
      1, 4, 5, 1, 3, 6, 7, 3, 8, 9 };
   const std::array< GlobalIndex, 10 > q2_left_indices{
      6, 7, 1, 8, 9, 1, 3, 0, 2, 3 };
   const std::array< GlobalIndex, 4 > q1_right_indices{ 1, 4, 3, 5 };

   const auto q1_left = MakeElementDoFRestriction(
      left_mesh,
      q1_fe,
      FixedStencilRestrictionSpecification< Q1Stencil >{
         BorrowHost( q1_left_indices ), 4, 4, 10 } );
   const auto q2_right = MakeElementDoFRestriction(
      right_mesh,
      q2_fe,
      FixedStencilRestrictionSpecification< Q2PlusStencil >{
         BorrowHost( q2_right_indices ), 10, 6, 10 } );
   const auto q2_left = MakeElementDoFRestriction(
      left_mesh,
      q2_fe,
      FixedStencilRestrictionSpecification< Q2MinusStencil >{
         BorrowHost( q2_left_indices ), 10, 6, 10 } );
   const auto q1_right = MakeElementDoFRestriction(
      right_mesh,
      q1_fe,
      FixedStencilRestrictionSpecification< Q1Stencil >{
         BorrowHost( q1_right_indices ), 4, 4, 10 } );

   using Q1Restriction = std::remove_cvref_t< decltype( q1_left ) >;
   using Q2PlusRestriction = std::remove_cvref_t< decltype( q2_right ) >;
   using Q2MinusRestriction = std::remove_cvref_t< decltype( q2_left ) >;
   static_assert( static_restriction_entry_count_v< Q1Restriction > == 1 );
   static_assert(
      static_restriction_entry_count_v< Q2PlusRestriction > ==
         dynamic_restriction_entry_count );
   static_assert(
      static_restriction_entry_count_for_v<
         Q2PlusRestriction,
         NativeRestrictionRowTag< 0, 3 > > == 2 );
   static_assert(
      static_restriction_entry_count_for_v<
         Q2MinusRestriction,
         NativeRestrictionRowTag< 0, 5 > > == 2 );

   const auto space = [&]
   {
      if constexpr ( Q2IsMinus )
      {
         return MakeMixedFiniteElementSpace(
            partition,
            std::tuple{ q2_fe, q1_fe },
            std::tuple{ q2_left, q1_right } );
      }
      else
      {
         return MakeMixedFiniteElementSpace(
            partition,
            std::tuple{ q1_fe, q2_fe },
            std::tuple{ q1_left, q2_right } );
      }
   }();

   const auto & minus_space = space.template GetCellFiniteElementSpace< 0 >();
   const auto & plus_space = space.template GetCellFiniteElementSpace< 1 >();
   bool success = true;
   success = Check(
      GetNumberOfGlobalDofs( space ) == 10 &&
         GetAlgebraicDofExtent( space ) == 10,
      "Q1-Q2 mixed space has incorrect logical or algebraic dimensions." ) &&
      success;

   const auto & q2_restriction = [&] () -> const auto &
   {
      if constexpr ( Q2IsMinus )
      {
         return minus_space.restriction;
      }
      else
      {
         return plus_space.restriction;
      }
   }();
   const std::array< GlobalIndex, 2 > q2_midpoint{
      Q2IsMinus ? GlobalIndex{ 2 } : GlobalIndex{ 0 },
      1 };
   const std::array< Real, 10 > probe_values{
      0.0, 2.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
   success = Check(
      Near( GatherRestrictionRow(
         q2_restriction,
         0,
         q2_midpoint,
         probe_values ), 4.0 ),
      "Q2 interface midpoint did not gather the Q1 endpoint average." ) &&
      success;
   std::array< Real, 10 > midpoint_scatter{};
   ScatterAddRestrictionRowAdjoint(
      q2_restriction,
      0,
      q2_midpoint,
      8.0,
      midpoint_scatter );
   success = Check(
      Near( midpoint_scatter[1], 4.0 ) &&
         Near( midpoint_scatter[3], 4.0 ),
      "Q2 interface midpoint did not scatter half to each Q1 endpoint." ) &&
      success;

   Real ex_dot_z = 0.0;
   Real x_dot_etz = 0.0;
   std::array< Real, 10 > etz{};
   for ( GlobalIndex row = 0; row < 9; ++row )
   {
      const auto local_dof = UnflattenMultiIndex<
         typename std::remove_cvref_t< decltype( q2_restriction ) >::
            dof_shape_type >( row );
      const Real local_value = 0.2 + 0.17 * static_cast< Real >( row );
      ex_dot_z += GatherRestrictionRow(
         q2_restriction,
         0,
         local_dof,
         probe_values ) * local_value;
      ScatterAddRestrictionRowAdjoint(
         q2_restriction,
         0,
         local_dof,
         local_value,
         etz );
   }
   for ( GlobalIndex i = 0; i < 10; ++i )
   {
      x_dot_etz += probe_values[i] * etz[i];
   }
   success = Check(
      Near( ex_dot_z, x_dot_etz ),
      "Q2 fixed-stencil restriction failed the adjoint identity." ) &&
      success;

   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   const auto cell_form = integrate( Cells< "domain" >{}, u * v );
   const auto context = MakeWeakFormContext(
      MakeTrialField< "u" >( space ),
      MakeIntegrationDomain< "domain" >( partition ) );
   const auto rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   const auto matrix_free = MakeGenericOperator< SerialKernelConfiguration >(
      cell_form, context, rule );
   const auto raw = GenericAssembly<
      MatrixAssemblyType::RawCOO,
      SerialKernelConfiguration >( cell_form, context, rule );
   const auto finalized = FinalizeRawCOOToCOOHost( raw );
   const auto coo = GenericAssembly<
      MatrixAssemblyType::COO,
      SerialKernelConfiguration >( cell_form, context, rule );
   const auto csr = GenericAssembly<
      MatrixAssemblyType::CSR,
      SerialKernelConfiguration >( cell_form, context, rule );
   const auto csc = GenericAssembly<
      MatrixAssemblyType::CSC,
      SerialKernelConfiguration >( cell_form, context, rule );
   const auto sgbsr_minus = AssemblePartSGBSR(
      minus_space,
      left_mesh,
      cell_form,
      rule );
   const auto sgbsr_plus = AssemblePartSGBSR(
      plus_space,
      right_mesh,
      cell_form,
      rule );

   success = Check(
      raw.num_rows == 10 && raw.num_cols == 10 && raw.nnz_raw == 116,
      "Q1-Q2 cell RawCOO dimensions or exact capacity are incorrect." ) &&
      success;
   success = CheckRawTriplets( raw ) && success;
   success = Check(
      raw.nnz_raw > finalized.nnz &&
         HasSortedUniqueCoordinates( finalized ),
      "Q1-Q2 cell RawCOO duplicates were not canonically reduced." ) &&
      success;

   Vector input( 10 );
   FillDeterministicInput( input );
   const Vector expected = ApplyDenseOracle(
      MakeIndependentCellMassOracle< Q2IsMinus >(), input );
   const Vector y_matrix_free = ApplyMap( matrix_free, input );
   const Vector y_finalized = ApplyMap( finalized, input );
   const Vector y_coo = ApplyMap( coo, input );
   const Vector y_csr = ApplyMap( csr, input );
   const Vector y_csc = ApplyMap( csc, input );
   Vector y_sgbsr( 10 );
   y_sgbsr = 0.0;
   ApplyAdd( sgbsr_minus, input, y_sgbsr );
   ApplyAdd( sgbsr_plus, input, y_sgbsr );
   success = VectorsNear(
      y_matrix_free, expected,
      "Q1-Q2 matrix-free cell mass disagrees with the dense oracle." ) && success;
   success = VectorsNear(
      y_finalized, expected,
      "Q1-Q2 finalized RawCOO cell mass disagrees with the dense oracle." ) && success;
   success = VectorsNear(
      y_coo, expected,
      "Q1-Q2 COO cell mass disagrees with the dense oracle." ) && success;
   success = VectorsNear(
      y_csr, expected,
      "Q1-Q2 CSR cell mass disagrees with the dense oracle." ) && success;
   success = VectorsNear(
      y_csc, expected,
      "Q1-Q2 CSC cell mass disagrees with the dense oracle." ) && success;
   success = VectorsNear(
      y_sgbsr, expected,
      "Q1-Q2 SGBSR cell mass disagrees with the dense oracle." ) && success;

   Vector add_output( 10 );
   add_output = 2.0;
   ApplyAdd( csr, input, add_output );
   const Real * add_values = add_output.ReadHostData();
   const Real * expected_values = expected.ReadHostData();
   for ( GlobalIndex i = 0; i < 10; ++i )
   {
      success = Check(
         Near( add_values[i], 2.0 + expected_values[i] ),
         "Q1-Q2 CSR ApplyAdd did not preserve the destination." ) && success;
   }

   const auto check_facet_form = [&] (
      const auto & facet_form,
      const bool expect_zero,
      const char * label )
   {
      const auto facet_raw = GenericAssembly<
         MatrixAssemblyType::RawCOO,
         SerialKernelConfiguration >( facet_form, context, rule );
      const auto facet_coo = FinalizeRawCOOToCOOHost( facet_raw );
      const auto facet_csr = GenericAssembly<
         MatrixAssemblyType::CSR,
         SerialKernelConfiguration >( facet_form, context, rule );
      const auto facet_csc = GenericAssembly<
         MatrixAssemblyType::CSC,
         SerialKernelConfiguration >( facet_form, context, rule );
      const auto facet_matrix_free =
         MakeGenericOperator< SerialKernelConfiguration >(
            facet_form, context, rule );

      bool facet_success = Check(
         facet_raw.num_rows == 10 && facet_raw.num_cols == 10 &&
            facet_raw.nnz_raw == 196,
         "Q1-Q2 facet RawCOO dimensions or directed-segment capacity are incorrect." );
      // RawCOO intentionally reserves dense element-pair blocks. For the
      // four directed segments this is 4x4, 4x10, 10x4, and 10x10. Reducing
      // this to the 2/4 Lobatto trace rows would require basis-specific trace
      // sparsity metadata and would not be valid for general modal bases.
      facet_success = CheckRawTriplets( facet_raw ) && facet_success;
      facet_success = Check(
         HasSortedUniqueCoordinates( facet_coo ),
         "Q1-Q2 facet COO is not canonical." ) && facet_success;

      const Vector facet_expected = ApplyMap( facet_matrix_free, input );
      const Vector facet_coo_output = ApplyMap( facet_coo, input );
      const Vector facet_csr_output = ApplyMap( facet_csr, input );
      const Vector facet_csc_output = ApplyMap( facet_csc, input );
      facet_success = VectorsNear(
         facet_coo_output, facet_expected, label ) && facet_success;
      facet_success = VectorsNear(
         facet_csr_output, facet_expected, label ) && facet_success;
      facet_success = VectorsNear(
         facet_csc_output, facet_expected, label ) && facet_success;
      Real norm2 = 0.0;
      for ( GlobalIndex i = 0; i < 10; ++i )
      {
         const Real value = facet_expected.ReadHostData()[i];
         norm2 += value * value;
      }
      facet_success = Check(
         expect_zero ? norm2 <= tolerance * tolerance : norm2 > 1.0e-16,
         expect_zero ?
            "Q1-Q2 continuous trace has a nonzero jump action." :
            "Q1-Q2 average facet action is unexpectedly zero." ) &&
         facet_success;
      return facet_success;
   };

   const auto jump_form = integrate(
      InteriorFacets< "domain" >{}, jump( u ) * jump( v ) );
   const auto average_form = integrate(
      InteriorFacets< "domain" >{}, average( u ) * average( v ) );
   success = check_facet_form(
      jump_form,
      true,
      "Q1-Q2 jump sparse action disagrees with matrix-free." ) && success;
   success = check_facet_form(
      average_form,
      false,
      "Q1-Q2 average sparse action disagrees with matrix-free." ) && success;

   Vector hot_output( 10 );
   const std::size_t allocations_before =
      allocation_count.load( std::memory_order_relaxed );
   for ( int repetition = 0; repetition < 8; ++repetition )
   {
      Apply( matrix_free, input, hot_output );
      Apply( HostCSRBackend<>{}, csr, input, hot_output );
      Apply( sgbsr_minus, input, hot_output );
      ApplyAdd( sgbsr_plus, input, hot_output );
   }
   success = Check(
      allocation_count.load( std::memory_order_relaxed ) == allocations_before,
      "Fixed-stencil host operator application allocated in the hot path." ) &&
      success;
   return success;
}

} // namespace

int main()
{
   return TestQ1Q2Case< false >() && TestQ1Q2Case< true >() ? 0 : 1;
}
