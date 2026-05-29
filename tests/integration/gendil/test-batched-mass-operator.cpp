// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-mass-operator skipped because GENDIL_USE_DEVICE "
      << "is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{

template < typename VectorType >
Real AbsoluteL2Error( const VectorType & a, const VectorType & b )
{
   GENDIL_VERIFY( a.Size() == b.Size(), "Vector sizes do not match." );

   Real err_sq = 0.0;
   for ( Integer i = 0; i < a.Size(); ++i )
   {
      const Real d = a[ i ] - b[ i ];
      err_sq += d * d;
   }
   return std::sqrt( err_sq );
}

template < typename VectorType >
Real RelativeL2Error( const VectorType & a, const VectorType & b )
{
   const Real abs_err = AbsoluteL2Error( a, b );

   Real norm_b_sq = 0.0;
   for ( Integer i = 0; i < b.Size(); ++i )
   {
      norm_b_sq += b[ i ] * b[ i ];
   }

   const Real norm_b = std::sqrt( norm_b_sq );
   if ( norm_b == 0.0 )
   {
      return abs_err;
   }
   return abs_err / norm_b;
}

void FillDeterministicInput( Vector & x )
{
   for ( Integer i = 0; i < x.Size(); ++i )
   {
      x[ i ] =
         0.125 +
         0.03125 * static_cast< Real >( i ) +
         0.17 * static_cast< Real >( ( i * 7 ) % 11 );
   }
}

template < typename KernelPolicy, typename FiniteElementSpace, typename Rule, typename Sigma >
Vector ApplyMass(
   const FiniteElementSpace & fe_space,
   const Rule & integration_rule,
   Sigma & sigma,
   const Vector & x )
{
   Vector y( fe_space.GetNumberOfFiniteElementDofs() );
   y = 0.0;

   auto op =
      MakeMassFiniteElementOperator< KernelPolicy >(
         fe_space,
         integration_rule,
         sigma );
   op( x, y );
   GENDIL_DEVICE_SYNC;

   return y;
}

template < typename A, typename B >
bool CheckClose(
   const char * label,
   const A & observed,
   const B & expected,
   const Real tolerance )
{
   const Real rel_error = RelativeL2Error( observed, expected );
   std::cout << label << " relative L2 error = " << rel_error << '\n';
   if ( rel_error > tolerance )
   {
      std::cout << "FAILED: " << label << " exceeded tolerance "
                << tolerance << ".\n";
      return false;
   }
   return true;
}

template < typename Layout, Integer MaxSharedDimensions, Integer BatchSize >
bool RunThreadedMassCase()
{
   static constexpr Integer num_cells = 10;
   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 1;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 1.0 + 0.25 * x + x * x;
   };

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   using LegacyConfig =
      ThreadFirstKernelConfiguration< Layout, MaxSharedDimensions >;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   auto y_legacy =
      ApplyMass< LegacyConfig >( fe_space, integration_rule, sigma, x );
   auto y_batch1 =
      ApplyMass< DeviceBatch1 >( fe_space, integration_rule, sigma, x );
   auto y_batchn =
      ApplyMass< DeviceBatchN >( fe_space, integration_rule, sigma, x );

   constexpr Real tolerance = 1e-12;
   bool success = true;
   success =
      CheckClose( "DeviceBatch1 vs LegacyConfig", y_batch1, y_legacy, tolerance ) &&
      success;
   success =
      CheckClose( "DeviceBatchN vs LegacyConfig", y_batchn, y_legacy, tolerance ) &&
      success;
   return success;
}

template < Integer BatchSize >
bool RunRegisterOnlyMassCase()
{
   static constexpr Integer num_cells = 10;
   static constexpr Integer order = 3;
   static constexpr Integer num_quad_1d = order + 1;

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLobattoFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   IntegrationRuleNumPoints< num_quad_1d > num_quads;
   auto integration_rule = MakeIntegrationRule( num_quads );

   auto sigma = [] GENDIL_HOST_DEVICE (
      const std::array< Real, 1 > & X ) -> Real
   {
      const Real x = X[ 0 ];
      return 0.75 + x + 0.5 * x * x;
   };

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   using Layout = ThreadBlockLayout<>;
   static constexpr Integer MaxSharedDimensions = 0;
   using DeviceBatch1 =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, 1 >;
   using DeviceBatchN =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   auto y_batch1 =
      ApplyMass< DeviceBatch1 >( fe_space, integration_rule, sigma, x );
   auto y_batchn =
      ApplyMass< DeviceBatchN >( fe_space, integration_rule, sigma, x );

   constexpr Real tolerance = 1e-12;
   return CheckClose(
      "Register-only DeviceBatchN vs DeviceBatch1",
      y_batchn,
      y_batch1,
      tolerance );
}

bool TestThreadedMass()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = 4;

   static_assert( Layout::GetNumberOfThreads() == 4 );

   return RunThreadedMassCase<
      Layout,
      MaxSharedDimensions,
      BatchSize >();
}

bool TestRegisterOnlyMass()
{
   static constexpr Integer BatchSize = device_warp_size;
   static_assert( BatchSize >= 1 );

   return RunRegisterOnlyMassCase< BatchSize >();
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedMass() && success;
   success = TestRegisterOnlyMass() && success;

   return success ? 0 : 1;
}

#endif
