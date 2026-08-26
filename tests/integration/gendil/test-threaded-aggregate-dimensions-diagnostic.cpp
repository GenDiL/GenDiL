// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <cmath>
#include <iostream>
#include <utility>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-threaded-aggregate-dimensions-diagnostic skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{

using TrialIntegrationRule =
   decltype( MakeIntegrationRule( IntegrationRuleNumPoints< 3, 3 >{} ) );
using TestIntegrationRule =
   decltype( MakeIntegrationRule( IntegrationRuleNumPoints< 3 >{} ) );
using Configuration =
   ThreadFirstKernelConfiguration< ThreadBlockLayout< 3, 3 >, 2 >;

GENDIL_HOST_DEVICE
Real TrialValue( const GlobalIndex retained, const GlobalIndex integrated )
{
   return Real( 10 * ( retained + 1 ) + integrated + 1 );
}

GENDIL_HOST_DEVICE
Real PoisonValue( const GlobalIndex retained )
{
   return Real( 100 + 7 * retained );
}

GENDIL_HOST_DEVICE
Real ExpectedValue( const GlobalIndex retained )
{
   return Real( 30 * ( retained + 1 ) + 6 );
}

void RunThreadedAggregation( HostDevicePointer< Real > output_data )
{
   constexpr size_t required_shared_mem =
      Product( typename TestIntegrationRule::points::num_points_tensor{} );
   using Context = KernelContext< Configuration, required_shared_mem >;

   Configuration::CandidateBlockLoop(
      1,
      [=] GENDIL_HOST_DEVICE () mutable
      {
         GENDIL_SHARED Real shared_memory[
            Context::shared_memory_block_size ];
         Context kernel_conf( shared_memory );

         auto poison = MakeSharedQuadraturePointValuesContainer(
            kernel_conf,
            TestIntegrationRule{} );
         QuadraturePointLoop< TestIntegrationRule >(
            kernel_conf,
            [&] ( auto const & quad_index )
            {
               WriteQuadratureLocalValues(
                  quad_index,
                  PoisonValue( quad_index[ 0 ] ),
                  poison,
                  std::make_index_sequence<
                     TestIntegrationRule::space_dim >{} );
            } );
         kernel_conf.Synchronize();
         kernel_conf.SharedAllocator.reset();

         auto input = MakeQuadraturePointValuesContainer(
            kernel_conf,
            TrialIntegrationRule{} );
         QuadraturePointLoop< TrialIntegrationRule >(
            kernel_conf,
            [&] ( auto const & quad_index )
            {
               WriteQuadratureLocalValues(
                  kernel_conf,
                  quad_index,
                  TrialValue( quad_index[ 0 ], quad_index[ 1 ] ),
                  input );
            } );

         auto result =
            AggregateDimensions<
               TrialIntegrationRule,
               TestIntegrationRule >(
                  kernel_conf,
                  input,
                  std::index_sequence< 0 >{} );

         Real * output = output_data;
         QuadraturePointLoop< TestIntegrationRule >(
            kernel_conf,
            [&] ( auto const & quad_index )
            {
               output[ quad_index[ 0 ] ] =
                  ReadQuadratureLocalValues(
                     kernel_conf,
                     quad_index,
                     result );
            } );
      } );
}

} // namespace

int main()
{
   DeviceBuffer< Real > output( 3, real_sentinel );
   RunThreadedAggregation( output.data );
   GENDIL_DEVICE_SYNC;
   output.CopyToHost();

   constexpr Real tolerance = 1.0e-12;
   bool success = true;
   for ( GlobalIndex retained = 0; retained < 3; ++retained )
   {
      const Real observed = output.data.host_pointer[ retained ];
      const Real expected = ExpectedValue( retained );
      if ( std::abs( observed - expected ) > tolerance )
      {
         std::cerr
            << "Threaded AggregateDimensions retained entry " << retained
            << ": expected " << expected << ", got " << observed << '\n';
         success = false;
      }
   }

   return success ? 0 : 1;
}

#endif
