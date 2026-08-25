// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <cmath>
#include <iostream>
#include <type_traits>

using namespace gendil;

namespace
{

#if defined( GENDIL_USE_DEVICE )
using ProjectionKernelPolicy =
   ThreadFirstKernelConfiguration< ThreadBlockLayout< 3, 3 >, 2 >;
#else
using ProjectionKernelPolicy = SerialKernelConfiguration;
#endif

template < bool UseLobatto >
bool RunProjectionCase( const char * label )
{
   constexpr Integer order = 2;
   constexpr Integer num_points = order + 1;
   constexpr Real integrated_interval_length = 3.0;
   constexpr Real integrated_reference_factor = 1.25;

   Cartesian1DMesh retained_mesh( 2.0, 1 );
   Cartesian1DMesh integrated_mesh( integrated_interval_length, 1 );
   auto trial_mesh =
      MakeCartesianProductMesh( retained_mesh, integrated_mesh );

   auto trial_finite_element = []
   {
      if constexpr ( UseLobatto )
      {
         return MakeLobattoFiniteElement(
            FiniteElementOrders< order, order >{} );
      }
      else
      {
         return MakeLegendreFiniteElement(
            FiniteElementOrders< order, order >{} );
      }
   }();
   auto test_finite_element = []
   {
      if constexpr ( UseLobatto )
      {
         return MakeLobattoFiniteElement( FiniteElementOrders< order >{} );
      }
      else
      {
         return MakeLegendreFiniteElement( FiniteElementOrders< order >{} );
      }
   }();
   auto trial_space =
      MakeFiniteElementSpace( trial_mesh, trial_finite_element );
   auto test_space =
      MakeFiniteElementSpace( retained_mesh, test_finite_element );

   auto trial_rule = []
   {
      if constexpr ( UseLobatto )
      {
         return MakeLobattoIntegrationRule(
            IntegrationRuleNumPoints< num_points, num_points >{} );
      }
      else
      {
         return MakeIntegrationRule(
            IntegrationRuleNumPoints< num_points, num_points >{} );
      }
   }();
   auto test_rule = []
   {
      if constexpr ( UseLobatto )
      {
         return MakeLobattoIntegrationRule(
            IntegrationRuleNumPoints< num_points >{} );
      }
      else
      {
         return MakeIntegrationRule(
            IntegrationRuleNumPoints< num_points >{} );
      }
   }();

   using NodalPoints = std::conditional_t<
      UseLobatto,
      GaussLobattoLegendrePoints< num_points >,
      GaussLegendrePoints< num_points > >;

   Vector input( trial_space.GetNumberOfFiniteElementDofs() );
   input = 0.0;
   auto input_view =
      MakeReadWriteElementTensorView< SerialKernelConfiguration >(
         trial_space,
         input );
   for ( LocalIndex integrated = 0; integrated < num_points; ++integrated )
   {
      const Real eta = NodalPoints::GetCoord( integrated );
      const Real integrated_factor = 1.0 + 0.5 * eta;
      for ( LocalIndex retained = 0; retained < num_points; ++retained )
      {
         const Real xi = NodalPoints::GetCoord( retained );
         const Real retained_factor = 2.0 + xi + xi * xi;
         input_view( retained, integrated, 0 ) =
            retained_factor * integrated_factor;
      }
   }

   Vector output( test_space.GetNumberOfFiniteElementDofs() );
   output = 0.0;
   auto projection = MakeNodalSubspaceProjection< ProjectionKernelPolicy >(
      trial_space,
      test_space,
      trial_rule,
      test_rule );
   projection( input, output );
   GENDIL_DEVICE_SYNC;

   const auto output_view =
      MakeReadOnlyElementTensorView< SerialKernelConfiguration >(
         test_space,
         output );
   constexpr Real tolerance = 1e-12;
   bool success = true;
   for ( LocalIndex retained = 0; retained < num_points; ++retained )
   {
      const Real xi = NodalPoints::GetCoord( retained );
      const Real expected =
         integrated_interval_length * integrated_reference_factor *
         ( 2.0 + xi + xi * xi );
      const Real actual = output_view( retained, 0 );
      if ( std::abs( actual - expected ) > tolerance )
      {
         std::cerr
            << label << " node " << retained << ": expected " << expected
            << ", got " << actual << '\n';
         success = false;
      }
   }
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = RunProjectionCase< false >( "Gauss-Legendre" ) && success;
   success = RunProjectionCase< true >( "Gauss-Lobatto-Legendre" ) && success;
   return success ? 0 : 1;
}
