// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

using namespace gendil;

int main(int argc, char *argv[])
{
   ///////////////////
   // Integration Rule
   constexpr Integer order = 1;
   constexpr Integer num_quad = order + 2;

   // Space number of quadrature points
   constexpr Integer num_quad_1 = num_quad;
   constexpr Integer num_quad_2 = num_quad;
   constexpr Integer num_quad_3 = num_quad;
   constexpr Integer num_quad_4 = num_quad;
   constexpr Integer num_quad_5 = num_quad;
   constexpr Integer num_quad_6 = num_quad;
   IntegrationRuleNumPoints< num_quad_1, num_quad_2, num_quad_3, num_quad_4, num_quad_5, num_quad_6 > num_quads;

   // High-dimension integration rule
   auto int_rule = MakeIntegrationRule( num_quads );
   using int_rule_type = decltype( int_rule );

   QuadraturePointLoop< int_rule_type >( []( const auto & quad_index ){
      std::cout << quad_index << ": " << int_rule_type::GetPoint( quad_index ) << std::endl;
   } );

   return 0;
}
