// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-batched-mass-operator.hpp"

int main( int argc, char ** argv )
{
   gendil::benchmarks::MassBenchmarkOptions options;
   const auto parse_result =
      gendil::benchmarks::ParseMassBenchmarkOptions(
         argc,
         argv,
         options,
         std::cerr );
   if ( parse_result ==
        gendil::benchmarks::MassBenchmarkOptionParseResult::exit_success )
   {
      return 0;
   }
   if ( parse_result ==
        gendil::benchmarks::MassBenchmarkOptionParseResult::exit_failure )
   {
      return 1;
   }

   gendil::benchmarks::PrintMassHeader();
   gendil::benchmarks::RunMassDimension< 2 >( options.target_num_dofs );
   return 0;
}
