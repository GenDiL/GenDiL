// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-batched-global-face-advection-operator.hpp"

int main( int argc, char ** argv )
{
   gendil::benchmarks::GlobalFaceAdvectionBenchmarkOptions options;
   const auto parse_result =
      gendil::benchmarks::ParseGlobalFaceAdvectionBenchmarkOptions(
         argc,
         argv,
         options,
         std::cerr );
   if ( parse_result ==
        gendil::benchmarks::BenchmarkOptionParseResult::exit_success )
   {
      return 0;
   }
   if ( parse_result ==
        gendil::benchmarks::BenchmarkOptionParseResult::exit_failure )
   {
      return 1;
   }

   gendil::benchmarks::PrintGlobalFaceAdvectionHeader();
   gendil::benchmarks::RunGlobalFaceAdvectionDimension< 3 >(
      options.target_num_dofs );
   return 0;
}
