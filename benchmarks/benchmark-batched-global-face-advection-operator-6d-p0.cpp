// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-batched-global-face-advection-operator.hpp"

int main( int argc, char ** argv )
{
   return gendil::benchmarks::RunGlobalFaceAdvectionDriver< 6, 0 >(
      argc,
      argv );
}
