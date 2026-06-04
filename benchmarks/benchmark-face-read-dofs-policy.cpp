// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-face-read-dofs-policy.hpp"

int main()
{
   gendil::benchmarks::PrintFaceHeader();
   gendil::benchmarks::RunFaceDimension< 1 >();
   gendil::benchmarks::RunFaceDimension< 2 >();
   gendil::benchmarks::RunFaceDimension< 3 >();
   gendil::benchmarks::RunFaceDimension< 4 >();
   gendil::benchmarks::RunFaceDimension< 5 >();
   gendil::benchmarks::RunFaceDimension< 6 >();
   return 0;
}
