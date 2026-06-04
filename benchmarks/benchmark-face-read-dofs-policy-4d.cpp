// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-face-read-dofs-policy.hpp"

int main()
{
   gendil::benchmarks::PrintFaceHeader();
   gendil::benchmarks::RunFaceDimension< 4 >();
   return 0;
}
