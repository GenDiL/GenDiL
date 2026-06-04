// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-batched-global-face-advection-operator.hpp"

int main()
{
   gendil::benchmarks::PrintGlobalFaceAdvectionHeader();
   gendil::benchmarks::RunGlobalFaceAdvectionDimension< 3 >();
   return 0;
}
