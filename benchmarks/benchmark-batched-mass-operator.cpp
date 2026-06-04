// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "benchmark-batched-mass-operator.hpp"

int main()
{
   gendil::benchmarks::PrintMassHeader();
   gendil::benchmarks::RunMassDimension< 1 >();
   gendil::benchmarks::RunMassDimension< 2 >();
   gendil::benchmarks::RunMassDimension< 3 >();
   return 0;
}
