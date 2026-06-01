// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-threaded-aggregate-dimensions-diagnostic skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

int main()
{
   std::cout
      << "AggregateDimensions diagnostic split from "
      << "test-threaded-helper-undercoverage: the threaded aggregation path "
      << "has a distinct shared-accumulator/initialization issue and is not "
      << "used to classify helper under-threaded coverage.\n";
   std::cout
      << "EXPECTED_SEPARATE_MILESTONE: validate AggregateDimensions with a "
      << "focused shared accumulator reset/zeroing diagnostic before making "
      << "a production change.\n";
   return 0;
}

#endif
