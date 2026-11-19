// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

int main()
{
   GENDIL_BENCHMARK_FUNCTION;
   gendil::Vector values( 3 );
   values = 2.0;
   values *= 3.0;
   const auto * result = values.ReadHostData();
   return result[0] == 6.0 && result[1] == 6.0 && result[2] == 6.0 ? 0 : 1;
}
