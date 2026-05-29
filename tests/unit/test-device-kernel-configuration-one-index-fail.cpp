// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

using namespace gendil;

int main()
{
   using Layout = ThreadBlockLayout< 2 >;
   static constexpr Integer MaxSharedDimensions = 1;
   static constexpr Integer BatchSize = 2;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   Config::BlockLoop(
      4,
      [] GENDIL_HOST_DEVICE ( GlobalIndex )
      {
      } );

   return 0;
}
