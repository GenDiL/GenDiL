// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)


#include <gendil/gendil.hpp>

using namespace gendil;

int main()
{
   std::cout << ">>> before GC\n";
   auto &gc = GarbageCollector::Instance();
   std::cout << ">>> after  GC\n";

   // minimal test:
   HostDevicePointer<int> test;
   AllocateHostPointer( 4, test );
   gc.RegisterHostDevicePtr( test );
   gc.Cleanup();

   return 0;
}