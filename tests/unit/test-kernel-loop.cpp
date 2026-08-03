// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/vectoraccess.hpp>
#include <gendil/Utilities/Loop/kernelloop.hpp>

#include <iostream>

using namespace gendil;

namespace
{

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

template < bool OnDevice >
bool TestKernelLoopPlacement()
{
   constexpr GlobalIndex size = 257;
   Vector values( size );
   auto * data = WriteKernelVector< OnDevice >( values );

   KernelLoop< OnDevice >(
      size,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex i )
      {
         data[i] = Real( 0 );
      } );
   KernelLoop< OnDevice >(
      size,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex i )
      {
         data[i] += Real( 1 );
      } );
   KernelLoop< OnDevice >(
      GlobalIndex( 0 ),
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex )
      {
         data[0] = Real( -1 );
      } );
   KernelLoop< OnDevice >(
      -3,
      [=] GENDIL_HOST_DEVICE ( const int )
      {
         data[0] = Real( -2 );
      } );

   const auto * host_data = ReadHostVector( values );
   bool success = true;
   for ( GlobalIndex i = 0; i < size; ++i )
   {
      success = Check(
         host_data[i] == Real( 1 ),
         "KernelLoop did not execute exactly once per valid index." ) &&
         success;
   }
   return success;
}

} // namespace

int main()
{
   bool success = TestKernelLoopPlacement< false >();
   success = TestKernelLoopPlacement< true >() && success;
   return success ? 0 : 1;
}
