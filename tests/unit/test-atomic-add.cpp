// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/vectoraccess.hpp>
#include <gendil/Utilities/Loop/kernelloop.hpp>
#include <gendil/Utilities/MathHelperFunctions/atomicadd.hpp>

#include <algorithm>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

static_assert(
   std::is_same_v<
      decltype(
         gendil::AtomicAddInPlace(
            std::declval< int & >(),
            std::declval< int >() ) ),
      void > );

bool TestAtomicAddCapture()
{
   constexpr int count = 4096;
   int value = 0;
   std::vector< int > previous_values( count );

   #pragma omp parallel for
   for ( int i = 0; i < count; ++i )
   {
      previous_values[i] = gendil::AtomicAdd( value, 1 );
   }

   std::sort( previous_values.begin(), previous_values.end() );
   bool success = value == count;
   for ( int i = 0; i < count; ++i )
   {
      success = previous_values[i] == i && success;
   }
   return success;
}

template < typename T >
bool TestHostAtomicAddInPlace()
{
   constexpr int count = 4096;
   T value = T( 0 );

   #pragma omp parallel for
   for ( int i = 0; i < count; ++i )
   {
      gendil::AtomicAddInPlace( value, T( 1 ) );
   }
   return value == T( count );
}

#ifdef GENDIL_USE_DEVICE
bool TestDeviceAtomicAddInPlace()
{
   constexpr GlobalIndex count = 4096;
   Vector value( 1 );
   auto * data = WriteKernelVector< true >( value );

   KernelLoop< true >(
      GlobalIndex( 1 ),
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex )
      {
         data[0] = Real( 0 );
      } );
   KernelLoop< true >(
      count,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex )
      {
         gendil::AtomicAddInPlace( data[0], Real( 1 ) );
      } );

   return ReadHostVector( value )[0] == Real( count );
}
#endif

} // namespace

int main()
{
   const bool capture_success = TestAtomicAddCapture();
   const bool integer_update_success = TestHostAtomicAddInPlace< int >();
   const bool real_update_success = TestHostAtomicAddInPlace< Real >();
#ifdef GENDIL_USE_DEVICE
   const bool device_update_success = TestDeviceAtomicAddInPlace();
#else
   constexpr bool device_update_success = true;
#endif

   if ( !capture_success )
   {
      std::cout << "Host AtomicAdd did not atomically capture old values.\n";
   }
   if ( !integer_update_success || !real_update_success )
   {
      std::cout << "Host AtomicAddInPlace lost updates under contention.\n";
   }
   if ( !device_update_success )
   {
      std::cout << "Device AtomicAddInPlace lost updates under contention.\n";
   }

   const bool success =
      capture_success &&
      integer_update_success &&
      real_update_success &&
      device_update_success;
   return success ? 0 : 1;
}
