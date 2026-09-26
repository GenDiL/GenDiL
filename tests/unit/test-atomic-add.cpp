// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/Vector/access.hpp>
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

template < bool OnDevice >
bool TestKernelAtomicAddInPlace()
{
   constexpr GlobalIndex count = 4096;
   Vector value( 1 );
   auto data = WriteKernelVector< OnDevice >( value );

   KernelLoop< OnDevice >(
      GlobalIndex( 1 ),
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex ) mutable
      {
         data[0] = Real( 0 );
      } );
   KernelLoop< OnDevice >(
      count,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex ) mutable
      {
         gendil::AtomicAddInPlace( data[0], Real( 1 ) );
      } );

   return ReadHostVector( value )[0] == Real( count );
}

} // namespace

int main()
{
   const bool capture_success = TestAtomicAddCapture();
   const bool integer_update_success = TestHostAtomicAddInPlace< int >();
   const bool real_update_success = TestHostAtomicAddInPlace< Real >();
   const bool host_kernel_update_success =
      TestKernelAtomicAddInPlace< false >();
#ifdef GENDIL_USE_DEVICE
   const bool device_update_success = TestKernelAtomicAddInPlace< true >();
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
   if ( !host_kernel_update_success )
   {
      std::cout << "Host kernel-view AtomicAddInPlace lost updates under "
         "contention.\n";
   }

   const bool success =
      capture_success &&
      integer_update_success &&
      real_update_success &&
      host_kernel_update_success &&
      device_update_success;
   return success ? 0 : 1;
}
