// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/vectoraccess.hpp>

#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

static_assert( HostAccessibleVector< Vector > );
static_assert( DeviceAccessibleVector< Vector > );
static_assert( KernelAccessibleVector< false, Vector > );
static_assert( KernelAccessibleVector< true, Vector > );
static_assert(
   std::is_same_v<
      decltype( ReadHostVector( std::declval< const Vector & >() ) ),
      const Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteHostVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( WriteHostVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadDeviceVector( std::declval< const Vector & >() ) ),
      const Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteDeviceVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( WriteDeviceVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype(
         ReadKernelVector< false >(
            std::declval< const Vector & >() ) ),
      const Real * > );
static_assert(
   std::is_same_v<
      decltype(
         ReadWriteKernelVector< true >(
            std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype(
         WriteKernelVector< false >(
            std::declval< Vector & >() ) ),
      Real * > );

#ifdef GENDIL_USE_MFEM
static_assert( HostAccessibleVector< mfem::Vector > );
static_assert( DeviceAccessibleVector< mfem::Vector > );
static_assert( KernelAccessibleVector< false, mfem::Vector > );
static_assert( KernelAccessibleVector< true, mfem::Vector > );
static_assert(
   std::is_same_v<
      decltype( ReadHostVector( std::declval< const mfem::Vector & >() ) ),
      const mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteHostVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( WriteHostVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadDeviceVector( std::declval< const mfem::Vector & >() ) ),
      const mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteDeviceVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( WriteDeviceVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype(
         ReadKernelVector< true >(
            std::declval< const mfem::Vector & >() ) ),
      const mfem::real_t * > );
#endif

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool TestGenDiLVectorAccess()
{
   Vector vector( 3 );
   bool success = Check(
      GetVectorSize( vector ) == size_t( 3 ),
      "GetVectorSize returned the wrong GenDiL vector size." );

   auto * host_data = WriteHostVector( vector );
   host_data[0] = 1.0;
   host_data[1] = 2.0;
   host_data[2] = 3.0;
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "WriteHostVector produced the wrong GenDiL validity state." ) &&
      success;

   (void) ReadDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadDeviceVector did not synchronize the GenDiL vector." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   (void) ReadWriteDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      !vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadWriteDeviceVector produced the wrong GenDiL validity state." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadWriteDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   const auto * synchronized_host = ReadHostVector( vector );
   success = Check(
      synchronized_host[0] == 1.0 &&
      synchronized_host[1] == 2.0 &&
      synchronized_host[2] == 3.0,
      "Vector access synchronization changed GenDiL values." ) &&
      success;

   (void) WriteDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      !vector.IsHostValid() && vector.IsDeviceValid(),
      "WriteDeviceVector produced the wrong GenDiL validity state." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU WriteDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   return success;
}

bool TestKernelVectorAccess()
{
   Vector vector( 2 );
   auto * host_data = WriteKernelVector< false >( vector );
   host_data[0] = Real( 7 );
   host_data[1] = Real( 9 );

   const auto * kernel_read = ReadKernelVector< true >( vector );
#ifdef GENDIL_USE_DEVICE
   bool success = Check(
      vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadKernelVector<true> did not synchronize device data." );
#else
   bool success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadKernelVector<true> should use host storage." );
#endif

   (void) kernel_read;
   auto * kernel_write = ReadWriteKernelVector< false >( vector );
   kernel_write[1] += Real( 1 );
   success = Check(
      ReadHostVector( vector )[1] == Real( 10 ),
      "Kernel vector access changed synchronized values." ) && success;
   return success;
}

#ifdef GENDIL_USE_MFEM
bool TestMFEMVectorAccess()
{
   mfem::Vector vector( 3 );
   auto * data = WriteHostVector( vector );
   data[0] = 4.0;
   data[1] = 5.0;
   data[2] = 6.0;

   const auto * read = ReadHostVector( vector );
   return
      Check(
         GetVectorSize( vector ) == size_t( 3 ),
         "GetVectorSize returned the wrong MFEM vector size." ) &&
      Check(
         read[0] == 4.0 && read[1] == 5.0 && read[2] == 6.0,
         "MFEM host vector access changed values." );
}
#endif

} // namespace

int main()
{
   bool success = TestGenDiLVectorAccess();
   success = TestKernelVectorAccess() && success;
#ifdef GENDIL_USE_MFEM
   success = TestMFEMVectorAccess() && success;
#endif
   return success ? 0 : 1;
}
