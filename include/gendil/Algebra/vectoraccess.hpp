// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/Loop/kernelloop.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace gendil
{

/// Return the number of entries in a GenDiL vector.
inline size_t GetVectorSize( const Vector & vector )
{
   return vector.Size();
}

/// Return read-only host data, synchronizing the GenDiL vector to the host.
inline const Real * ReadHostVector( const Vector & vector )
{
   return vector.ReadHostData();
}

/// Return mutable host data, preserving values and invalidating device data.
inline Real * ReadWriteHostVector( Vector & vector )
{
   return vector.ReadWriteHostData();
}

/// Return write-only host data without synchronization, invalidating device data.
inline Real * WriteHostVector( Vector & vector )
{
   return vector.WriteHostData();
}

/// Return read-only device data, synchronizing the GenDiL vector to the device.
inline const Real * ReadDeviceVector( const Vector & vector )
{
   return vector.ReadDeviceData();
}

/// Return mutable device data, preserving values and invalidating host data.
inline Real * ReadWriteDeviceVector( Vector & vector )
{
   return vector.ReadWriteDeviceData();
}

/// Return write-only device data without synchronization, invalidating host data.
inline Real * WriteDeviceVector( Vector & vector )
{
   return vector.WriteDeviceData();
}

#ifdef GENDIL_USE_MFEM

/// Return the number of entries in an MFEM vector as a size_t.
inline size_t GetVectorSize( const mfem::Vector & vector )
{
   GENDIL_VERIFY(
      vector.Size() >= 0,
      "An MFEM vector cannot have a negative size." );
   return static_cast< size_t >( vector.Size() );
}

/// Return read-only host data, synchronizing the MFEM vector to the host.
inline const mfem::real_t * ReadHostVector( const mfem::Vector & vector )
{
   return vector.HostRead();
}

/// Return mutable host data, preserving values and invalidating device data.
inline mfem::real_t * ReadWriteHostVector( mfem::Vector & vector )
{
   return vector.HostReadWrite();
}

/// Return write-only host data without synchronization, invalidating device data.
inline mfem::real_t * WriteHostVector( mfem::Vector & vector )
{
   return vector.HostWrite();
}

/// Return read-only device data, synchronizing through MFEM's memory manager.
inline const mfem::real_t * ReadDeviceVector( const mfem::Vector & vector )
{
   return vector.Read();
}

/// Return mutable device data, synchronizing values and invalidating host data.
inline mfem::real_t * ReadWriteDeviceVector( mfem::Vector & vector )
{
   return vector.ReadWrite();
}

/// Return write-only device data without synchronization, invalidating host data.
inline mfem::real_t * WriteDeviceVector( mfem::Vector & vector )
{
   return vector.Write();
}

#endif

/**
 * Vector type exposing the complete host access interface.
 *
 * This constraint keeps sparse overloads from accepting unrelated vector
 * types and preserves their focused unsupported-vector diagnostics.
 */
template < typename VectorType >
concept HostAccessibleVector =
   requires(
      const std::remove_reference_t< VectorType > & input,
      std::remove_reference_t< VectorType > & output )
   {
      { GetVectorSize( input ) } -> std::same_as< size_t >;
      ReadHostVector( input );
      ReadWriteHostVector( output );
      WriteHostVector( output );
   };

/**
 * Vector type exposing the complete device access interface.
 *
 * Device access falls back to host storage where supported by the underlying
 * vector implementation.
 */
template < typename VectorType >
concept DeviceAccessibleVector =
   requires(
      const std::remove_reference_t< VectorType > & input,
      std::remove_reference_t< VectorType > & output )
   {
      { GetVectorSize( input ) } -> std::same_as< size_t >;
      ReadDeviceVector( input );
      ReadWriteDeviceVector( output );
      WriteDeviceVector( output );
   };

/// Vector type accessible in the memory space selected by OnDevice.
template < bool OnDevice, typename VectorType >
concept KernelAccessibleVector =
   ( OnDevice && DeviceAccessibleVector< VectorType > ) ||
   ( !OnDevice && HostAccessibleVector< VectorType > );

/// Return read-only data, synchronizing the selected kernel memory space.
template < bool OnDevice, typename VectorType >
requires KernelAccessibleVector< OnDevice, VectorType >
auto ReadKernelVector( const VectorType & vector )
{
   if constexpr ( OnDevice )
   {
      return ReadDeviceVector( vector );
   }
   else
   {
      return ReadHostVector( vector );
   }
}

/// Return mutable data, preserving values and invalidating the opposite space.
template < bool OnDevice, typename VectorType >
requires KernelAccessibleVector< OnDevice, VectorType >
auto ReadWriteKernelVector( VectorType & vector )
{
   if constexpr ( OnDevice )
   {
      return ReadWriteDeviceVector( vector );
   }
   else
   {
      return ReadWriteHostVector( vector );
   }
}

/// Return write-only data without syncing and invalidate the opposite space.
template < bool OnDevice, typename VectorType >
requires KernelAccessibleVector< OnDevice, VectorType >
auto WriteKernelVector( VectorType & vector )
{
   if constexpr ( OnDevice )
   {
      return WriteDeviceVector( vector );
   }
   else
   {
      return WriteHostVector( vector );
   }
}

/**
 * @brief Zero every vector entry in the selected kernel memory space.
 *
 * This operation acquires write-only storage because it overwrites the complete
 * vector. The selected memory space becomes valid and the opposite space is
 * invalidated according to the vector access contract.
 */
template < bool OnDevice, typename VectorType >
requires KernelAccessibleVector< OnDevice, VectorType >
void Zero( VectorType & vector )
{
   auto data = WriteKernelVector< OnDevice >( vector );
   KernelLoop< OnDevice >(
      GetVectorSize( vector ),
      [data] GENDIL_HOST_DEVICE ( const size_t i )
      {
         data[i] = Real{0};
      } );
}

} // namespace gendil
