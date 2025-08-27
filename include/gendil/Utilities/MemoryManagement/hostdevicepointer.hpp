// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Utilities/types.hpp"

namespace gendil
{

template < typename T >
struct HostDevicePointer
{
   using element_type = T;

   T * host_pointer = nullptr;
#ifdef GENDIL_USE_DEVICE
   T * device_pointer = nullptr;
#endif

   HostDevicePointer() = default;
   HostDevicePointer( const HostDevicePointer& ) = default;

   // Move constructor
   HostDevicePointer(HostDevicePointer&& other) noexcept
      : host_pointer(other.host_pointer)
#ifdef GENDIL_USE_DEVICE
      , device_pointer(other.device_pointer)
#endif
   {
      other.host_pointer = nullptr;
#ifdef GENDIL_USE_DEVICE
      other.device_pointer = nullptr;
#endif
   }

   // Move assignment operator
   HostDevicePointer& operator=(HostDevicePointer&& other) noexcept
   {
      if (this != &other)
      {
         host_pointer = other.host_pointer;
         other.host_pointer = nullptr;

#ifdef GENDIL_USE_DEVICE
         device_pointer = other.device_pointer;
         other.device_pointer = nullptr;
#endif
      }
      return *this;
   }

   GENDIL_HOST_DEVICE
   operator T * () const
   {
      #ifdef GENDIL_DEVICE_CODE
         return device_pointer;
      #else
         return host_pointer;
      #endif
   }

   GENDIL_HOST_DEVICE
   T& operator[]( GlobalIndex i ) const
   {
      #ifdef GENDIL_DEVICE_CODE
         return device_pointer[ i ];
      #else
         return host_pointer[ i ];
      #endif
   }
};

/// @brief Copies @a size elements from @a x.host_pointer to @a x.device_pointer
template < typename T >
void ToDevice( GlobalIndex size, const HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaMemcpy((void*)x.device_pointer, x.host_pointer, size * sizeof( T ), cudaMemcpyHostToDevice) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipMemcpy((void*)x.device_pointer, x.host_pointer, size * sizeof( T ), hipMemcpyHostToDevice) );
   #endif
}

/// @brief Copies @a size elements from @a x.device_pointer to @a x.host_pointer
template < typename T >
void ToHost( GlobalIndex size, const HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaMemcpy(x.host_pointer, x.device_pointer, size * sizeof( T ), cudaMemcpyDeviceToHost) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipMemcpy(x.host_pointer, x.device_pointer, size * sizeof( T ), hipMemcpyDeviceToHost) );
   #endif
}

template< typename T >
void FreeHostPointer( HostDevicePointer< T > & x )
{
   delete[] x.host_pointer;
}

template < typename T >
void FreeDevicePointer( HostDevicePointer< T > & x )
{
#if defined(GENDIL_USE_DEVICE)
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaFree( x.device_pointer ) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipFree( x.device_pointer ) );
   #endif
#endif
}

template < typename T >
void AllocateHostPointer( size_t size, HostDevicePointer< T > & x )
{
   x.host_pointer = new T[size];
}

template < typename T >
void AllocateDevicePointer( size_t size, HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      cudaError_t error = cudaMalloc(&x.device_pointer, size * sizeof( T ) );
   #elif defined(GENDIL_USE_HIP)
      hipError_t error = hipMalloc(&x.device_pointer, size * sizeof( T ) );
   #endif
}

} // namespace gendil
