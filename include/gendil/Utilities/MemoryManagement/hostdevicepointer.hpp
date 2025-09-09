// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

/**
 * @file hostdevicepointer.hpp
 * @brief Pointer-like handle that abstracts the physical memory space (host vs device)
 *        behind a single logical address space.
 *
 * `HostDevicePointer<T>` stores both a host pointer and, when a GPU backend is enabled,
 * a device pointer. In code compiled for the device (`GENDIL_DEVICE_CODE`) it implicitly
 * resolves to the device pointer; otherwise it resolves to the host pointer. This lets
 * higher-level objects (e.g., read-only views) be written once and used transparently
 * on CPU or GPU without `#ifdef`s.
 *
 * Design goals:
 *  - **Single logical space**: user code treats memory uniformly; the compilation
 *    environment selects the correct physical pointer.
 *  - **Transparent read-only views**: build const-correct, non-mutable wrappers atop
 *    `HostDevicePointer<const T>` to provide safe access on both CPU and GPU.
 *  - **No RAII**: allocation, deallocation, and H↔D copies are explicit and under
 *    the caller's control (see Allocate/Free, ToDevice/ToHost).
 *
 * @note The handle is shallow and trivially copyable; use move semantics to transfer
 *       ownership. Synchronization of contents (ToDevice/ToHost) is explicit.
 */


#include "gendil/Utilities/types.hpp"

namespace gendil
{

/**
 * @brief Pair of raw pointers to the same logical buffer in host/device memory,
 *        with pointer selection driven by the compilation environment.
 *
 * In host compilation, `operator T*()` and `operator[]` refer to `host_pointer`.
 * In device compilation (`GENDIL_DEVICE_CODE`), they refer to `device_pointer`.
 * This enables writing device-agnostic code.
 *
 * @warning The handle does not own memory; lifetime and transfers (ToDevice/ToHost)
 *          are explicit. Copying is shallow; free exactly once per allocation.
 */
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

   /**
    * @brief Implicit pointer conversion to the active memory space.
    *
    * Returns `device_pointer` when compiling for device (`GENDIL_DEVICE_CODE`),
    * otherwise returns `host_pointer`.
    *
    * @note Intended for convenience inside kernels and host code; ensure the
    *       corresponding pointer is valid in the compilation context.
    */
   GENDIL_HOST_DEVICE
   operator T * () const
   {
      #ifdef GENDIL_DEVICE_CODE
         return device_pointer;
      #else
         return host_pointer;
      #endif
   }

   /**
    * @brief Element access routed to the active memory space.
    *
    * Indexes `device_pointer` under `GENDIL_DEVICE_CODE`, otherwise `host_pointer`.
    * Behavior is undefined if the corresponding pointer is null.
    *
    * @param i Global element offset.
    * @return Reference to the i-th element.
    */
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

/**
 * @brief Copy `size` elements from host to device for the given handle.
 *
 * Performs a blocking copy using `cudaMemcpy(..., cudaMemcpyHostToDevice)` or
 * `hipMemcpy(..., hipMemcpyHostToDevice)` when the respective backend is enabled.
 * If neither CUDA nor HIP is enabled, this function is a no-op.
 *
 * @tparam T Element type.
 * @param size Number of elements to copy.
 * @param x    Handle with valid `host_pointer` and `device_pointer`.
 *
 * @pre `x.device_pointer` is non-null when a GPU backend is enabled.
 * @note Errors are checked via `GENDIL_GPU_CHECK` if that macro is defined.
 * 
 *  * @par Example
 * @code
 * HostDevicePointer<double> x;
 * AllocateHostPointer(N, x);               // host alloc
 * #if defined(GENDIL_USE_DEVICE)
 *   AllocateDevicePointer(N, x);           // device alloc
 *   // initialize on host ...
 *   ToDevice(N, x);                        // sync → device
 *   // launch kernels using x (device pointer auto-selected)
 *   ToHost(N, x);                          // sync → host
 *   FreeDevicePointer(x);
 * #endif
 * FreeHostPointer(x);
 * @endcode
 * 
 */
template < typename T >
void ToDevice( GlobalIndex size, const HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaMemcpy((void*)x.device_pointer, x.host_pointer, size * sizeof( T ), cudaMemcpyHostToDevice) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipMemcpy((void*)x.device_pointer, x.host_pointer, size * sizeof( T ), hipMemcpyHostToDevice) );
   #endif
}

/**
 * @brief Copy `size` elements from device to host for the given handle.
 *
 * Performs a blocking copy using `cudaMemcpy(..., cudaMemcpyDeviceToHost)` or
 * `hipMemcpy(..., hipMemcpyDeviceToHost)` when the respective backend is enabled.
 * If neither CUDA nor HIP is enabled, this function is a no-op.
 *
 * @tparam T Element type.
 * @param size Number of elements to copy.
 * @param x    Handle with valid `host_pointer` and `device_pointer`.
 *
 * @pre `x.device_pointer` is non-null when a GPU backend is enabled.
 * @note Errors are checked via `GENDIL_GPU_CHECK` if that macro is defined.
 */
template < typename T >
void ToHost( GlobalIndex size, const HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaMemcpy(x.host_pointer, x.device_pointer, size * sizeof( T ), cudaMemcpyDeviceToHost) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipMemcpy(x.host_pointer, x.device_pointer, size * sizeof( T ), hipMemcpyDeviceToHost) );
   #endif
}

/**
 * @brief Free the host allocation associated with the handle.
 *
 * Calls `delete[]` on `x.host_pointer`. The pointer is not set to null.
 *
 * @tparam T Element type.
 * @param x Handle to free from.
 *
 * @warning Do not call twice for the same allocation; consider nulling
 *          `x.host_pointer` after freeing if you keep the handle alive.
 */
template< typename T >
void FreeHostPointer( HostDevicePointer< T > & x )
{
   delete[] x.host_pointer;
}

/**
 * @brief Free the device allocation associated with the handle.
 *
 * Uses `cudaFree` or `hipFree` when `GENDIL_USE_DEVICE` is enabled; otherwise
 * this function is a no-op.
 *
 * @tparam T Element type.
 * @param x Handle to free from.
 *
 * @note Errors are checked via `GENDIL_GPU_CHECK` if that macro is defined.
 */
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

/**
 * @brief Allocate `size` elements on the host and store the pointer into @p x.
 *
 * @tparam T Element type.
 * @param size Number of elements.
 * @param x    Handle receiving the host pointer.
 *
 * @post `x.host_pointer` points to a `new[]`-allocated array of length `size`.
 * @note Memory is value-initialized only for class/struct types; for fundamental
 *       types it is uninitialized.
 */
template < typename T >
void AllocateHostPointer( size_t size, HostDevicePointer< T > & x )
{
   x.host_pointer = new T[size];
}

/**
 * @brief Allocate `size` elements on the device and store the pointer into @p x.
 *
 * Uses `cudaMalloc` or `hipMalloc` depending on the enabled backend. If neither
 * CUDA nor HIP is enabled, this function is a no-op.
 *
 * @tparam T Element type.
 * @param size Number of elements.
 * @param x    Handle receiving the device pointer.
 *
 * @post On success, `x.device_pointer` points to a device allocation holding
 *       `size * sizeof(T)` bytes; contents are uninitialized.
 * @warning Current implementation does not check the return code; consider
 *          wrapping with `GENDIL_GPU_CHECK` for consistency with copies. @todo Validate.
 */
template < typename T >
void AllocateDevicePointer( size_t size, HostDevicePointer< T > & x )
{
   #if defined(GENDIL_USE_CUDA)
      GENDIL_GPU_CHECK( cudaMalloc(&x.device_pointer, size * sizeof( T ) ) );
   #elif defined(GENDIL_USE_HIP)
      GENDIL_GPU_CHECK( hipMalloc(&x.device_pointer, size * sizeof( T ) ) );
   #endif
}

} // namespace gendil
