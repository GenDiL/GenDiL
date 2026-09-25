// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <cstdlib>
#include <utility>
#include <array>
#include <iostream>
#include <tuple>
#include <concepts>
#include <functional>
#ifdef GENDIL_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#ifdef GENDIL_USE_HIP
#include <hip/hip_runtime.h>
#endif

// Check if C++23 stdfloat header is available
// Note: GCC 13 with -std=c++23 reports __cplusplus as 202100L, not 202302L
// IMPORTANT: Must include BEFORE opening gendil namespace to avoid creating gendil::std
#if __cplusplus >= 202100L && __has_include(<stdfloat>)
  #define GENDIL_HAS_STDFLOAT 1
  #include <stdfloat>
#else
  #define GENDIL_HAS_STDFLOAT 0
#endif

namespace gendil {

// Type used for local indices
using LocalIndex = int;

// Type used for global indices
#if GENDIL_USE_DEVICE
using GlobalIndex = size_t;
#else
using GlobalIndex = size_t;
#endif

// Type used for positive integers
#if GENDIL_USE_DEVICE
using Integer = size_t;
#else
using Integer = size_t;
#endif

// Type used for real numbers - configurable via CMake
#ifndef GENDIL_REAL_TYPE
  // Default to double for backward compatibility
  #define GENDIL_REAL_TYPE double
#endif

// Helper macros for string/token operations
#define GENDIL_XSTR(x) GENDIL_STR(x)
#define GENDIL_STR(x) #x
#define GENDIL_REAL_TYPE_STR GENDIL_XSTR(GENDIL_REAL_TYPE)

// Select Real type based on CMake configuration
// CMake defines GENDIL_REAL_TYPE_<TYPE> flags for preprocessor selection
#if defined(GENDIL_REAL_TYPE_DOUBLE) || !defined(GENDIL_REAL_TYPE_FLOAT) && !defined(GENDIL_REAL_TYPE_FLOAT16) && !defined(GENDIL_REAL_TYPE_FLOAT32) && !defined(GENDIL_REAL_TYPE_FLOAT64) && !defined(GENDIL_REAL_TYPE_FLOAT128) && !defined(GENDIL_REAL_TYPE_BFLOAT16)
  using Real = double;

#elif defined(GENDIL_REAL_TYPE_FLOAT)
  using Real = float;

#elif defined(GENDIL_REAL_TYPE_FLOAT16)
  #if GENDIL_HAS_STDFLOAT
    using Real = ::std::float16_t;
  #else
    #error "GENDIL_REAL_TYPE=float16_t requested but std::float16_t is not available. Requires C++23 and compiler support."
  #endif

#elif defined(GENDIL_REAL_TYPE_FLOAT32)
  #if GENDIL_HAS_STDFLOAT
    using Real = ::std::float32_t;
  #else
    #error "GENDIL_REAL_TYPE=float32_t requested but std::float32_t is not available. Requires C++23 and compiler support."
  #endif

#elif defined(GENDIL_REAL_TYPE_FLOAT64)
  #if GENDIL_HAS_STDFLOAT
    using Real = ::std::float64_t;
  #else
    #error "GENDIL_REAL_TYPE=float64_t requested but std::float64_t is not available. Requires C++23 and compiler support."
  #endif

#elif defined(GENDIL_REAL_TYPE_FLOAT128)
  #if GENDIL_HAS_STDFLOAT
    using Real = ::std::float128_t;
  #else
    #error "GENDIL_REAL_TYPE=float128_t requested but std::float128_t is not available. Requires C++23 and compiler support. Note: float128_t has very limited compiler support."
  #endif

#elif defined(GENDIL_REAL_TYPE_BFLOAT16)
  #if GENDIL_HAS_STDFLOAT
    using Real = ::std::bfloat16_t;
  #else
    #error "GENDIL_REAL_TYPE=bfloat16_t requested but std::bfloat16_t is not available. Requires C++23 and compiler support."
  #endif

#else
  #error "Invalid GENDIL_REAL_TYPE configuration. Must be one of: double, float, float16_t, float32_t, float64_t, float128_t, bfloat16_t"
#endif

// Portable stream insertion operators for C++23 fixed-width floating-point types
// Note: The C++23 standard does not require operator<< for these types.
// Some compilers (like GCC 13+) provide them, others don't.
// We provide helper functions to ensure portability.

// Helper functions for standard types (identity functions for portability)
inline float to_printable(float val) { return val; }
inline double to_printable(double val) { return val; }
inline long double to_printable(long double val) { return val; }

#if GENDIL_HAS_STDFLOAT

// Helper functions for printing - convert C++23 float types to standard types for guaranteed portability
// Note: GCC 13+ already provides operator<< for these types, but to_printable() is available
// for explicit conversion when needed (e.g., formatting, avoiding implicit conversions)
inline float to_printable(::std::float16_t val) { return static_cast<float>(val); }
inline float to_printable(::std::float32_t val) { return static_cast<float>(val); }
inline double to_printable(::std::float64_t val) { return static_cast<double>(val); }
inline long double to_printable(::std::float128_t val) { return static_cast<long double>(val); }
inline float to_printable(::std::bfloat16_t val) { return static_cast<float>(val); }

#endif // GENDIL_HAS_STDFLOAT

struct Empty { };

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define GENDIL_DEVICE_CODE
#endif

#ifndef _GENDIL_FUNC_NAME
#ifndef _MSC_VER
#define _GENDIL_FUNC_NAME __PRETTY_FUNCTION__
#else
// for Visual Studio C++
#define _GENDIL_FUNC_NAME __FUNCSIG__
#endif
#endif

// CUDA
#if defined( GENDIL_USE_CUDA )

using Stream_t = cudaStream_t;

#define GENDIL_HOST_DEVICE __host__ __device__

void cuda_error(
   cudaError_t error,
   const char *message,
   const char *func,
   const char *file,
   int line )
{
   std::cout << "\n\nCUDA error: (" << cudaGetErrorString(error) << ") failed with message:\n"
             << message
             << "\n ... in file: " << file << ':' << line
             << "\n ... in function: " << func << '\n';
   std::abort();
}

#define GENDIL_DEVICE_SYNC GENDIL_GPU_CHECK(cudaDeviceSynchronize())
#define GENDIL_STREAM_SYNC GENDIL_GPU_CHECK(cudaStreamSynchronize(0))
#define GENDIL_GPU_CHECK(x) \
   do \
   { \
      cudaError_t err = (x); \
      if (err != cudaSuccess) \
      { \
         cuda_error(err, #x, _GENDIL_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)

// HIP
#elif defined( GENDIL_USE_HIP )

using Stream_t = hipStream_t;

void hip_error(
   hipError_t error,
   const char *message,
   const char *func,
   const char *file,
   int line )
{
   std::cout << "\n\nHIP error: (" << hipGetErrorString(error) << ") failed with message:\n"
             << message
             << "\n ... in file: " << file << ':' << line
             << "\n ... in function: " << func << '\n';
   std::abort();
}

#define GENDIL_HOST_DEVICE __host__ __device__
#define GENDIL_DEVICE_SYNC GENDIL_GPU_CHECK(hipDeviceSynchronize())
#define GENDIL_STREAM_SYNC GENDIL_GPU_CHECK(hipStreamSynchronize(0))
#define GENDIL_GPU_CHECK(x) \
   do \
   { \
      hipError_t err = (x); \
      if (err != hipSuccess) \
      { \
         hip_error(err, #x, _GENDIL_FUNC_NAME, __FILE__, __LINE__); \
      } \
   } \
   while (0)

#else

#define GENDIL_HOST_DEVICE
#define GENDIL_DEVICE_SYNC 
#define GENDIL_STREAM_SYNC
#define GENDIL_GPU_CHECK(x) x

#endif

#if GENDIL_USE_DEVICE
#define GENDIL_DEVICE __device__
#else
#define GENDIL_DEVICE
#endif

#ifdef GENDIL_DEVICE_CODE
#define GENDIL_SHARED __shared__
#else
#define GENDIL_SHARED
#endif

#ifdef GENDIL_DEVICE_CODE
#define GENDIL_SYNC_THREADS() __syncthreads()
#else
#define GENDIL_SYNC_THREADS()
#endif

#if defined( GENDIL_USE_CUDA )
#define GENDIL_INLINE __forceinline__
#elif defined( GENDIL_USE_HIP )
#define GENDIL_INLINE inline  __attribute__((always_inline))
#elif defined (_WIN32)
#define GENDIL_INLINE inline
#else
#define GENDIL_INLINE inline  __attribute__((always_inline))
#endif

}
