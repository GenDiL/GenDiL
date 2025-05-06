// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <cstddef>
#include <utility>
#include <array>
#include <iostream>
#ifdef GENDIL_USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#ifdef GENDIL_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace gendil {

// Type used for local indices
using LocalIndex = int;

// Type used for global indices
#if GENDIL_USE_DEVICE
using GlobalIndex = int;
#else
using GlobalIndex = size_t;
#endif

// Type used for positive integers
#if GENDIL_USE_DEVICE
using Integer = size_t;
#else
using Integer = size_t;
#endif

// Type used for real numbers
using Real = double;

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
