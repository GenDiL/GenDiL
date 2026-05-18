// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "types.hpp"

#include <iostream>
#include <assert.h>
#include <cstdlib>

#if defined(GENDIL_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(GENDIL_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace gendil
{

GENDIL_HOST_DEVICE GENDIL_INLINE
void Assert(bool should_be_true, const char msg[] = "")
{
    // FIXME: this debug check doesn't seem to actually work.
    #ifndef NDEBUG
    if (not should_be_true)
    {
        printf("---------- gendil error ----------\n%s\n----------------------------------\n", msg);
        assert(should_be_true);
    }
    #endif
}

inline void Assert(bool condition, const char* condition_str,
                   const char* file, int line, const char* msg = "")
{
   if (!condition)
   {
      std::cerr << "Assertion failed: (" << condition_str << ")"
                << " in file " << file << ", line " << line;
      if (msg && *msg)
      {
         std::cerr << ": " << msg;
      }
      std::cerr << std::endl;
      std::abort();
   }
}

// CPU
#ifndef GENDIL_DEVICE_CODE
#ifdef NDEBUG
#define GENDIL_ASSERT(cond, ...)
#else
#define GENDIL_ASSERT(cond, ...) Assert((cond), #cond, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)
#endif

#define GENDIL_VERIFY(cond, ...) Assert((cond), #cond, __FILE__, __LINE__ __VA_OPT__(,) __VA_ARGS__)
// GPU
#else
#define GENDIL_ASSERT(cond, ...)
#define GENDIL_VERIFY(cond, ...)
#endif


#if defined(GENDIL_USE_CUDA)

using DeviceError_t = cudaError_t;
using DeviceStream_t = cudaStream_t;
using DeviceProperties = cudaDeviceProp;

inline const char * DeviceGetErrorString(DeviceError_t error)
{
   return cudaGetErrorString(error);
}

inline DeviceError_t DevicePeekAtLastError()
{
   return cudaPeekAtLastError();
}

inline DeviceError_t DeviceGetLastError()
{
   return cudaGetLastError();
}

inline DeviceError_t DeviceStreamSynchronize(DeviceStream_t stream)
{
   return cudaStreamSynchronize(stream);
}

inline DeviceError_t DeviceGetDevice(int * device)
{
   return cudaGetDevice(device);
}

inline DeviceError_t DeviceGetDeviceProperties(DeviceProperties * props, int device)
{
   return cudaGetDeviceProperties(props, device);
}

inline constexpr DeviceError_t DeviceSuccess = cudaSuccess;

#elif defined(GENDIL_USE_HIP)

using DeviceError_t = hipError_t;
using DeviceStream_t = hipStream_t;
using DeviceProperties = hipDeviceProp_t;

inline const char * DeviceGetErrorString(DeviceError_t error)
{
   return hipGetErrorString(error);
}

inline DeviceError_t DevicePeekAtLastError()
{
   return hipPeekAtLastError();
}

inline DeviceError_t DeviceGetLastError()
{
   return hipGetLastError();
}

inline DeviceError_t DeviceStreamSynchronize(DeviceStream_t stream)
{
   return hipStreamSynchronize(stream);
}

inline DeviceError_t DeviceGetDevice(int * device)
{
   return hipGetDevice(device);
}

inline DeviceError_t DeviceGetDeviceProperties(DeviceProperties * props, int device)
{
   return hipGetDeviceProperties(props, device);
}

inline constexpr DeviceError_t DeviceSuccess = hipSuccess;

#endif

#if defined(GENDIL_USE_DEVICE)

inline void DeviceCheck(DeviceError_t error,
                        const char * expression,
                        const char * file,
                        int line,
                        const char * context = nullptr)
{
   if (error != DeviceSuccess)
   {
      std::cerr << "GenDiL device error at "
                << file << ":" << line << "\n";

      if (context && *context)
      {
         std::cerr << "  context: " << context << "\n";
      }

      std::cerr << "  expression: " << expression << "\n"
                << "  error: " << DeviceGetErrorString(error)
                << std::endl;

      std::abort();
   }
}

#define GENDIL_DEVICE_CHECK(expr) \
   ::gendil::DeviceCheck((expr), #expr, __FILE__, __LINE__)

inline void CheckLastDeviceLaunch(const char * launch_name,
                                  const char * file,
                                  int line)
{
   DeviceCheck(DeviceGetLastError(),
               "DeviceGetLastError()",
               file,
               line,
               launch_name);
}

#define GENDIL_CHECK_LAST_DEVICE_LAUNCH(launch_name) \
   ::gendil::CheckLastDeviceLaunch((launch_name), __FILE__, __LINE__)

inline void CheckNoPendingDeviceError(const char * launch_name,
                                      const char * file,
                                      int line)
{
   DeviceCheck(DeviceGetLastError(),
               "DeviceGetLastError() before kernel launch",
               file,
               line,
               launch_name);
}

#define GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(launch_name) \
   ::gendil::CheckNoPendingDeviceError((launch_name), __FILE__, __LINE__)

#else

#define GENDIL_DEVICE_CHECK(expr) do { (void)sizeof(expr); } while (0)
#define GENDIL_CHECK_LAST_DEVICE_LAUNCH(launch_name) do {} while (0)
#define GENDIL_CHECK_NO_PENDING_DEVICE_ERROR(launch_name) do {} while (0)

#endif

#if defined(GENDIL_USE_DEVICE)
inline void CheckDeviceLaunchConfiguration(
   dim3 gridDim,
   dim3 blockDim,
   size_t dynamic_shared_mem)
{
   int device = 0;
   GENDIL_DEVICE_CHECK(DeviceGetDevice(&device));

   DeviceProperties props;
   GENDIL_DEVICE_CHECK(DeviceGetDeviceProperties(&props, device));

   const unsigned int threads_per_block =
      blockDim.x * blockDim.y * blockDim.z;

   if (gridDim.x == 0 || gridDim.y == 0 || gridDim.z == 0)
   {
      std::cerr << "Invalid GenDiL launch: zero grid dimension.\n"
                << "  gridDim = (" << gridDim.x << ", "
                                    << gridDim.y << ", "
                                    << gridDim.z << ")\n";
      std::abort();
   }

   if (blockDim.x == 0 || blockDim.y == 0 || blockDim.z == 0)
   {
      std::cerr << "Invalid GenDiL launch: zero block dimension.\n"
                << "  blockDim = (" << blockDim.x << ", "
                                     << blockDim.y << ", "
                                     << blockDim.z << ")\n";
      std::abort();
   }

   if (threads_per_block > props.maxThreadsPerBlock)
   {
      std::cerr << "Invalid GenDiL launch: too many threads per block.\n"
                << "  blockDim = (" << blockDim.x << ", "
                                     << blockDim.y << ", "
                                     << blockDim.z << ")\n"
                << "  threads/block = " << threads_per_block << "\n"
                << "  maxThreadsPerBlock = "
                << props.maxThreadsPerBlock << "\n";
      std::abort();
   }

   if (blockDim.x > props.maxThreadsDim[0] ||
       blockDim.y > props.maxThreadsDim[1] ||
       blockDim.z > props.maxThreadsDim[2])
   {
      std::cerr << "Invalid GenDiL launch: block dimension exceeds device limit.\n"
                << "  blockDim = (" << blockDim.x << ", "
                                     << blockDim.y << ", "
                                     << blockDim.z << ")\n"
                << "  maxThreadsDim = (" << props.maxThreadsDim[0] << ", "
                                          << props.maxThreadsDim[1] << ", "
                                          << props.maxThreadsDim[2] << ")\n";
      std::abort();
   }

   if (gridDim.x > props.maxGridSize[0] ||
       gridDim.y > props.maxGridSize[1] ||
       gridDim.z > props.maxGridSize[2])
   {
      std::cerr << "Invalid GenDiL launch: grid dimension exceeds device limit.\n"
                << "  gridDim = (" << gridDim.x << ", "
                                    << gridDim.y << ", "
                                    << gridDim.z << ")\n"
                << "  maxGridSize = (" << props.maxGridSize[0] << ", "
                                        << props.maxGridSize[1] << ", "
                                        << props.maxGridSize[2] << ")\n";
      std::abort();
   }

   if (dynamic_shared_mem > props.sharedMemPerBlock)
   {
      std::cerr << "Invalid GenDiL launch: too much dynamic shared memory.\n"
                << "  dynamic shared memory = "
                << dynamic_shared_mem << "\n"
                << "  sharedMemPerBlock = "
                << props.sharedMemPerBlock << "\n";
      std::abort();
   }
}
#endif

} // namespace gendil
