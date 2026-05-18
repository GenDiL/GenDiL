// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once
#include <random>

/**
 * @file vector.hpp
 * @brief Host/device-aware dynamically-sized vector with explicit sync control.
 *
 * `gendil::Vector` manages a pair of host/device pointers (via
 * `HostDevicePointer<Real>`) and exposes explicit read/write accessors that
 * handle on-demand transfers. This keeps GPU/CPU data movement under caller
 * control while offering a simple, STL-like interface.
 *
 * Key points:
 *  - Dual residency (host + device) with lazy copies on first read.
 *  - Explicit read/write APIs that set validity flags and avoid accidental traffic.
 *  - Basic BLAS-like helpers (axpy, scaling, +=, -=, *=, /=) with OpenMP loops.
 *  - Optional MFEM interop: wrap as `mfem::Vector` without copying.
 *
 * @note Device paths are compiled only when `GENDIL_USE_DEVICE` is defined;
 *       otherwise device accessors fall back to host.
 */


#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Utilities/Loop/deviceloop.hpp"
#include "gendil/Utilities/debug.hpp"
#include "gendil/Interfaces/MFEM/vector.hpp"

namespace gendil
{

namespace details
{

template <typename InitFunc>
struct VectorInitFunctor
{
   Real* data;
   InitFunc init_func;

   GENDIL_HOST_DEVICE
   void operator()(Integer i) const
   {
      data[i] = init_func(i);
   }
};

struct VectorCopyFunctor
{
   Real* dst;
   const Real* src;

   GENDIL_HOST_DEVICE
   void operator()(Integer i) const
   {
      dst[i] = src[i];
   }
};

} // namespace details

/**
 * @class Vector
 * @brief Dual-resident vector (host/device) with manual synchronization.
 *
 * The vector owns memory on host and (optionally) device. Two validity flags
 * (`host_valid`, `device_valid`) track where the most recent copy lives.
 * Accessor methods (`Read*`, `Write*`, `ReadWrite*`) ensure availability and
 * set flags so callers can avoid unintended transfers.
 *
 * @invariant If `host_valid && device_valid`, both copies hold the same data.
 * @invariant If exactly one of the flags is true, that side holds the truth.
 * @invariant After `Write*` accessors, only the written side is valid.
 *
 * @warning The `operator[]` methods touch **host** memory (and may copy from
 *          device); they invalidate the device copy on non-const access.
 *          Use the explicit pointer accessors for GPU code.
 */
class Vector
{
public:
   /**
    * @brief Construct an empty vector (`Size()==0`).
    * @post `host_valid==false`, `device_valid==false`, no allocations.
    */
   Vector() : n(0) {}

   /**
    * @brief Construct a vector of length @p N with uninitialized entries.
    * @param N Element count.
    * @post Host and device buffers are allocated; both validity flags are false
    *       until first write (`WriteHostData`/`WriteDeviceData`).
    */
   explicit Vector(size_t N)
   : n(N)
   {
      AllocateHostPointer< Real >( n, ptr );
      AllocateDevicePointer< Real >( n, ptr );
   }

   /**
    * @brief Convenience overload delegating to the `size_t` constructor.
    */
   explicit Vector(int n) : Vector(static_cast<size_t>(n)) {}

   /**
    * @brief Construct sized to a finite-element space.
    * @tparam FiniteElementSpace  Type exposing `GetNumberOfFiniteElementDofs()`.
    * @param finite_element_space Space providing the number of DoFs.
    */
   template < typename FiniteElementSpace >
   explicit Vector( const FiniteElementSpace & finite_element_space )
   : Vector( finite_element_space.GetNumberOfFiniteElementDofs() )
   { }

   /**
    * @brief Construct and initialize from index-to-value function.
    * @tparam Lambda Callable: Integer -> Real
    * @param N Element count.
    * @param init_func Function mapping index i to initial value.
    * @post Host buffer is valid and initialized; device is invalid.
    *
    * Example: Vector v(10, [](Integer i) { return static_cast<Real>(i); });
    */
   template < typename Lambda >
   Vector(Integer N, Lambda&& init_func)
   : Vector(N)
   {
      Real* data = WriteDeviceData();

      using InitFunc = std::decay_t<Lambda>;

      DeviceLoop(
         N,
         details::VectorInitFunctor<InitFunc>{
            data,
            std::forward<Lambda>(init_func)
         }
      );
   }

   /**
    * @brief Free host and device storage.
    * @note Safe to destroy even if one side was never allocated or valid.
    */
   ~Vector()
   {
      FreeHostPointer(ptr);
      FreeDevicePointer(ptr);
   }

   /**
    * @brief Return the number of elements.
    */
   GENDIL_HOST_DEVICE
   size_t Size() const
   {
      return n;
   }

   /**
    * @brief Element access on host (mutable).
    * @param index Global index (0 ≤ index < Size()).
    * @return Reference to host element.
    *
    * Ensures a host copy exists (`ReadWriteHostData()`); marks device invalid.
    * @warning Triggers host copy if the latest data is on device.
    * @warning On GPU it is the responsibility of the caller to ensure that the device copy is valid before calling this method.
    */
   GENDIL_HOST_DEVICE
   Real & operator[]( const GlobalIndex & index )
   {
      #ifndef GENDIL_DEVICE_CODE
      ReadWriteHostData();
      #endif
      return ptr[ index ];
   }

   /**
    * @brief Element access on host (const).
    * @param index Global index (0 ≤ index < Size()).
    * @return Const reference to host element.
    *
    * Ensures a host copy exists (`ReadHostData()`); will make the host copy valid.
    * 
    * @warning On GPU it is the responsibility of the caller to ensure that the device copy is valid before calling this method.
    * 
    */
   GENDIL_HOST_DEVICE
   const Real & operator[]( const GlobalIndex & index ) const
   {
      #ifndef GENDIL_DEVICE_CODE
      ReadHostData();
      #endif
      return ptr[ index ];
   }

   /**
    * @brief Deep copy constructor (host path).
    *
    * Allocates host/device buffers and copies elements **on host**.
    * @note Device-to-device copy is a TODO; currently this copies via host.
    */
   Vector(const Vector &in)
   : n(in.n)
   {
      AllocateHostPointer<Real>(n, ptr);
      AllocateDevicePointer<Real>(n, ptr);

      const Real * in_ptr = in.ReadDeviceData();
      Real * my_ptr = this->WriteDeviceData();

      DeviceLoop(n, details::VectorCopyFunctor{my_ptr, in_ptr});
   }

   /**
    * @brief Deep copy assignment (sizes must match).
    * @param x Source vector.
    * @pre `Size() == x.Size()`.
    * @note Copies on host path; device copy is marked invalid until next sync.
    */
   Vector& operator=( const Vector & x )
   {
      if (this == &x)
      {
         return *this;
      }

      GENDIL_VERIFY(n == x.n, "Vector sizes must match for assignment.");

      const Real* x_ptr = x.ReadDeviceData();
      Real* data = WriteDeviceData();

      DeviceLoop(n, [=] GENDIL_HOST_DEVICE (Integer i)
      {
         data[i] = x_ptr[i];
      });

      return *this;
   }

   /**
    * @brief Fill the vector with a scalar value (host path).
    * @param val Value assigned to each entry.
    * @post Host valid, device invalid.
    */
   Vector& operator=( Real val )
   {
      Real* data = WriteDeviceData();

      DeviceLoop(n, [=] GENDIL_HOST_DEVICE (Integer i)
      {
         data[i] = val;
      });

      return *this;
   }

   /**
    * @brief Move constructor. Steals buffers and validity flags.
    * @post `other` becomes empty with both validity flags false.
    */
   Vector(Vector &&other) noexcept
   : ptr(std::move(other.ptr)), n(other.n),
     host_valid(other.host_valid), device_valid(other.device_valid)
   {
      other.n = 0;
      other.host_valid = false;
      other.device_valid = false;
   }

   /**
    * @brief Move assignment. Steals buffers and validity flags.
    * @post `other` becomes empty with both validity flags false.
    */
   Vector &operator=(Vector &&other) noexcept
   {
      if (this != &other)
      {
         ptr = std::move(other.ptr);
         n = other.n;
         host_valid = other.host_valid;
         device_valid = other.device_valid;

         other.n = 0;
         other.host_valid = false;
         other.device_valid = false;
      }
      return *this;
   }

   /**
    * @brief Whether the host copy currently holds valid data.
    */
   bool IsHostValid() const
   {
      return host_valid;
   }

   /**
    * @brief Whether the device copy currently holds valid data.
    */
   bool IsDeviceValid() const
   {
      return device_valid;
   }

   /**
    * @brief Ensure host data is available; return read-only host pointer.
    * @return `const Real*` host pointer.
    *
    * If only device is valid, copies device→host and sets `host_valid=true`
    * (device remains valid). Requires at least one side to be valid.
    */
   const Real* ReadHostData() const
   {
      GENDIL_VERIFY(host_valid || device_valid, "Vector data is not valid on either host or device.");
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
         host_valid = true;
      }
      return ptr.host_pointer;
   }

   /**
    * @brief Ensure host data is available for in-place modification.
    * @return `Real*` host pointer.
    *
    * If only device is valid, copies device→host. Marks device invalid.
    */
   Real* ReadWriteHostData()
   {
      GENDIL_VERIFY(host_valid || device_valid, "Vector data is not valid on either host or device.");
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
         host_valid = true;
      }
      device_valid = false;
      return ptr.host_pointer;
   }

   /**
    * @brief Obtain a host pointer for write-only access.
    * @return `Real*` host pointer.
    *
    * Marks host valid and device invalid (no copy performed).
    */
   Real* WriteHostData()
   {
      host_valid = true;
      device_valid = false;
      return ptr.host_pointer;
   }

   /**
    * @brief Ensure device data is available; return read-only device pointer.
    * @return `const Real*` device pointer (or host fallback if no device).
    *
    * If only host is valid, copies host→device and sets `device_valid=true`
    * (host remains valid).
    * @note When `GENDIL_USE_DEVICE` is not defined, this delegates to `ReadHostData()`.
    */
   const Real* ReadDeviceData() const
   {
#ifdef GENDIL_USE_DEVICE
      GENDIL_VERIFY(host_valid || device_valid, "Vector data is not valid on either host or device.");
      if (host_valid && !device_valid)
      {
         ToDevice( n, ptr );
         device_valid = true;
      }
      return ptr.device_pointer;
#else
      return ReadHostData();
#endif
   }

   /**
    * @brief Ensure device data is available for in-place modification.
    * @return `Real*` device pointer (or host fallback if no device).
    *
    * If only host is valid, copies host→device. Marks host invalid.
    * @note When `GENDIL_USE_DEVICE` is not defined, this delegates to `ReadWriteHostData()`.
    */
   Real* ReadWriteDeviceData()
   {
#ifdef GENDIL_USE_DEVICE
      GENDIL_VERIFY(host_valid || device_valid, "Vector data is not valid on either host or device.");
      if (host_valid && !device_valid)
      {
         ToDevice( n, ptr );
         device_valid = true;
      }
      host_valid = false;
      return ptr.device_pointer;
#else
      return ReadWriteHostData();
#endif
   }

   /**
    * @brief Obtain a device pointer for write-only access.
    * @return `Real*` device pointer (or host fallback if no device).
    *
    * Marks device valid and host invalid (no copy performed).
    * @note When `GENDIL_USE_DEVICE` is not defined, this delegates to `WriteHostData()`.
    */
   Real* WriteDeviceData()
   {
#ifdef GENDIL_USE_DEVICE
      host_valid = false;
      device_valid = true;
      return ptr.device_pointer;
#else
      return WriteHostData();
#endif
   }

   /**
    * @brief Make both host and device copies valid (two-way sync).
    *
    * If only one side is valid, this performs the missing copy so both sides
    * become valid. No action if both are already valid.
    *
    * @pre At least one side must be valid.
    */
   void Sync()
   {
      GENDIL_VERIFY(host_valid || device_valid, "Vector data is not valid on either host or device.");


   #ifdef GENDIL_USE_DEVICE
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
         host_valid = true;
      }
      if (host_valid && !device_valid)
      {
         ToDevice( n, ptr );
         device_valid = true;
      }
   #else
      if (!host_valid)
      {
         GENDIL_VERIFY(false, "CPU-only Vector has no valid host data.");
      }
      device_valid = false;
   #endif
   }

#ifdef GENDIL_USE_MFEM
   /**
    * @brief Wrap this vector as an `mfem::Vector` without copying.
    *
    * Returns a lightweight adapter that shares the underlying pointers with MFEM
    * and uses a callback to restore GenDiL validity flags on adapter destruction.
    *
    * @warning Loses cv-qualification to satisfy MFEM's API.
    * @note When `GENDIL_USE_DEVICE` is enabled, device pointer and flags are
    *       forwarded as well; otherwise a host-only adapter is returned.
    */
   GenDiLMFEMVector ToMFEMVector()
   {
      auto restore = [this](bool h, bool d){
        this->host_valid   = h;
        this->device_valid = d;
      };

#ifdef GENDIL_USE_DEVICE
      bool mfem_host_valid = host_valid;
      bool mfem_device_valid = device_valid;
      host_valid = false;
      device_valid = false;
      return GenDiLMFEMVector(
         ptr.host_pointer,
         ptr.device_pointer,
         int(n),
         mfem_host_valid,
         mfem_device_valid,
         restore
      );
#else
      return GenDiLMFEMVector(
         ptr.host_pointer,
         int(n),
         restore
      );
#endif
   }

#endif

private:
   // raw host/device storage
   HostDevicePointer<Real> ptr;

   // element count
   size_t n;

   // sync flags
   mutable bool host_valid{false}, device_valid{false};
};

std::ostream& operator<<(std::ostream& os, const Vector& vec)
{
   if (vec.Size() == 0)
   {
      return os;
   }

   for (size_t i = 0; i < vec.Size()-1; i++)
   {
      os << vec[i] << ", ";
   }

   if (vec.Size() > 0) {
      os << vec[vec.Size()-1];
   }

   return os;
}

/**
 * @brief In-place vector addition: @f$ x \leftarrow x + y @f$.
 * @pre `x.Size() == y.Size()`.
 * @note Operates on host; marks device invalid for @p x.
 */
Vector & operator+=( Vector & x, Vector const & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");

   Real* u( x.ReadWriteDeviceData() );
   const Real* v( y.ReadDeviceData() );
   DeviceLoop(x.Size(), [=] GENDIL_HOST_DEVICE (Integer i)
   {
      u[i] += v[i];
   });

   return x;
}

/**
 * @brief In-place vector subtraction: @f$ x \leftarrow x - y @f$.
 * @pre `x.Size() == y.Size()`.
 * @note Operates on host; marks device invalid for @p x.
 */
Vector & operator-=( Vector & x, Vector const & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");

   Real* u( x.ReadWriteDeviceData() );
   const Real* v( y.ReadDeviceData() );
   DeviceLoop(x.Size(), [=] GENDIL_HOST_DEVICE (Integer i)
   {
      u[i] -= v[i];
   });

   return x;
}

/**
 * @brief In-place scaling: @f$ x \leftarrow a\,x @f$.
 * @param x Vector to scale.
 * @param a Scalar multiplier.
 * @note Operates on host; marks device invalid for @p x.
 */
Vector& operator*=(
   Vector & x,
   const Real & a )
{
   Real* u( x.ReadWriteDeviceData() );
   DeviceLoop(x.Size(), [=] GENDIL_HOST_DEVICE (Integer i)
   {
      u[i] *= a;
   });

   return x;
}

/**
 * @brief In-place division: @f$ x \leftarrow x / a @f$.
 * @param x Vector to scale.
 * @param a Scalar divisor.
 * @note Operates on host; marks device invalid for @p x.
 */
Vector& operator/=(
   Vector & x,
   const Real & a )
{
   GENDIL_VERIFY(a != 0, "Division by zero");

   Real* u( x.ReadWriteDeviceData() );
   DeviceLoop(x.Size(), [=] GENDIL_HOST_DEVICE (Integer i)
   {
      u[i] /= a;
   });

   return x;
}

/**
 * @brief y ← a·x + y  (AXPY).
 * @param a Scalar multiplier.
 * @param x Input vector.
 * @param y Output/accumulator vector.
 * @pre `x.Size() == y.Size()`.
 * @note Operates on host; marks device invalid for @p y.
 */
void Axpy(
   const Real & a,
   const Vector & x,
   Vector & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");

   const Real* u( x.ReadDeviceData() );
   Real* v( y.ReadWriteDeviceData() );
   DeviceLoop(x.Size(), [=] GENDIL_HOST_DEVICE (Integer i)
   {
      v[ i ] = v[ i ] + a * u[ i ] ;
   });
}

void FillRandom(
   Vector& v,
   unsigned int seed = 12345,
   Real min_value = Real(-1),
   Real max_value = Real(1))
{
   v.WriteHostData(); // Ensure host buffer is allocated and valid

   std::mt19937 rng(seed);
   std::uniform_real_distribution<Real> dist(min_value, max_value);

   for (size_t i = 0; i < v.Size(); ++i)
   {
      v[i] = dist(rng);
   }
}

}
