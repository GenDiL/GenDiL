// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include "gendil/Utilities/MemoryManagement/hostdevicepointer.hpp"
#include "gendil/Interfaces/MFEM/vector.hpp"

namespace gendil
{

class Vector
{
public:
   explicit Vector(size_t N)
   : n(N)
   {
      AllocateHostPointer< Real >( n, ptr );
      AllocateDevicePointer< Real >( n, ptr );
   }

   explicit Vector(int n) : Vector(static_cast<size_t>(n)) {}

   template < typename FiniteElementSpace >
   explicit Vector( const FiniteElementSpace & finite_element_space )
   : Vector( finite_element_space.GetNumberOfFiniteElementDofs() )
   { }

   ~Vector()
   {
      FreeHostPointer(ptr);
      FreeDevicePointer(ptr);
   }

   size_t Size() const
   {
      return n;
   }

   Real & operator[]( const GlobalIndex & index )
   {
      ReadWriteHostData();
      return ptr[ index ];
   }

   const Real & operator[]( const GlobalIndex & index ) const
   {
      ReadHostData();
      return ptr[ index ];
   }

   Vector(const Vector &in)
   : n(in.n)
   {
      AllocateHostPointer<Real>(n, ptr);
      AllocateDevicePointer<Real>(n, ptr);

      // TODO: Make it device compatible
      const Real * in_ptr = in.ReadHostData();
      Real * my_ptr = this->WriteHostData();

      #pragma omp parallel for
      for (size_t i = 0; i < Size(); ++i)
      {
        my_ptr[i] = in_ptr[i];
      }
   }

   Vector& operator=( const Vector & x )
   {
      GENDIL_VERIFY(n == x.n, "Vector sizes must match for assignment.");

      // TODO: Make it device compatible
      Real* ptr = WriteHostData();
      const Real * x_ptr = x.ReadHostData();

      #pragma omp parallel for
      for ( size_t i=0; i<Size(); ++i )
      {
         ptr[i] = x_ptr[i];
      }
      return *this;
   }

   Vector& operator=( Real val )
   {
      // TODO: Make it device compatible
      Real* ptr = WriteHostData();

      #pragma omp parallel for
      for ( size_t i=0; i<Size(); ++i )
      {
         ptr[i] = val;
      }
      return *this;
   }

   Vector(Vector &&other) noexcept
   : ptr(std::move(other.ptr)), n(other.n),
     host_valid(other.host_valid), device_valid(other.device_valid)
   {
      other.n = 0;
      other.host_valid = false;
      other.device_valid = false;
   }

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

   bool IsHostValid() const
   {
      return host_valid;
   }

   bool IsDeviceValid() const
   {
      return device_valid;
   }

   const Real* ReadHostData() const
   {
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
         host_valid = true;
      }
      return ptr.host_pointer;
   }

   Real* ReadWriteHostData()
   {
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
         host_valid = true;
         device_valid = false;
      }
      return ptr.host_pointer;
   }

   Real* WriteHostData()
   {
      host_valid = true;
      device_valid = false;
      return ptr.host_pointer;
   }

   const Real* ReadDeviceData() const
   {
#ifdef GENDIL_USE_DEVICE
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

   Real* ReadWriteDeviceData()
   {
#ifdef GENDIL_USE_DEVICE
      if (host_valid && !device_valid)
      {
         ToDevice( n, ptr );
         host_valid = false;
         device_valid = true;
      }
      return ptr.device_pointer;
#else
      return ReadWriteHostData();
#endif
   }

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

   // Explicit sync calls
   void Sync()
   {
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
   }

#ifdef GENDIL_USE_MFEM
   /**
    * @brief Transform a Vector into an mfem::Vector.
    * 
    * @note Object will loose cv-qualifier.
    * 
    * @return GenDiLMFEMVector 
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

Vector & operator+=( Vector & x, Vector const & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");
   // TODO: Make it device compatible
   Real* u( x.ReadWriteHostData() );
   const Real* v( y.ReadHostData() );
   #pragma omp parallel for
   for (size_t i = 0; i < x.Size(); ++i) {
      u[i] += v[i];
   }
   return x;
}

Vector & operator-=( Vector & x, Vector const & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");
   // TODO: Make it device compatible
   Real* u( x.ReadWriteHostData() );
   const Real* v( y.ReadHostData() );
   #pragma omp parallel for
   for (size_t i = 0; i < x.Size(); ++i) {
      u[i] -= v[i];
   }
   return x;
}

Vector& operator*=(
   Vector & x,
   const Real & a )
{
   // TODO: Make it device compatible
   Real* u( x.ReadWriteHostData() );
   #pragma omp parallel for
   for (size_t i = 0; i < x.Size(); ++i) {
      u[i] *= a;
   }
   return x;
}

Vector& operator/=(
   Vector & x,
   const Real & a )
{
   // TODO: Make it device compatible
   Real* u( x.ReadWriteHostData() );
   #pragma omp parallel for
   for (size_t i = 0; i < x.Size(); ++i) {
      u[i] /= a;
   }
   return x;
}

// y = ax + y
void Axpy(
   const Real & a,
   const Vector & x,
   Vector & y )
{
   GENDIL_VERIFY(x.Size() == y.Size(), "Vector sizes must match for assignment.");
   // TODO: Make it device compatible
   const Real* u( x.ReadHostData() );
   Real* v( y.ReadWriteHostData() );
   #pragma omp parallel for
   for (size_t i = 0; i < x.Size(); ++i) {
      v[ i ] = v[ i ] + a * u[ i ] ;
   }
}

}
