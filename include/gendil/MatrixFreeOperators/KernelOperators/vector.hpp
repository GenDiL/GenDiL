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

   template < typename FiniteElementSpace >
   explicit Vector( const FiniteElementSpace & finite_element_space )
   : Vector( finite_element_space.GetNumberOfFiniteElementDofs() )
   { }

   size_t Size() const
   {
      return n;
   }

   void operator=( Real val )
   {
      host_valid = true;
      device_valid = false;

      #pragma omp parallel for
      for ( size_t i=0; i<n; ++i )
      {
         ptr.host_pointer[i] = val;
      }
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
      }
      host_valid   = true;
      return ptr.host_pointer;
   }

   Real* ReadWriteHostData()
   {
      if (!host_valid && device_valid)
      {
         ToHost( n, ptr );
      }
      host_valid   = true;
      device_valid = false;
      return ptr.host_pointer;
   }

   Real* WriteHostData()
   {
      host_valid   = true;
      device_valid = false;
      return ptr.host_pointer;
   }

   const Real* ReadDeviceData() const
   {
#ifdef GENDIL_USE_DEVICE
      if (host_valid && !device_valid)
      {
         ToDevice( n, ptr );
      }
      device_valid = true;
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
      }
      host_valid   = false;
      device_valid = true;
      return ptr.device_pointer;
#else
      return ReadWriteHostData();
#endif
   }

   Real* WriteDeviceData()
   {
#ifdef GENDIL_USE_DEVICE
      host_valid   = false;
      device_valid = true;
      return ptr.device_pointer;
#else
      return WriteHostData();
#endif
   }

   // Explicit sync calls if you prefer
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
         device_valid = false;
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
   

}
