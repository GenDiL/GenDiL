// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#ifdef GENDIL_USE_MFEM
#include <mfem.hpp>

namespace gendil
{

/**
 * @brief A wrapper for mfem::Vector used when transforming a gendil::Vector into an mfem::Vector.
 * 
 * @note The restore function is called in the destructor to synchronize the flags with the original gendil::Vector.
 */
class GenDiLMFEMVector : public mfem::Vector
{
public:
   using RestoreFn = std::function<void(bool/*final_h*/,bool/*final_d*/)>;

   GenDiLMFEMVector(
      Real* h_ptr,
      int length,
      RestoreFn restore )
   : mfem::Vector(h_ptr, length), restore(restore)
   {
      bool own = false;
   
      SetSize( length );
      auto & M = GetMemory();
      M.Wrap(
         h_ptr,
         length,
         mfem::MemoryType::HOST,
         own);
   }

   GenDiLMFEMVector(
      Real* h_ptr,
      Real* d_ptr,
      int  length,
      bool host_valid,
      bool device_valid,
      RestoreFn restore )
   : mfem::Vector(), restore(restore)
   {
      bool own = false;
   
      SetSize( length );
      auto & M = GetMemory();
      M.Wrap(
         h_ptr,
         d_ptr,
         length,
         mfem::MemoryType::HOST,
         own,
         host_valid,
         device_valid);
      M.SetDeviceMemoryType(mfem::MemoryType::DEVICE);
   }

   ~GenDiLMFEMVector()
   {
      auto & M = GetMemory();
      bool final_h = M.HostIsValid();
      bool final_d = M.DeviceIsValid();
      restore(final_h, final_d);
   }

private:
   RestoreFn restore;
};

}
#endif
