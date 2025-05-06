// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/prelude.hpp"
#include <type_traits>
#include <vector>
#include <functional>
#include <mutex>

namespace gendil {

class GarbageCollector {
public:
   static GarbageCollector& Instance()
   {
      static GarbageCollector inst;
      return inst;
   }

   // register any pointer + custom deleter
   void RegisterPtr( void* ptr, std::function<void(void*)> deleter )
   {
      items.emplace_back( ptr, std::move( deleter ) );
   }

   template < typename T >
   void RegisterHostPtr( T* ptr )
   {
      RegisterPtr(
         static_cast< void* >( const_cast< std::remove_const_t<T>* >( ptr ) ),
         []( void* p )
         {
            delete[] static_cast<T*>( p );
         }
      );
   }

   template < typename T >
   void RegisterDevicePtr( T* dptr )
   {
      RegisterPtr(
         static_cast< void* >( const_cast< std::remove_const_t<T>* >( dptr ) ),
         []( void* p )
         {
            #if defined( GENDIL_USE_CUDA )
               GENDIL_GPU_CHECK( cudaFree( p ) );
            #elif defined( GENDIL_USE_HIP )
               GENDIL_GPU_CHECK( hipFree( p ) );
            #endif
         }
      );
   }

   template < typename T >
   void RegisterHostDevicePtr( const HostDevicePointer<T> & p )
   {
      RegisterHostPtr( p.host_pointer );
#if defined( GENDIL_USE_DEVICE )
      RegisterDevicePtr( p.device_pointer );
#endif
   }

   // empty the garbage collector
   void Cleanup()
   {
      for (auto it = items.rbegin(); it != items.rend(); ++it)
      {
         it->second( it->first );
      }
      items.clear();
   }

   static void RunCleanup()
   {
      Instance().Cleanup();
   }

private:
   GarbageCollector() = default;

   ~GarbageCollector()
   {
      Cleanup();
   }

   GarbageCollector( const GarbageCollector & ) = delete;

   GarbageCollector& operator=( const GarbageCollector & ) = delete;
   
   std::vector< std::pair< void*, std::function< void( void* ) > > > items;
};

}
