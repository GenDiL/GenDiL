// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vector.hpp"
#include "gendil/Interfaces/Hypre/hypreerror.hpp"

#include <type_traits>

namespace gendil
{

struct HypreParVectorView
{
   HYPRE_ParVector vector = nullptr;

   HypreParVectorView() = default;

   HypreParVectorView(
      HYPRE_Complex * data,
      const HYPRE_Int local_size,
      MPI_Comm comm = hypre_MPI_COMM_SELF,
      const HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST )
   {
      RequireHypreInitialized(
         "HypreParVectorView requires an active HypreSession or prior HYPRE_Initialize()." );

      HYPRE_BigInt partitioning[2] = {
         HYPRE_BigInt( 0 ),
         static_cast< HYPRE_BigInt >( local_size )
      };

      CheckHypreError(
         HYPRE_ParVectorCreate(
            comm,
            static_cast< HYPRE_BigInt >( local_size ),
            partitioning,
            &vector ),
         "HYPRE_ParVectorCreate failed" );

      auto * hypre_vector =
         reinterpret_cast< hypre_ParVector * >( vector );
      hypre_Vector * local_vector =
         hypre_ParVectorLocalVector( hypre_vector );

      CheckHypreError(
         hypre_SeqVectorSetData( local_vector, data ),
         "hypre_SeqVectorSetData failed" );

      CheckHypreError(
         hypre_ParVectorInitialize_v2( hypre_vector, memory_location ),
         "hypre_ParVectorInitialize_v2 failed" );
   }

   HypreParVectorView( const HypreParVectorView & ) = delete;
   HypreParVectorView & operator=( const HypreParVectorView & ) = delete;

   HypreParVectorView( HypreParVectorView && other ) noexcept
   : vector( other.vector )
   {
      other.vector = nullptr;
   }

   HypreParVectorView & operator=( HypreParVectorView && other ) noexcept
   {
      if ( this != &other )
      {
         Destroy();
         vector = other.vector;
         other.vector = nullptr;
      }
      return *this;
   }

   ~HypreParVectorView()
   {
      Destroy();
   }

   operator HYPRE_ParVector() const
   {
      return vector;
   }

private:
   void Destroy()
   {
      if ( vector != nullptr )
      {
         CheckHypreError(
            HYPRE_ParVectorDestroy( vector ),
            "HYPRE_ParVectorDestroy failed" );
         vector = nullptr;
      }
   }
};

inline HypreParVectorView MakeHostHypreParVectorView(
   const Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      vector.Size() == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

   if constexpr ( std::is_same_v< Real, HYPRE_Complex > )
   {
      const Real * data = vector.ReadHostData();
      // Hypre's ParVector C API is non-const even for routines that treat the
      // vector as input. This view is semantically read-only; passing it to a
      // mutating Hypre routine is unsupported.
      return HypreParVectorView(
         const_cast< HYPRE_Complex * >( data ),
         expected_size,
         comm,
         HYPRE_MEMORY_HOST );
   }
   else
   {
      GENDIL_VERIFY(
         false,
         "HypreParVectorView requires GenDiL Vector storage to be exactly HYPRE_Complex." );
      return {};
   }
}

inline HypreParVectorView MakeHostHypreParVectorView(
   Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      vector.Size() == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

   if constexpr ( std::is_same_v< Real, HYPRE_Complex > )
   {
      Real * data = vector.ReadWriteHostData();
      return HypreParVectorView(
         data,
         expected_size,
         comm,
         HYPRE_MEMORY_HOST );
   }
   else
   {
      GENDIL_VERIFY(
         false,
         "HypreParVectorView requires GenDiL Vector storage to be exactly HYPRE_Complex." );
      return {};
   }
}

inline HypreParVectorView MakeDeviceHypreParVectorView(
   const Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      vector.Size() == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

#ifdef GENDIL_USE_HYPRE_DEVICE
   if constexpr ( std::is_same_v< Real, HYPRE_Complex > )
   {
      const Real * data = vector.ReadDeviceData();
      // Hypre's ParVector C API is non-const even for routines that treat the
      // vector as input. This view is semantically read-only; passing it to a
      // mutating Hypre routine is unsupported.
      return HypreParVectorView(
         const_cast< HYPRE_Complex * >( data ),
         expected_size,
         comm,
         HYPRE_MEMORY_DEVICE );
   }
   else
   {
      GENDIL_VERIFY(
         false,
         "HypreParVectorView requires GenDiL Vector storage to be exactly HYPRE_Complex." );
      return {};
   }
#else
   (void) comm;
   GENDIL_VERIFY(
      false,
      "DeviceMatVecBackend HypreParVectorView requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return {};
#endif
}

inline HypreParVectorView MakeDeviceHypreParVectorView(
   Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      vector.Size() == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

#ifdef GENDIL_USE_HYPRE_DEVICE
   if constexpr ( std::is_same_v< Real, HYPRE_Complex > )
   {
      Real * data = vector.ReadWriteDeviceData();
      return HypreParVectorView(
         data,
         expected_size,
         comm,
         HYPRE_MEMORY_DEVICE );
   }
   else
   {
      GENDIL_VERIFY(
         false,
         "HypreParVectorView requires GenDiL Vector storage to be exactly HYPRE_Complex." );
      return {};
   }
#else
   (void) comm;
   GENDIL_VERIFY(
      false,
      "DeviceMatVecBackend HypreParVectorView requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return {};
#endif
}

template < typename MatrixBackend >
inline HypreParVectorView MakeHypreParVectorView(
   const MatrixBackend & backend,
   const Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   if constexpr ( is_host_matvec_backend_v< MatrixBackend > )
   {
      return MakeHostHypreParVectorView( vector, expected_size, comm );
   }
   else
   {
      return MakeDeviceHypreParVectorView( vector, expected_size, comm );
   }
}

template < typename MatrixBackend >
inline HypreParVectorView MakeHypreParVectorView(
   const MatrixBackend & backend,
   Vector & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   if constexpr ( is_host_matvec_backend_v< MatrixBackend > )
   {
      return MakeHostHypreParVectorView( vector, expected_size, comm );
   }
   else
   {
      return MakeDeviceHypreParVectorView( vector, expected_size, comm );
   }
}

} // namespace gendil
