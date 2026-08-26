// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vectoraccess.hpp"
#include "gendil/Interfaces/Hypre/hypreerror.hpp"
#include "gendil/Utilities/dependentfalse.hpp"

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

template < HostAccessibleVector VectorType >
inline HypreParVectorView MakeHostHypreParVectorView(
   const VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( ReadHostVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      const auto * data = ReadHostVector( vector );
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
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
}

template < HostAccessibleVector VectorType >
inline HypreParVectorView MakeHostHypreParVectorView(
   VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( ReadWriteHostVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      auto * data = ReadWriteHostVector( vector );
      return HypreParVectorView(
         data,
         expected_size,
         comm,
         HYPRE_MEMORY_HOST );
   }
   else
   {
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
}

template < HostAccessibleVector VectorType >
inline HypreParVectorView MakeHostHypreParVectorWriteView(
   VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( WriteHostVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      return HypreParVectorView(
         WriteHostVector( vector ),
         expected_size,
         comm,
         HYPRE_MEMORY_HOST );
   }
   else
   {
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
}

template < DeviceAccessibleVector VectorType >
inline HypreParVectorView MakeDeviceHypreParVectorView(
   const VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

#ifdef GENDIL_USE_HYPRE_DEVICE
   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( ReadDeviceVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      const auto * data = ReadDeviceVector( vector );
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
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
#else
   (void) vector;
   (void) comm;
   GENDIL_VERIFY(
      false,
      "DeviceMatVecBackend HypreParVectorView requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return {};
#endif
}

template < DeviceAccessibleVector VectorType >
inline HypreParVectorView MakeDeviceHypreParVectorView(
   VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

#ifdef GENDIL_USE_HYPRE_DEVICE
   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( ReadWriteDeviceVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      auto * data = ReadWriteDeviceVector( vector );
      return HypreParVectorView(
         data,
         expected_size,
         comm,
         HYPRE_MEMORY_DEVICE );
   }
   else
   {
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
#else
   (void) vector;
   (void) comm;
   GENDIL_VERIFY(
      false,
      "DeviceMatVecBackend HypreParVectorView requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return {};
#endif
}

template < DeviceAccessibleVector VectorType >
inline HypreParVectorView MakeDeviceHypreParVectorWriteView(
   VectorType & vector,
   const HYPRE_Int expected_size,
   MPI_Comm comm = hypre_MPI_COMM_SELF )
{
   GENDIL_VERIFY(
      GetVectorSize( vector ) == static_cast< size_t >( expected_size ),
      "MakeHypreParVectorView received a vector with the wrong size." );

#ifdef GENDIL_USE_HYPRE_DEVICE
   using ValueType = std::remove_cv_t<
      std::remove_pointer_t< decltype( WriteDeviceVector( vector ) ) > >;
   if constexpr ( std::is_same_v< ValueType, HYPRE_Complex > )
   {
      return HypreParVectorView(
         WriteDeviceVector( vector ),
         expected_size,
         comm,
         HYPRE_MEMORY_DEVICE );
   }
   else
   {
      static_assert(
         dependent_false_v< ValueType >,
         "HypreParVectorView requires vector storage element type to be "
         "exactly HYPRE_Complex." );
   }
#else
   (void) vector;
   (void) comm;
   GENDIL_VERIFY(
      false,
      "DeviceMatVecBackend HypreParVectorView requires GENDIL_USE_HYPRE_DEVICE. Configure GenDiL with CUDA/HIP and a matching device-enabled Hypre." );
   return {};
#endif
}

template < typename MatrixBackend, typename VectorType >
inline HypreParVectorView MakeHypreParVectorView(
   const MatrixBackend &,
   const VectorType & vector,
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

template < typename MatrixBackend, typename VectorType >
inline HypreParVectorView MakeHypreParVectorView(
   const MatrixBackend &,
   VectorType & vector,
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
