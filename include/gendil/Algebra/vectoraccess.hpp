// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "gendil/Algebra/vector.hpp"
#include "gendil/Utilities/View/Containers/referencecontainer.hpp"
#include "gendil/Utilities/View/Layouts/vectorlayout.hpp"
#include "gendil/Utilities/View/Layouts/offsetlayout.hpp"

#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace gendil
{

/// Return the number of entries in a GenDiL vector.
GENDIL_HOST_DEVICE
inline size_t GetVectorSize( const Vector & vector )
{
   return vector.Size();
}

/// Return the logical extent of any complete rank-one generic view.
template<class Container, class Layout>
   requires (
      Layout::rank == 1 &&
      has_complete_extents_v<Layout>)
GENDIL_HOST_DEVICE
inline size_t GetVectorSize(
   const View<Container, Layout>& vector)
{
   return static_cast<size_t>(vector.Size());
}

/// Return read-only host data, synchronizing the GenDiL vector to the host.
inline const Real * ReadHostVector( const Vector & vector )
{
   return vector.ReadHostData();
}

/// Return mutable host data, preserving values and invalidating device data.
inline Real * ReadWriteHostVector( Vector & vector )
{
   return vector.ReadWriteHostData();
}

/// Return write-only host data without synchronization, invalidating device data.
inline Real * WriteHostVector( Vector & vector )
{
   return vector.WriteHostData();
}

/// Return read-only device data, synchronizing the GenDiL vector to the device.
inline const Real * ReadDeviceVector( const Vector & vector )
{
   return vector.ReadDeviceData();
}

/// Return mutable device data, preserving values and invalidating host data.
inline Real * ReadWriteDeviceVector( Vector & vector )
{
   return vector.ReadWriteDeviceData();
}

/// Return write-only device data without synchronization, invalidating host data.
inline Real * WriteDeviceVector( Vector & vector )
{
   return vector.WriteDeviceData();
}

#ifdef GENDIL_USE_MFEM

/// Return the number of entries in an MFEM vector as a size_t.
inline size_t GetVectorSize( const mfem::Vector & vector )
{
   GENDIL_VERIFY(
      vector.Size() >= 0,
      "An MFEM vector cannot have a negative size." );
   return static_cast< size_t >( vector.Size() );
}

/// Return read-only host data, synchronizing the MFEM vector to the host.
inline const mfem::real_t * ReadHostVector( const mfem::Vector & vector )
{
   return vector.HostRead();
}

/// Return mutable host data, preserving values and invalidating device data.
inline mfem::real_t * ReadWriteHostVector( mfem::Vector & vector )
{
   return vector.HostReadWrite();
}

/// Return write-only host data without synchronization, invalidating device data.
inline mfem::real_t * WriteHostVector( mfem::Vector & vector )
{
   return vector.HostWrite();
}

/// Return read-only device data, synchronizing through MFEM's memory manager.
inline const mfem::real_t * ReadDeviceVector( const mfem::Vector & vector )
{
   return vector.Read();
}

/// Return mutable device data, synchronizing values and invalidating host data.
inline mfem::real_t * ReadWriteDeviceVector( mfem::Vector & vector )
{
   return vector.ReadWrite();
}

/// Return write-only device data without synchronization, invalidating host data.
inline mfem::real_t * WriteDeviceVector( mfem::Vector & vector )
{
   return vector.Write();
}

#endif

namespace vector_access_detail
{

template<class ParentVector>
GENDIL_HOST_DEVICE GENDIL_INLINE
void ValidateVectorSubviewRange(
   const ParentVector& parent,
   const size_t offset,
   const size_t extent)
{
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
   const size_t parent_size = GetVectorSize(parent);
   GENDIL_ASSERT(
      offset <= parent_size && extent <= parent_size - offset,
      "Vector subview range exceeds its parent vector." );
#endif
}

} // namespace vector_access_detail

/**
 * @brief Create an owner-bound vector subview without selecting storage.
 *
 * The parent remains the sole synchronization and validity authority. Indexed
 * const access follows the parent's const policy (read-only for GenDiL Vector).
 * Pointer-backed kernel access obtained from this view is deeply const, so a
 * writable access captured by value must be used from a mutable closure. The
 * parent must outlive the returned view and must not move.
 * Subrange bounds are preconditions, checked only in CPU debug builds.
 */
template<class ParentVector>
   requires (
      !is_view_v<ParentVector> &&
      requires(const ParentVector& parent)
      {
         { GetVectorSize(parent) } -> std::same_as<size_t>;
      })
GENDIL_HOST_DEVICE
auto MakeVectorView(
   ParentVector& parent,
   const size_t offset,
   const size_t extent)
{
   vector_access_detail::ValidateVectorSubviewRange(
      parent, offset, extent);
   return MakeView(
      ReferenceContainer<ParentVector>{parent},
      MakeOffsetLayout(VectorLayout{extent}, offset));
}

/** Create a storage-bound subview by copying and advancing its shallow handle. */
template<class T, class IndexType>
GENDIL_HOST_DEVICE
auto MakeVectorView(
   const View<PointerContainer<T, IndexType>, VectorLayout>& parent,
   const size_t offset,
   const size_t extent)
{
   vector_access_detail::ValidateVectorSubviewRange(
      parent, offset, extent);
   return MakeVectorView(parent.data.data + offset, extent);
}

/** Create a nested owner-bound subview flattened to the original parent. */
template<class ParentVector, class IndexType>
GENDIL_HOST_DEVICE
auto MakeVectorView(
   const View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& parent,
   const size_t offset,
   const size_t extent)
{
   vector_access_detail::ValidateVectorSubviewRange(
      parent, offset, extent);
   GENDIL_ASSERT(
      offset <= std::numeric_limits<GlobalIndex>::max() - parent.layout.BaseOffset(),
      "Vector subview offset overflow." );
   const GlobalIndex combined_offset = parent.layout.BaseOffset() + offset;
   return MakeView(
      parent.data,
      MakeOffsetLayout(VectorLayout{extent}, combined_offset));
}

template<class ParentVector, class IndexType>
   requires requires(const ParentVector& parent)
   {
      ReadHostVector(parent);
   }
auto ReadHostVector(
   const View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return ReadHostVector(std::as_const(vector.data.Get())) +
      vector.layout.BaseOffset();
}

template<class ParentVector, class IndexType>
   requires (
      !std::is_const_v<ParentVector> &&
      requires(ParentVector& parent)
      {
         ReadWriteHostVector(parent);
      })
auto ReadWriteHostVector(
   View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return ReadWriteHostVector(vector.data.Get()) +
      vector.layout.BaseOffset();
}

template<class ParentVector, class IndexType>
   requires (
      !std::is_const_v<ParentVector> &&
      requires(ParentVector& parent)
      {
         WriteHostVector(parent);
      })
auto WriteHostVector(
   View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return WriteHostVector(vector.data.Get()) + vector.layout.BaseOffset();
}

template<class ParentVector, class IndexType>
   requires requires(const ParentVector& parent)
   {
      ReadDeviceVector(parent);
   }
auto ReadDeviceVector(
   const View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return ReadDeviceVector(std::as_const(vector.data.Get())) +
      vector.layout.BaseOffset();
}

template<class ParentVector, class IndexType>
   requires (
      !std::is_const_v<ParentVector> &&
      requires(ParentVector& parent)
      {
         ReadWriteDeviceVector(parent);
      })
auto ReadWriteDeviceVector(
   View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return ReadWriteDeviceVector(vector.data.Get()) +
      vector.layout.BaseOffset();
}

template<class ParentVector, class IndexType>
   requires (
      !std::is_const_v<ParentVector> &&
      requires(ParentVector& parent)
      {
         WriteDeviceVector(parent);
      })
auto WriteDeviceVector(
   View<
      ReferenceContainer<ParentVector, IndexType>,
      OffsetLayout<VectorLayout>>& vector)
{
   return WriteDeviceVector(vector.data.Get()) + vector.layout.BaseOffset();
}

/// Vector type exposing read-only host access.
template < typename VectorType >
concept HostReadableVector =
   requires( const std::remove_reference_t< VectorType > & input )
   {
      { GetVectorSize( input ) } -> std::same_as< size_t >;
      ReadHostVector( input );
   };

/// Vector type exposing mutable host access.
template < typename VectorType >
concept HostWritableVector =
   requires( std::remove_reference_t< VectorType > & output )
   {
      { GetVectorSize( output ) } -> std::same_as< size_t >;
      ReadWriteHostVector( output );
      WriteHostVector( output );
   };

/// Vector type exposing both read-only and mutable host access.
template < typename VectorType >
concept HostAccessibleVector =
   HostReadableVector< VectorType > && HostWritableVector< VectorType >;

/// Vector type exposing read-only device access.
template < typename VectorType >
concept DeviceReadableVector =
   requires( const std::remove_reference_t< VectorType > & input )
   {
      { GetVectorSize( input ) } -> std::same_as< size_t >;
      ReadDeviceVector( input );
   };

/// Vector type exposing mutable device access.
template < typename VectorType >
concept DeviceWritableVector =
   requires( std::remove_reference_t< VectorType > & output )
   {
      { GetVectorSize( output ) } -> std::same_as< size_t >;
      ReadWriteDeviceVector( output );
      WriteDeviceVector( output );
   };

/// Vector type exposing both read-only and mutable device access.
template < typename VectorType >
concept DeviceAccessibleVector =
   DeviceReadableVector< VectorType > && DeviceWritableVector< VectorType >;

/// Vector type readable in the memory space selected by OnDevice.
template < bool OnDevice, typename VectorType >
concept KernelReadableVector =
   ( OnDevice && DeviceReadableVector< VectorType > ) ||
   ( !OnDevice && HostReadableVector< VectorType > );

/// Vector type writable in the memory space selected by OnDevice.
template < bool OnDevice, typename VectorType >
concept KernelWritableVector =
   ( OnDevice && DeviceWritableVector< VectorType > ) ||
   ( !OnDevice && HostWritableVector< VectorType > );

/// Vector type accessible in the memory space selected by OnDevice.
template < bool OnDevice, typename VectorType >
concept KernelAccessibleVector =
   KernelReadableVector< OnDevice, VectorType > &&
   KernelWritableVector< OnDevice, VectorType >;

/**
 * @brief Return a read-only pointer-backed vector view in kernel space.
 *
 * Synchronization is performed before the view is created. The returned value
 * is inexpensive to capture in a host or device kernel and remains valid only
 * while the vector's selected storage remains allocated and unmoved.
 */
template < bool OnDevice, typename VectorType >
requires KernelReadableVector< OnDevice, VectorType >
auto ReadKernelVector( const VectorType & vector )
{
   const size_t extent = GetVectorSize( vector );
   if constexpr ( OnDevice )
   {
      return MakeVectorView( ReadDeviceVector( vector ), extent );
   }
   else
   {
      return MakeVectorView( ReadHostVector( vector ), extent );
   }
}

/**
 * @brief Return a mutable pointer-backed vector view, preserving its values.
 *
 * Acquiring the view invalidates the opposite memory space according to the
 * underlying vector's synchronization contract. Access must remain non-const;
 * a view captured by value in a kernel therefore requires a mutable closure.
 */
template < bool OnDevice, typename VectorType >
requires KernelWritableVector< OnDevice, VectorType >
auto ReadWriteKernelVector( VectorType & vector )
{
   const size_t extent = GetVectorSize( vector );
   if constexpr ( OnDevice )
   {
      return MakeVectorView( ReadWriteDeviceVector( vector ), extent );
   }
   else
   {
      return MakeVectorView( ReadWriteHostVector( vector ), extent );
   }
}

/**
 * @brief Return a write-only pointer-backed vector view in kernel space.
 *
 * No values are synchronized before access. The opposite memory space is
 * invalidated according to the underlying vector's storage contract. Access
 * must remain non-const; a view captured by value in a kernel therefore
 * requires a mutable closure.
 */
template < bool OnDevice, typename VectorType >
requires KernelWritableVector< OnDevice, VectorType >
auto WriteKernelVector( VectorType & vector )
{
   const size_t extent = GetVectorSize( vector );
   if constexpr ( OnDevice )
   {
      return MakeVectorView( WriteDeviceVector( vector ), extent );
   }
   else
   {
      return MakeVectorView( WriteHostVector( vector ), extent );
   }
}

} // namespace gendil
