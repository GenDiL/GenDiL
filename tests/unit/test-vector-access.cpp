// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Algebra/Vector/access.hpp>

#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

using ReadKernelAccess = decltype(
   ReadKernelVector< false >( std::declval< const Vector & >() ) );
using ReadWriteKernelAccess = decltype(
   ReadWriteKernelVector< true >( std::declval< Vector & >() ) );
using WriteKernelAccess = decltype(
   WriteKernelVector< false >( std::declval< Vector & >() ) );

template < class Access >
concept MutableElementAccess = requires( Access & access )
{
   access[0] = Real{};
};

template < class Access >
concept ConstElementAccess = requires( const Access & access )
{
   access[0] = Real{};
};

template<class Index>
concept CanMakeRawVectorView = requires(Real* data, Index extent)
{
   MakeVectorView(data, extent);
};

static_assert(!std::is_constructible_v<VectorLayout, bool>);
static_assert(!std::is_constructible_v<VectorLayout, double>);
static_assert(!std::is_constructible_v<VectorLayout, StaticExtents<6>>);
static_assert(!std::is_constructible_v<VectorLayout, DynamicExtents<1>>);
static_assert(!std::is_constructible_v<VectorLayout, std::array<int, 1>>);
static_assert(!CanMakeRawVectorView<bool>);
static_assert(!CanMakeRawVectorView<double>);
static_assert(CanMakeRawVectorView<int>);
static_assert(CanMakeRawVectorView<size_t>);

static_assert( std::is_same_v<
   decltype( std::declval< PointerContainer< Real > & >()[0] ),
   Real & > );
static_assert( std::is_same_v<
   decltype( std::declval< const PointerContainer< Real > & >()[0] ),
   const Real & > );
static_assert( std::is_same_v<
   decltype( std::declval< PointerContainer< const Real > & >()[0] ),
   const Real & > );
static_assert( std::is_same_v<
   decltype( std::declval< const PointerContainer< const Real > & >()[0] ),
   const Real & > );

static_assert( HostAccessibleVector< Vector > );
static_assert( DeviceAccessibleVector< Vector > );
static_assert( KernelAccessibleVector< false, Vector > );
static_assert( KernelAccessibleVector< true, Vector > );
static_assert(
   std::is_same_v<
      decltype( ReadHostVector( std::declval< const Vector & >() ) ),
      const Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteHostVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( WriteHostVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadDeviceVector( std::declval< const Vector & >() ) ),
      const Real * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteDeviceVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   std::is_same_v<
      decltype( WriteDeviceVector( std::declval< Vector & >() ) ),
      Real * > );
static_assert(
   !std::is_pointer_v< ReadKernelAccess > );
static_assert(
   std::is_same_v< typename ReadKernelAccess::element_type, const Real > );
static_assert(std::is_same_v<
   decltype(std::declval<ReadKernelAccess>().data),
   PointerContainer<const Real>>);
static_assert(
   std::is_same_v<
      decltype( std::declval< const ReadKernelAccess & >()[0] ),
      const Real & > );
static_assert( !std::is_pointer_v< ReadWriteKernelAccess > );
static_assert(
   std::is_same_v<
      typename ReadWriteKernelAccess::element_type,
      Real > );
static_assert(
   std::is_same_v<
      decltype( std::declval< ReadWriteKernelAccess & >()[0] ),
      Real & > );
static_assert(
   std::is_same_v<
      decltype( std::declval< const ReadWriteKernelAccess & >()[0] ),
      const Real & > );
static_assert( !std::is_pointer_v< WriteKernelAccess > );
static_assert(
   std::is_same_v< typename WriteKernelAccess::element_type, Real > );
static_assert(std::is_same_v<
   decltype(std::declval<WriteKernelAccess>().data),
   PointerContainer<Real>>);
static_assert(
   std::is_same_v<
      decltype( std::declval< WriteKernelAccess & >()[0] ),
      Real & > );
static_assert(
   std::is_same_v<
      decltype( std::declval< const WriteKernelAccess & >()[0] ),
      const Real & > );
static_assert( !MutableElementAccess< ReadKernelAccess > );
static_assert( !ConstElementAccess< ReadKernelAccess > );
static_assert( MutableElementAccess< ReadWriteKernelAccess > );
static_assert( !ConstElementAccess< ReadWriteKernelAccess > );
static_assert( MutableElementAccess< WriteKernelAccess > );
static_assert( !ConstElementAccess< WriteKernelAccess > );
static_assert( std::is_trivially_copyable_v< ReadKernelAccess > );
static_assert( std::is_trivially_copyable_v< ReadWriteKernelAccess > );
static_assert( std::is_trivially_copyable_v< WriteKernelAccess > );
static_assert(std::is_same_v<decltype(std::declval<ReadKernelAccess>().layout),
   VectorLayout>);
static_assert(std::is_same_v<decltype(std::declval<ReadWriteKernelAccess>().layout),
   VectorLayout>);
static_assert(std::is_same_v<decltype(std::declval<WriteKernelAccess>().layout),
   VectorLayout>);
static_assert(sizeof(VectorLayout) == sizeof(GlobalIndex));
static_assert(has_strides_v<VectorLayout>);
static_assert(decltype(GetStride<0>(VectorLayout{}))::value == 1);
static_assert(
   sizeof(ReadKernelAccess) ==
      sizeof(PointerContainer<const Real>) + sizeof(VectorLayout));
static_assert(
   sizeof(ReadWriteKernelAccess) ==
      sizeof(PointerContainer<Real>) + sizeof(VectorLayout));
static_assert(
   sizeof(WriteKernelAccess) ==
      sizeof(PointerContainer<Real>) + sizeof(VectorLayout));
#ifdef GENDIL_USE_MFEM
using MFEMMutableSubview = decltype(MakeVectorView(
   std::declval<mfem::Vector&>(), size_t{}, size_t{}));
using MFEMReadOnlySubview = decltype(MakeVectorView(
   std::declval<const mfem::Vector&>(), size_t{}, size_t{}));
static_assert( HostAccessibleVector< mfem::Vector > );
static_assert( DeviceAccessibleVector< mfem::Vector > );
static_assert( KernelAccessibleVector< false, mfem::Vector > );
static_assert( KernelAccessibleVector< true, mfem::Vector > );
static_assert( HostAccessibleVector< MFEMMutableSubview > );
static_assert( DeviceAccessibleVector< MFEMMutableSubview > );
static_assert( HostReadableVector< MFEMReadOnlySubview > );
static_assert( !HostWritableVector< MFEMReadOnlySubview > );
static_assert(
   std::is_same_v<
      decltype( ReadHostVector( std::declval< const mfem::Vector & >() ) ),
      const mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteHostVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( WriteHostVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadDeviceVector( std::declval< const mfem::Vector & >() ) ),
      const mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( ReadWriteDeviceVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   std::is_same_v<
      decltype( WriteDeviceVector( std::declval< mfem::Vector & >() ) ),
      mfem::real_t * > );
static_assert(
   !std::is_pointer_v<
      decltype(
         ReadKernelVector< true >(
            std::declval< const mfem::Vector & >() ) ) > );
#endif

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

// CPU model of independently resident buffers: stale values remain observable
// if a partial write invalidates the only current copy of the untouched entries.
namespace residency_model
{

struct SplitResidentVector
{
   std::array<Real, 5> host{-1, -1, -1, -1, -1};
   std::array<Real, 5> device{10, 20, 30, 40, 50};
   bool host_valid = false;
   bool device_valid = true;
   size_t transfers = 0;
   Real& operator[](size_t i) { return host[i]; }
   const Real& operator[](size_t i) const { return host[i]; }
};

size_t GetVectorSize(const SplitResidentVector&) { return 5; }

Real* WriteHostVector(SplitResidentVector& vector)
{
   vector.host_valid = true;
   vector.device_valid = false;
   return vector.host.data();
}

Real* ReadWriteHostVector(SplitResidentVector& vector)
{
   if (!vector.host_valid)
   {
      vector.host = vector.device;
      ++vector.transfers;
   }
   return WriteHostVector(vector);
}

Real* WriteDeviceVector(SplitResidentVector& vector)
{
   vector.device_valid = true;
   vector.host_valid = false;
   return vector.device.data();
}

Real* ReadWriteDeviceVector(SplitResidentVector& vector)
{
   if (!vector.device_valid)
   {
      vector.device = vector.host;
      ++vector.transfers;
   }
   return WriteDeviceVector(vector);
}

} // namespace residency_model

bool TestPartialWritePreservesResidency()
{
   Vector fresh_host(5), fresh_device(5);
   auto fresh_host_middle = MakeVectorView(fresh_host, size_t{2}, size_t{1});
   auto fresh_device_middle = MakeVectorView(fresh_device, size_t{2}, size_t{1});
   WriteKernelVector<false>(fresh_host_middle)[0] = 8;
   (void)WriteKernelVector<true>(fresh_device_middle);
   bool success = Check(fresh_host.IsHostValid() && ReadHostVector(fresh_host)[2] == 8 &&
      (fresh_device.IsHostValid() || fresh_device.IsDeviceValid()),
      "Partial writes to fresh owners incorrectly required initialized values.");

   residency_model::SplitResidentVector vector;
   auto middle = MakeVectorView(vector, size_t{1}, size_t{3});
   auto nested = MakeVectorView(middle, size_t{1}, size_t{1});
   WriteKernelVector<false>(nested)[0] = 7;
   success = Check(
      vector.host == std::array<Real, 5>{10, 20, 7, 40, 50} &&
      vector.host_valid && !vector.device_valid && vector.transfers == 1,
      "Partial host write discarded current device entries outside its range.") && success;

   WriteKernelVector<true>(middle)[1] = 9;
   success = Check(
      vector.device == std::array<Real, 5>{10, 20, 9, 40, 50} &&
      !vector.host_valid && vector.device_valid && vector.transfers == 2,
      "Partial device write discarded current host entries outside its range.") &&
      success;

   auto complete = MakeVectorView(vector, size_t{0}, size_t{5});
   auto host = WriteKernelVector<false>(complete);
   for (size_t i = 0; i < 5; ++i) { host[i] = Real(i); }
   auto device = WriteKernelVector<true>(complete);
   for (size_t i = 0; i < 5; ++i) { device[i] = Real(2 * i); }
   return Check(vector.transfers == 2,
      "Complete-owner view writes introduced an unnecessary transfer.") && success;
}

bool TestOwningVectorCopy()
{
   Vector source(2), destination(2);
   auto* values = source.WriteHostData();
   values[0] = 3;
   values[1] = 5;
   destination = source;
   source.ReadWriteHostData()[0] = 7;
   const auto* copied = destination.ReadHostData();
   return Check(source.Size() == 2 && destination.Size() == 2 &&
      copied != source.ReadHostData() && copied[0] == 3 && copied[1] == 5,
      "Owning vector assignment did not copy independent entries.");
}

bool TestKernelSubviewAccess()
{
   Vector owner(4);
   auto* values = WriteHostVector(owner);
   for (size_t i = 0; i < 4; ++i) { values[i] = Real(i + 1); }
   auto subrange = MakeVectorView(owner, 1, 2);
   auto input = ReadKernelVector<false>(subrange);
   auto output = WriteKernelVector<false>(subrange);
   static_assert(std::is_trivially_copyable_v<decltype(input)>);
   static_assert(std::is_trivially_copyable_v<decltype(output)>);
   static_assert(std::is_same_v<decltype(input[0]), const Real&>);
   static_assert(std::is_same_v<decltype(output[0]), Real&>);
   static_assert(std::is_same_v<decltype(std::as_const(output)[0]), const Real&>);
   bool success = Check(GetVectorSize(subrange) == 2 && input[0] == 2,
      "Kernel subview read lost its extent or origin.");
   output[0] = 9;
   const auto* result = ReadHostVector(owner);
   return Check(result[0] == 1 && result[1] == 9 && result[2] == 3 && result[3] == 4,
      "Kernel subview write changed entries outside its target.") && success;
}

bool TestGenDiLVectorAccess()
{
   Vector vector( 3 );
   bool success = Check(
      GetVectorSize( vector ) == size_t( 3 ),
      "GetVectorSize returned the wrong GenDiL vector size." );

   auto * host_data = WriteHostVector( vector );
   host_data[0] = 1.0;
   host_data[1] = 2.0;
   host_data[2] = 3.0;
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "WriteHostVector produced the wrong GenDiL validity state." ) &&
      success;

   (void) ReadDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadDeviceVector did not synchronize the GenDiL vector." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   (void) ReadWriteDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      !vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadWriteDeviceVector produced the wrong GenDiL validity state." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadWriteDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   const auto * synchronized_host = ReadHostVector( vector );
   success = Check(
      synchronized_host[0] == 1.0 &&
      synchronized_host[1] == 2.0 &&
      synchronized_host[2] == 3.0,
      "Vector access synchronization changed GenDiL values." ) &&
      success;

   (void) WriteDeviceVector( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      !vector.IsHostValid() && vector.IsDeviceValid(),
      "WriteDeviceVector produced the wrong GenDiL validity state." ) &&
      success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU WriteDeviceVector should use GenDiL host storage." ) &&
      success;
#endif

   return success;
}

bool TestKernelVectorAccess()
{
   Vector vector( 2 );
   auto host_data = WriteKernelVector< false >( vector );
   host_data[0] = Real( 7 );
   host_data[1] = Real( 9 );

   bool success = Check(
      host_data.Size() == size_t( 2 ) &&
         GetVectorSize( host_data ) == size_t( 2 ),
      "Kernel vector access returned the wrong runtime extent." );

   auto kernel_read = ReadKernelVector< true >( vector );
#ifdef GENDIL_USE_DEVICE
   success = Check(
      vector.IsHostValid() && vector.IsDeviceValid(),
      "ReadKernelVector<true> did not synchronize device data." ) && success;
#else
   success = Check(
      vector.IsHostValid() && !vector.IsDeviceValid(),
      "CPU ReadKernelVector<true> should use host storage." ) && success;
#endif

   success = Check(
      kernel_read.Size() == size_t( 2 ),
      "Read kernel view returned the wrong runtime extent." ) && success;
   auto kernel_write = ReadWriteKernelVector< false >( vector );
   kernel_write[1] += Real( 1 );
   success = Check(
      ReadHostVector( vector )[1] == Real( 10 ),
      "Kernel vector access changed synchronized values." ) && success;
   return success;
}

bool TestVectorSubviewMetadata()
{
   Real values[6]{0, 1, 2, 3, 4, 5};
   const auto raw = MakeVectorView(values, size_t{6});
   static_assert(std::is_same_v<decltype(GetExtent<0>(raw)), GlobalIndex>);
   static_assert(std::is_same_v<decltype(raw.layout), VectorLayout>);
   auto raw_slice = MakeVectorView(raw, size_t{2}, size_t{3});
   raw_slice[1] = Real{9};

   Vector storage(6);
   auto* initial = WriteHostVector(storage);
   for (size_t i = 0; i < 6; ++i) { initial[i] = Real(i); }
   auto owner = MakeVectorView(storage, size_t{1}, size_t{4});
   static_assert(HostAccessibleVector<decltype(owner)>);
   static_assert(DeviceAccessibleVector<decltype(owner)>);
   static_assert(std::is_same_v<decltype(GetExtent<0>(owner)), GlobalIndex>);
   static_assert(has_strides_v<decltype(owner.layout)>);
   auto access = ReadWriteKernelVector<false>(owner);
   static_assert(std::is_same_v<decltype(access.layout), VectorLayout>);
   access[0] = Real{10};
   auto nested = [&]
   {
      auto local = MakeVectorView(owner, size_t{0}, size_t{4});
      return MakeVectorView(local, size_t{1}, size_t{2});
   }();
   WriteHostVector(nested)[1] = Real{30};

   const auto read_only = MakeVectorView(
      std::as_const(storage), size_t{1}, size_t{4});
   static_assert(HostReadableVector<decltype(read_only)>);
   static_assert(!HostWritableVector<decltype(read_only)>);
   static_assert(DeviceReadableVector<decltype(read_only)>);
   static_assert(!DeviceWritableVector<decltype(read_only)>);
   return Check(
      values[3] == Real{9} && GetVectorSize(raw_slice) == 3 &&
      GetExtentValue<0>(raw) == 6 && GetVectorSize(owner) == 4 &&
      ReadHostVector(read_only)[0] == Real{10} &&
      ReadHostVector(storage)[3] == Real{30} &&
      &nested.data.Get() == &storage && nested.layout.BaseOffset() == 2,
      "Vector subview changed access, size, or nested ownership.");
}

#ifdef GENDIL_USE_MFEM
bool TestMFEMVectorAccess()
{
   mfem::Vector vector( 3 );
   auto * data = WriteHostVector( vector );
   data[0] = 4.0;
   data[1] = 5.0;
   data[2] = 6.0;

   const auto * read = ReadHostVector( vector );
   bool success =
      Check(
         GetVectorSize( vector ) == size_t( 3 ),
         "GetVectorSize returned the wrong MFEM vector size." ) &&
      Check(
         read[0] == 4.0 && read[1] == 5.0 && read[2] == 6.0,
         "MFEM host vector access changed values." );

   return success;
}
#endif

} // namespace

int main()
{
   bool success = TestGenDiLVectorAccess();
   success = TestOwningVectorCopy() && success;
   success = TestKernelSubviewAccess() && success;
   success = TestKernelVectorAccess() && success;
   success = TestVectorSubviewMetadata() && success;
   success = TestPartialWritePreservesResidency() && success;
#ifdef GENDIL_USE_MFEM
   success = TestMFEMVectorAccess() && success;
#endif
   return success ? 0 : 1;
}
