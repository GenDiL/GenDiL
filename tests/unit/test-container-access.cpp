// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Utilities/Loop/kernelloop.hpp>
#include <gendil/Utilities/View/Layouts/vectorlayout.hpp>
#include <gendil/Utilities/arraytraits.hpp>

#include <array>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

static_assert(is_std_array_v<std::array<int, 3>>);
static_assert(is_std_array_v<const std::array<double, 0>&>);
static_assert(!is_std_array_v<int>);
static_assert(!is_std_array_v<int[3]>);

template < class Container, class Index = GlobalIndex >
using Access = decltype(
   std::declval< Container >()[std::declval< const Index & >()] );

template < class Container, class Index = GlobalIndex >
concept HasSubscript = requires
{
   std::declval< Container >()[std::declval< const Index & >()];
};

template < class Container >
concept HasCall = requires
{
   std::declval< Container >()( GlobalIndex{ 0 } );
};

// The four expected types are ordered &, const &, &&, const &&.
template < class Container, class L, class CL, class R, class CR >
constexpr bool AccessTypes =
   std::is_same_v< Access< Container & >, L > &&
   std::is_same_v< Access< const Container & >, CL > &&
   std::is_same_v< Access< Container && >, R > &&
   std::is_same_v< Access< const Container && >, CR >;

struct RefOrValueParent
{
   using element_type = Real;
   Real values[3];

   GENDIL_HOST_DEVICE
   Real & operator[]( GlobalIndex i ) & { return values[i]; }
   GENDIL_HOST_DEVICE
   const Real & operator[]( GlobalIndex i ) const & { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( GlobalIndex i ) && { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( GlobalIndex i ) const && { return values[i]; }
};

struct ReadProxy
{
   const Real * value;
   GENDIL_HOST_DEVICE operator Real() const { return *value; }
};

struct WriteProxy
{
   Real * value;
   GENDIL_HOST_DEVICE operator Real() const { return *value; }
   GENDIL_HOST_DEVICE
   WriteProxy & operator=( Real next ) { *value = next; return *this; }
};

struct ProxyParent
{
   using element_type = Real;
   Real * values;

   GENDIL_HOST_DEVICE
   WriteProxy operator[]( GlobalIndex i ) { return { values + i }; }
   GENDIL_HOST_DEVICE
   ReadProxy operator[]( GlobalIndex i ) const { return { values + i }; }
};

// External keys convert to map offsets, but the parent's actual key is const int&.
struct ExternalIndex
{
   GlobalIndex value;
   GENDIL_HOST_DEVICE operator GlobalIndex() const { return value; }
};

struct MappedIndexParent
{
   using element_type = Real;
   Real * values;

   GENDIL_HOST_DEVICE
   Real & operator[]( const int & i ) & { return values[i]; }
   GENDIL_HOST_DEVICE
   const Real & operator[]( const int & i ) const & { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( const int & i ) && { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( const int & i ) const && { return values[i]; }

   GENDIL_HOST_DEVICE Real operator[]( const ExternalIndex & ) && = delete;
   GENDIL_HOST_DEVICE Real operator[]( const ExternalIndex & ) const && = delete;
   GENDIL_HOST_DEVICE Real operator[]( int && ) && = delete;
   GENDIL_HOST_DEVICE Real operator[]( int && ) const && = delete;
};

struct ExternalOnlyRvalueParent
{
   using element_type = Real;
   Real * values;

   GENDIL_HOST_DEVICE
   Real & operator[]( const int & i ) & { return values[i]; }
   GENDIL_HOST_DEVICE
   const Real & operator[]( const int & i ) const & { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( const ExternalIndex & i ) && { return values[i.value]; }
   GENDIL_HOST_DEVICE
   Real operator[]( const ExternalIndex & i ) const && { return values[i.value]; }
   GENDIL_HOST_DEVICE
   Real operator[]( int && i ) && { return values[i]; }
   GENDIL_HOST_DEVICE
   Real operator[]( int && i ) const && { return values[i]; }

   GENDIL_HOST_DEVICE Real operator[]( const int & ) && = delete;
   GENDIL_HOST_DEVICE Real operator[]( const int & ) const && = delete;
};

struct ExternalLayout
{
   static constexpr size_t rank = 1;
   GENDIL_HOST_DEVICE
   ExternalIndex Offset( GlobalIndex i ) const { return { i }; }
};

using Inline = StaticContainer< Real, 3 >;
using Pointer = PointerContainer< Real >;
using ConstPointer = PointerContainer< const Real >;
using IndirectPointer = IndirectedContainer< Pointer >;
using IndirectConstPointer = IndirectedContainer< ConstPointer >;
using IndirectHandle = IndirectedContainer< HostDevicePointer< Real > >;
using IndirectRawPointer = IndirectedContainer< Real * >;
static_assert(std::is_same_v<IndirectRawPointer::element_type, Real>);
using IndirectInline = IndirectedContainer< Inline >;
using IndirectValue = IndirectedContainer< RefOrValueParent >;
using IndirectProxy = IndirectedContainer< ProxyParent >;
using Reference = ReferenceContainer< RefOrValueParent >;
using ConstReference = ReferenceContainer< const RefOrValueParent >;
using ShallowReference = ReferenceContainer< HostDevicePointer< Real > >;
using ConstShallowReference = ReferenceContainer< const HostDevicePointer< Real > >;
using ReadOnlyReference = ReferenceContainer< HostDevicePointer< const Real > >;

static_assert( AccessTypes< Pointer, Real &, const Real &, Real &, const Real & > );
static_assert( AccessTypes< ConstPointer,
   const Real &, const Real &, const Real &, const Real & > );
static_assert( AccessTypes< IndirectPointer,
   Real &, const Real &, Real &, const Real & > );
static_assert( AccessTypes< IndirectConstPointer,
   const Real &, const Real &, const Real &, const Real & > );
static_assert( AccessTypes< IndirectHandle, Real &, Real &, Real &, Real & > );
static_assert( AccessTypes< IndirectRawPointer, Real &, Real &, Real &, Real & > );
static_assert( AccessTypes< IndirectValue, Real &, const Real &, Real, Real > );
static_assert( AccessTypes< IndirectProxy,
   WriteProxy, ReadProxy, WriteProxy, ReadProxy > );
static_assert( AccessTypes< Reference, Real &, const Real &, Real &, const Real & > );
static_assert( AccessTypes< ConstReference,
   const Real &, const Real &, const Real &, const Real & > );
static_assert( AccessTypes< ShallowReference, Real &, Real &, Real &, Real & > );
static_assert( AccessTypes< ConstShallowReference, Real &, Real &, Real &, Real & > );
static_assert( AccessTypes< ReadOnlyReference,
   const Real &, const Real &, const Real &, const Real & > );
static_assert( AccessTypes< IndirectedContainer< Reference >,
   Real &, const Real &, Real &, const Real & > );
static_assert( AccessTypes< IndirectedContainer< ShallowReference >,
   Real &, Real &, Real &, Real & > );

static_assert( std::is_same_v< Access< Inline & >, Real & > );
static_assert( std::is_same_v< Access< const Inline & >, const Real & > );
static_assert( std::is_same_v< Access< IndirectInline & >, Real & > );
static_assert( std::is_same_v< Access< const IndirectInline & >, const Real & > );
static_assert( !HasSubscript< Inline && > && !HasSubscript< const Inline && > );
static_assert( !HasSubscript< IndirectInline && > &&
               !HasSubscript< const IndirectInline && > );

using InlineView = View< Inline, VectorLayout >;
using IndirectInlineView = View< IndirectInline, VectorLayout >;
using IndirectPointerView = View< IndirectPointer, VectorLayout >;
using IndirectValueView = View< IndirectValue, VectorLayout >;
using IndirectProxyView = View< IndirectProxy, VectorLayout >;

static_assert( !HasSubscript< InlineView && > &&
               !HasSubscript< const InlineView && > );
static_assert( !HasCall< InlineView && > && !HasCall< const InlineView && > );
static_assert( !HasSubscript< IndirectInlineView && > &&
               !HasSubscript< const IndirectInlineView && > );
static_assert( !HasCall< IndirectInlineView && > &&
               !HasCall< const IndirectInlineView && > );
static_assert( AccessTypes< IndirectPointerView,
   Real &, const Real &, Real &, const Real & > );
static_assert( AccessTypes< IndirectValueView, Real &, const Real &, Real, Real > );
static_assert( AccessTypes< IndirectProxyView,
   WriteProxy, ReadProxy, WriteProxy, ReadProxy > );

static_assert( std::is_constructible_v< Reference, RefOrValueParent & > );
static_assert( std::is_constructible_v< ConstReference, const RefOrValueParent & > );
static_assert( !std::is_constructible_v< Reference, RefOrValueParent && > );
static_assert( !std::is_constructible_v< ConstReference, RefOrValueParent && > );
static_assert( !std::is_constructible_v< ConstReference, const RefOrValueParent && > );
static_assert( std::is_same_v<
   decltype( std::declval< Reference & >().Get() ), RefOrValueParent & > );
static_assert( std::is_same_v<
   decltype( std::declval< const Reference & >().Get() ), const RefOrValueParent & > );

using Mapped = IndirectedContainer< MappedIndexParent, ExternalIndex >;
using RejectedMapped = IndirectedContainer< ExternalOnlyRvalueParent, ExternalIndex >;
using MappedView = View< Mapped, ExternalLayout >;
using RejectedMappedView = View< RejectedMapped, ExternalLayout >;

static_assert( std::is_same_v< Access< Mapped &, ExternalIndex >, Real & > );
static_assert( std::is_same_v< Access< const Mapped &, ExternalIndex >, const Real & > );
static_assert( std::is_same_v< Access< Mapped &&, ExternalIndex >, Real > );
static_assert( std::is_same_v< Access< const Mapped &&, ExternalIndex >, Real > );
static_assert( !HasSubscript< RejectedMapped &&, ExternalIndex > &&
               !HasSubscript< const RejectedMapped &&, ExternalIndex > );
static_assert( AccessTypes< MappedView, Real &, const Real &, Real, Real > );
static_assert( HasCall< MappedView && > && HasCall< const MappedView && > );
static_assert( !HasSubscript< RejectedMappedView && > &&
               !HasSubscript< const RejectedMappedView && > );
static_assert( !HasCall< RejectedMappedView && > &&
               !HasCall< const RejectedMappedView && > );

bool Check( bool condition, const char * message )
{
   if ( !condition ) { std::cerr << message << '\n'; }
   return condition;
}

HostDevicePointer< const int > HostMap( const int * indices )
{
   HostDevicePointer< const int > map;
   map.host_pointer = indices;
   return map;
}

bool TestIndirectReferences()
{
   Real raw_values[2]{3, 7};
   const int raw_indices[2]{1, 0};
   IndirectRawPointer raw{raw_values, HostMap(raw_indices)};
   bool success = Check(raw[0] == 7 && raw[1] == 3,
      "Raw-pointer indirection ignored the nonidentity map.");

   Real values[3]{ 10, 20, 30 };
   const int indices[3]{ 2, 0, 1 };
   const auto map = HostMap( indices );
   IndirectPointer indirect{ Pointer{ values }, map };
   const auto & read = indirect;
   success = Check(
      &indirect[0] == &values[2] && &read[1] == &values[0] && read[2] == 20,
      "Indirected pointer access ignored the nonidentity map or copied a reference." ) && success;

   indirect[0] = 31;
   std::move( indirect )[1] = 11;
   IndirectPointerView view{ { Pointer{ values }, map }, VectorLayout{ 3 } };
   view( 2 ) = 21;
   std::move( view )[0] += 1;
   const IndirectPointerView const_view{
      { Pointer{ values }, map }, VectorLayout{ 3 } };
   success = Check(
      values[0] == 11 && values[1] == 21 && values[2] == 32 &&
      std::move( const_view )( 0 ) == 32 && std::move( read )[2] == 21,
      "Indirected wrapper/View writes or const-rvalue reads reached the wrong element." ) && success;

   IndirectInline inline_access{ { { 4, 5, 6 } }, map };
   inline_access[0] = 7;
   return Check(
      std::as_const( inline_access )[0] == 7 && inline_access.data.data[2] == 7,
      "Lvalue inline-storage indirection did not preserve reference access." ) && success;
}

bool TestValueAndProxyResults()
{
   const int indices[3]{ 2, 0, 1 };
   const auto map = HostMap( indices );
   IndirectValue indirect{ { { 10, 20, 30 } }, map };
   indirect[0] = 31;
   const IndirectValue const_indirect{ { { 40, 50, 60 } }, map };

   // These full-expression temporaries must return values, never references to
   // parent operator[] return temporaries (the stack-use-after-return regression).
   const Real temporary_value = IndirectValue{ { { 1, 2, 3 } }, map }[0];
   const Real temporary_view_value = IndirectValueView{
      { { { 4, 5, 6 } }, map }, VectorLayout{ 3 } }( 1 );
   bool success = Check(
      std::move( indirect )[0] == 31 && std::move( const_indirect )[1] == 40 &&
      temporary_value == 3 && temporary_view_value == 4,
      "Value-returning parent access did not survive wrapper/View temporary access." );

   Real values[3]{ 10, 20, 30 };
   IndirectProxy proxies{ { values }, map };
   proxies[0] = 32;
   std::move( proxies )[1] = 12;
   const IndirectProxy const_proxies{ { values }, map };
   IndirectProxyView proxy_view{ { { values }, map }, VectorLayout{ 3 } };
   proxy_view( 2 ) = 22;
   std::move( proxy_view )[0] = 33;
   return Check(
      values[0] == 12 && values[1] == 22 && values[2] == 33 &&
      Real( const_proxies[0] ) == 33 && Real( std::move( const_proxies )[2] ) == 22,
      "Indirected access lost proxy read/write semantics." ) && success;
}

bool TestMappedIndexAvailability()
{
   Real values[3]{ 10, 20, 30 };
   const int indices[3]{ 2, 0, 1 };
   const auto map = HostMap( indices );
   Mapped mapped{ { values }, map };
   const Mapped const_mapped{ { values }, map };
   mapped[ExternalIndex{ 1 }] = 11;
   MappedView view{ { { values }, map }, ExternalLayout{} };
   const MappedView const_view{ { { values }, map }, ExternalLayout{} };
   return Check(
      std::move( mapped )[ExternalIndex{ 0 }] == 30 &&
      std::move( const_mapped )[ExternalIndex{ 1 }] == 11 &&
      std::move( view )( 2 ) == 20 && std::move( const_view )[0] == 30,
      "Rvalue availability was not based on the exact mapped const-int-reference key." );
}

bool TestReferenceDelegation()
{
   RefOrValueParent parent{ { 10, 20, 30 } };
   Reference reference( parent );
   reference[0] = 11;
   std::move( reference )[1] = 21;
   const Reference const_reference( parent );
   const ConstReference read_only( parent );
   bool success = Check(
      &const_reference[0] == &parent.values[0] &&
      &std::move( const_reference )[1] == &parent.values[1] &&
      &read_only.Get() == &parent && read_only[0] == 11 && read_only[1] == 21,
      "ReferenceContainer did not delegate const/rvalue access to its lvalue parent." );

   HostDevicePointer< Real > pointer;
   pointer.host_pointer = parent.values;
   const ShallowReference shallow( pointer );
   const ConstShallowReference const_parent( pointer );
   shallow[0] = 12;
   std::move( const_parent )[1] = 22;
   const int indices[3]{ 2, 0, 1 };
   const IndirectedContainer< ShallowReference > indirect{
      ShallowReference( pointer ), HostMap( indices ) };
   indirect[0] = 32;
   return Check(
      parent.values[0] == 12 && parent.values[1] == 22 && parent.values[2] == 32 &&
      &shallow.Get() == &pointer,
      "ReferenceContainer or indirection incorrectly deepened a shallow parent's constness." ) && success;
}

template < bool OnDevice >
bool TestKernelLocalReference()
{
   Real values[3]{ 10, 20, 30 };
   int indices[3]{ 2, 0, 1 };
   HostDevicePointer< Real > storage;
   storage.host_pointer = values;
   HostDevicePointer< int > index_storage;
   index_storage.host_pointer = indices;
   if constexpr ( OnDevice )
   {
      AllocateDevicePointer( 3, storage );
      AllocateDevicePointer( 3, index_storage );
      ToDevice( 3, storage );
      ToDevice( 3, index_storage );
   }
   auto map = HostMap( indices );
#ifdef GENDIL_USE_DEVICE
   map.device_pointer = index_storage.device_pointer;
#endif

   KernelLoop< OnDevice >(
      GlobalIndex{ 3 },
      [=] GENDIL_HOST_DEVICE ( GlobalIndex i )
      {
         // Bind on the executing side, never capture a host-created reference.
         Pointer parent{ storage };
         ReferenceContainer< Pointer > reference( parent );
         IndirectedContainer< decltype( reference ) > indirect{ reference, map };
         const auto & read = indirect;
         const Real previous = read[i];
         std::move( indirect )[i] = previous + Real( 7 * ( i + 1 ) );
      } );

   if constexpr ( OnDevice )
   {
      ToHost( 3, storage );
      FreeDevicePointer( storage );
      FreeDevicePointer( index_storage );
   }
   return Check(
      values[0] == 24 && values[1] == 41 && values[2] == 37,
      "Kernel-local ReferenceContainer indirect read/write failed." );
}

} // namespace

int main()
{
   bool success = TestIndirectReferences();
   success = TestValueAndProxyResults() && success;
   success = TestMappedIndexAvailability() && success;
   success = TestReferenceDelegation() && success;
   success = TestKernelLocalReference< false >() && success;
   success = TestKernelLocalReference< true >() && success;
   return success ? 0 : 1;
}
