// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Keep the public include first: full logical access must find its own get()
// dependency, without the test pre-including the index-sequence helper.
#include <gendil/gendil.hpp>

#include <iostream>
#include <type_traits>
#include <utility>

using namespace gendil;

namespace
{

// These are indexing metadata only. ThreadedView projects onto the calling
// thread's registers; this test does not request inter-thread communication.
struct ReorderedContext
{
   template < Integer > using threaded_dimensions = std::index_sequence<1>;
   template < Integer > using register_dimensions = std::index_sequence<2, 0>;
};

struct NoncontiguousContext
{
   template < Integer > using threaded_dimensions = std::index_sequence<0, 2>;
   template < Integer > using register_dimensions = std::index_sequence<3, 1>;
};

using PrefixContext = ThreadFirstKernelConfiguration<ThreadBlockLayout<2>, 1>;
using AllThreadedContext =
   ThreadFirstKernelConfiguration<ThreadBlockLayout<2, 3, 5>, 3>;

bool Check( const bool condition, const char * message )
{
   if ( !condition ) { std::cout << message << '\n'; }
   return condition;
}

template < typename Threaded >
GENDIL_HOST_DEVICE
constexpr void CheckRepresentation()
{
   using Registers = decltype( std::declval<Threaded>().data );
   static_assert( sizeof(Threaded) == sizeof(Registers) );
   static_assert( std::is_trivially_copyable_v<Registers> );
   static_assert( std::is_trivially_copyable_v<Threaded> );
}

template < typename Registers >
GENDIL_HOST_DEVICE
bool TestSilentProjection( const Registers & registers )
{
   using Threaded = ThreadedView<std::index_sequence<2, 3, 2>,
      ReorderedContext, Registers>;
   CheckRepresentation<Threaded>();
   static_assert( std::is_same_v<typename Threaded::register_shape,
      std::index_sequence<2, 2>> );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(1, 0, 0)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(1, 0, 0)), const Real&> );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(0, 1)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(0, 1)), const Real&> );
   Threaded view{ registers };
   view.data(0, 0) = 11;
   view.data(1, 0) = 12;
   view.data(0, 1) = 21;
   view.data(1, 1) = 22;
   const auto & read_only = view;

   // The old extent-as-axis projection selected (0, 0) for (1, 0, 0),
   // silently returning 11 instead of 21, with every index still in bounds.
   for ( int x = 0; x < 2; ++x )
   {
      for ( int thread = 0; thread < 3; ++thread )
      {
         for ( int z = 0; z < 2; ++z )
         {
            const Real expected = Real(11 + 10 * x + z);
            if ( view(x, thread, z) != expected ||
                 read_only(x, thread, z) != expected ||
                 view(z, x) != expected || read_only(z, x) != expected ||
                 &view(x, thread, z) != &view.data(z, x) ||
                 &read_only(x, thread, z) != &read_only.data(z, x) )
            { return false; }
         }
      }
   }
   view(1, 0, 0) = 37;
   return view.data(0, 1) == 37 && read_only(0, 1) == 37 &&
      read_only(1, 2, 0) == 37 && view.data(0, 0) == 11 &&
      view.data(1, 0) == 12 && view.data(1, 1) == 22;
}

template < typename Registers >
GENDIL_HOST_DEVICE
bool TestThreadFirstProjection( const Registers & registers )
{
   using Threaded = ThreadedView<std::index_sequence<2, 3, 5>,
      PrefixContext, Registers>;
   CheckRepresentation<Threaded>();
   static_assert( std::is_same_v<typename Threaded::register_shape,
      std::index_sequence<3, 5>> );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(1, 2, 4)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(1, 2, 4)), const Real&> );
   Threaded view{ registers };
   for ( int y = 0; y < 3; ++y )
   {
      for ( int z = 0; z < 5; ++z ) { view.data(y, z) = Real(10 * z + y + 1); }
   }
   const auto & read_only = view;
   for ( int thread = 0; thread < 2; ++thread )
   {
      for ( int y = 0; y < 3; ++y )
      {
         for ( int z = 0; z < 5; ++z )
         {
            const Real expected = Real(10 * z + y + 1);
            if ( view(thread, y, z) != expected ||
                 read_only(thread, y, z) != expected ||
                 view(y, z) != expected || read_only(y, z) != expected ||
                 &view(thread, y, z) != &view.data(y, z) ||
                 &read_only(thread, y, z) != &read_only.data(y, z) )
            { return false; }
         }
      }
   }
   // Extent 5 exceeds logical rank 3, so extents cannot be valid axis IDs.
   view(1, 2, 4) = 83;
   return view.data(2, 4) == 83 && read_only(0, 2, 4) == 83 &&
      read_only(2, 4) == 83 && view.data(1, 4) == 42;
}

template < typename Registers >
bool TestNoncontiguousProjection( const Registers & registers )
{
   using Threaded = ThreadedView<std::index_sequence<2, 5, 3, 7>,
      NoncontiguousContext, Registers>;
   CheckRepresentation<Threaded>();
   static_assert( std::is_same_v<typename Threaded::register_shape,
      std::index_sequence<7, 5>> );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(1, 4, 2, 6)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(1, 4, 2, 6)), const Real&> );
   Threaded view{ registers };
   for ( int y = 0; y < 5; ++y )
   {
      for ( int w = 0; w < 7; ++w ) { view.data(w, y) = Real(10 * y + w + 1); }
   }
   const auto & read_only = view;
   for ( int x = 0; x < 2; ++x )
   {
      for ( int y = 0; y < 5; ++y )
      {
         for ( int z = 0; z < 3; ++z )
         {
            for ( int w = 0; w < 7; ++w )
            {
               const Real expected = Real(10 * y + w + 1);
               if ( view(x, y, z, w) != expected ||
                    read_only(x, y, z, w) != expected ||
                    view(w, y) != expected || read_only(w, y) != expected ||
                    &view(x, y, z, w) != &view.data(w, y) ||
                    &read_only(x, y, z, w) != &read_only.data(w, y) )
               { return false; }
            }
         }
      }
   }
   view(1, 4, 2, 6) = 97;
   return view.data(6, 4) == 97 && read_only(0, 4, 0, 6) == 97 &&
      read_only(6, 4) == 97 && view.data(5, 4) == 46;
}

template < typename Registers >
bool TestHostProjection( const Registers & registers )
{
   using Threaded = ThreadedView<std::index_sequence<2, 3>,
      HostKernelConfiguration, Registers>;
   CheckRepresentation<Threaded>();
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(1, 2)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(1, 2)), const Real&> );
   Threaded view{ registers };
   const auto & read_only = view;
   for ( int x = 0; x < 2; ++x )
   {
      for ( int y = 0; y < 3; ++y )
      {
         view.data(x, y) = Real(10 * y + x + 1);
         if ( view(x, y) != Real(10 * y + x + 1) ||
              read_only(x, y) != Real(10 * y + x + 1) ||
              &view(x, y) != &view.data(x, y) ||
              &read_only(x, y) != &read_only.data(x, y) )
         { return false; }
      }
   }
   view(1, 2) = 73;
   return view.data(1, 2) == 73 && read_only(1, 2) == 73;
}

template < typename Registers >
GENDIL_HOST_DEVICE
bool TestScalarProjections( const Registers & registers )
{
   using Threaded = ThreadedView<std::index_sequence<2, 3, 5>,
      AllThreadedContext, Registers>;
   using Scalar = ThreadedView<std::index_sequence<>,
      HostKernelConfiguration, Registers>;
   CheckRepresentation<Threaded>();
   CheckRepresentation<Scalar>();
   static_assert( Threaded::rank == 3 && Threaded::register_shape::size() == 0 );
   static_assert( Scalar::rank == 0 && Scalar::register_shape::size() == 0 );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()(1, 2, 4)), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()(1, 2, 4)), const Real&> );
   static_assert( std::is_same_v<decltype(std::declval<Threaded&>()()), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Threaded&>()()), const Real&> );
   static_assert( std::is_same_v<decltype(std::declval<Scalar&>()()), Real&> );
   static_assert( std::is_same_v<decltype(std::declval<const Scalar&>()()), const Real&> );
   Threaded view{ registers };
   view.data() = 91;
   const auto & read_only = view;
   if ( view(1, 2, 4) != 91 || read_only(0, 0, 0) != 91 ||
        view() != 91 || read_only() != 91 ||
        &view(1, 2, 4) != &view.data() ||
        &read_only(1, 2, 4) != &read_only.data() )
   { return false; }
   view(1, 2, 4) = 92;
   if ( view.data() != 92 || read_only() != 92 ) { return false; }

   Scalar scalar{ registers };
   scalar.data() = 61;
   const auto & const_scalar = scalar;
   if ( scalar() != 61 || const_scalar() != 61 ||
        &scalar() != &scalar.data() || &const_scalar() != &const_scalar.data() )
   { return false; }
   scalar() = 62;
   return scalar.data() == 62 && const_scalar() == 62;
}

bool TestStoredProjections()
{
   Real values[35]{};
   bool success = Check(TestSilentProjection(MakeFixedFIFOView<2, 2>(values)),
      "Pointer-backed reordered projection failed.");
   success = Check(values[0] == 11 && values[1] == 12 &&
      values[2] == 37 && values[3] == 22,
      "Full logical write did not reach the expected backing element.") && success;
   success = Check(TestSilentProjection(MakeStaticFIFOView<Real, 2, 2>()),
      "Inline reordered projection failed.") && success;
   success = Check(TestThreadFirstProjection(MakeFixedFIFOView<3, 5>(values)),
      "Pointer-backed thread-first projection failed.") && success;
   success = Check(TestThreadFirstProjection(MakeStaticFIFOView<Real, 3, 5>()),
      "Inline thread-first projection failed.") && success;
   success = Check(TestNoncontiguousProjection(MakeFixedFIFOView<7, 5>(values)),
      "Pointer-backed noncontiguous projection failed.") && success;
   success = Check(TestNoncontiguousProjection(MakeStaticFIFOView<Real, 7, 5>()),
      "Inline noncontiguous projection failed.") && success;
   success = Check(TestHostProjection(MakeFixedFIFOView<2, 3>(values)),
      "Pointer-backed all-register host access failed.") && success;
   success = Check(TestHostProjection(MakeStaticFIFOView<Real, 2, 3>()),
      "Inline all-register host access failed.") && success;
   success = Check(TestScalarProjections(MakeFixedFIFOView<>(values)),
      "Pointer-backed scalar register/logical access failed.") && success;
   success = Check(TestScalarProjections(MakeStaticFIFOView<Real>()),
      "Inline scalar register/logical access failed.") && success;

   const Real read_only_values[]{11, 12, 21, 22};
   auto registers = MakeFixedFIFOView<2, 2>(read_only_values);
   using ReadOnly = ThreadedView<std::index_sequence<2, 3, 2>,
      ReorderedContext, decltype(registers)>;
   ReadOnly read_only{ registers };
   const auto & const_read_only = read_only;
   CheckRepresentation<ReadOnly>();
   static_assert( std::is_same_v<decltype(read_only(1, 0, 0)), const Real&> );
   static_assert( std::is_same_v<decltype(const_read_only(1, 0, 0)), const Real&> );
   success = Check(read_only(1, 0, 0) == 21 && const_read_only(1, 0, 0) == 21 &&
      &read_only(1, 0, 0) == read_only_values + 2 &&
      &const_read_only(1, 0, 0) == read_only_values + 2,
      "Const-pointee full logical access failed.") && success;
   return success;
}

template < bool OnDevice >
bool TestKernelProjection()
{
   constexpr GlobalIndex work_items = 7;
   constexpr GlobalIndex stride = 5;
   Vector output(work_items * stride);
   auto output_access = WriteKernelVector<OnDevice>(output);
   KernelLoop<OnDevice>(work_items,
      [=] GENDIL_HOST_DEVICE ( const GlobalIndex item ) mutable
      {
         auto registers = MakeFixedFIFOView<2, 2>(
            output_access.data.data + item * stride);
         const bool pointer_ok = TestSilentProjection(registers);
         const bool inline_ok = TestThreadFirstProjection(
            MakeStaticFIFOView<Real, 3, 5>());
         // Construct rank-zero register layouts inside the kernel so device
         // compilers check the scalar path used by fully threaded tensors.
         Real scalar_value{};
         const bool scalar_pointer_ok = TestScalarProjections(
            MakeFixedFIFOView<>(&scalar_value));
         const bool scalar_inline_ok = TestScalarProjections(
            MakeStaticFIFOView<Real>());
         output_access[item * stride + 4] =
            pointer_ok && inline_ok && scalar_pointer_ok && scalar_inline_ok
               ? Real{1} : Real{-1};
      });
   const Real * values = ReadHostVector(output);
   bool success = true;
   for ( GlobalIndex item = 0; item < work_items; ++item )
   {
      const Real * result = values + item * stride;
      success = Check(result[0] == 11 && result[1] == 12 && result[2] == 37 &&
         result[3] == 22 && result[4] == 1,
         "KernelLoop threaded-view full logical access failed.") && success;
   }
   return success;
}

} // namespace

int main()
{
   bool success = TestStoredProjections();
   success = TestKernelProjection<false>() && success;
   success = TestKernelProjection<true>() && success;
#ifdef GENDIL_USE_DEVICE
   std::cout << "Threaded-view indexing: KernelLoop<false> host and "
      "KernelLoop<true> device execution exercised.\n";
#else
   std::cout << "Threaded-view indexing: KernelLoop<false> host and "
      "KernelLoop<true> CPU fallback exercised; no device backend enabled.\n";
#endif
   return success ? 0 : 1;
}
