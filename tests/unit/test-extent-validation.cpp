// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

// Layout preconditions are debug-only. Exercise their diagnostics even when
// this fixture is built in an otherwise release-configured test suite.
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "gendil/Algebra/Vector/access.hpp"
#include "gendil/Utilities/View/extents.hpp"
#include "gendil/Utilities/View/Layouts/stridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/fixedstridedlayout.hpp"
#include "gendil/Utilities/View/Layouts/orientedlayout.hpp"
#include "gendil/Utilities/View/Layouts/offsetlayout.hpp"

#include <limits>
#include <string_view>

namespace {

// Query-only shapes remain supported by GetLogicalSize. Every axis is checked,
// even after an empty axis, and a custom total cannot bypass those checks.
struct QueryShape
{
   static constexpr size_t rank = 2;
   gendil::GlobalIndex first;
   long second;
   constexpr auto Get( gendil::AxisTag< 0 > ) const { return first; }
   constexpr auto Get( gendil::AxisTag< 1 > ) const { return second; }
   constexpr gendil::GlobalIndex LogicalSize() const { return 0; }
};

} // namespace

/**
 * Runtime rejection fixture. Each named case must terminate through GenDiL's
 * verification/assertion boundary; a normal return indicates an incorrectly accepted
 * value. The test harness must check abnormal/nonzero process termination.
 */
int main( int argc, char ** argv )
{
   using namespace gendil;
   if ( argc != 2 ) { return 2; }
   const std::string_view scenario{ argv[1] };
   if ( scenario == "negative" )
   {
      (void)DynamicExtents< 1 >{ -1 };
   }
   else if ( scenario == "query_negative_after_zero" )
   {
      (void)GetLogicalSize( QueryShape{ 0, -1 } );
   }
   else if ( scenario == "query_overflow" )
   {
      (void)GetLogicalSize( QueryShape{
         std::numeric_limits< GlobalIndex >::max(), 2 } );
   }
   else if ( scenario == "narrowing" )
   {
      (void)detail::CheckedExtentCast< unsigned char >( 256 );
   }
   else if ( scenario == "signed_narrowing" )
   {
      (void)detail::CheckedExtentCast< signed char >( 128 );
   }
   else if ( scenario == "array_negative" )
   {
      (void)DynamicExtents<2>{ std::array< int, 2 >{ 0, -1 } };
   }
   else if ( scenario == "product_overflow" )
   {
      (void)DynamicExtents< 2 >{
         std::numeric_limits< GlobalIndex >::max(), 2 };
   }
   else if ( scenario == "negative_stride" )
   {
      (void)StridedLayout< 1 >{ -1 };
   }
   else if ( scenario == "fifo_negative_after_zero" )
   {
      (void)MakeFIFOStridedLayout( std::array{ 0, -1 } );
   }
   else if ( scenario == "fifo_overflow_before_zero" )
   {
      (void)MakeFIFOStridedLayout( std::array< GlobalIndex, 3 >{
         std::numeric_limits< GlobalIndex >::max(), 2, 0 } );
   }
   else if ( scenario == "permuted_overflow_before_zero" )
   {
      (void)MakeStridedLayout( std::array< GlobalIndex, 3 >{
         2, std::numeric_limits< GlobalIndex >::max(), 0 },
         Permutation< 3 >{ { 2, 1, 3 } } );
   }
   else if ( scenario == "negative_vector_extent" )
   {
      Real value{};
      (void)MakeVectorView( &value, -1 );
   }
   else if ( scenario == "invalid_vector_range" )
   {
      Real values[4]{};
      auto parent = MakeVectorView( values, 4 );
      (void)MakeVectorView( parent, 3, 2 );
   }
   else if ( scenario == "negative_offset" )
   {
      (void)MakeOffsetLayout( StridedLayout< 1 >{ 1 }, -1 );
   }
   else if ( scenario == "strided_offset_overflow" )
   {
      (void)StridedLayout< 2, DynamicExtents< 2 > >{
         std::array< GlobalIndex, 2 >{
            std::numeric_limits< GlobalIndex >::max(), 1 },
         DynamicExtents< 2 >{ 2, 2 } };
   }
   else if ( scenario == "indirected_negative" )
   {
      Real value{};
      (void)MakeIndirectedFIFOView(
         PointerContainer< Real >{ &value },
         HostDevicePointer< const int >{}, -1 );
   }
   else if (scenario == "oriented_reverse_overflow")
   {
      (void)OrientedLayout<1>{
         std::pair{-2, std::numeric_limits<GlobalIndex>::max()}};
   }
   else if ( scenario == "oriented_base_offset_overflow" )
   {
      (void)OrientedLayout< 2 >{
         std::pair{ -1, std::numeric_limits< size_t >::max() },
         std::pair{ -1, std::numeric_limits< size_t >::max() } };
   }
   else if ( scenario == "oriented_stride_overflow" )
   {
      (void)MakeOrientedLayout(
         std::array< size_t, 2 >{
            static_cast< size_t >( std::numeric_limits< int >::max() ) + 1, 2 },
         IdentityOrientation< 2 >{} );
   }
   else if ( scenario == "oriented_product_overflow" )
   {
      (void)MakeOrientedLayout(
         std::array< size_t, 2 >{ 2, std::numeric_limits< size_t >::max() },
         IdentityOrientation< 2 >{} );
   }
   else if ( scenario == "oriented_zero_dimension" )
   {
      (void)MakeOrientedLayout(
         std::array< size_t, 2 >{ 3, 0 }, Permutation< 2 >{ { -2, 1 } } );
   }
   else if ( scenario == "oriented_pair_zero_dimension" )
   {
      (void)OrientedLayout< 1 >{ std::pair{ -1, size_t{ 0 } } };
   }
   else
   {
      return 2;
   }
   return 0;
}
