// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#if defined(GENDIL_TEST_FORCE_DEBUG) && defined(NDEBUG)
#undef NDEBUG
#endif

#include "gendil/Utilities/checkedarithmetic.hpp"

#include <concepts>
#include <limits>
#include <string_view>

using namespace gendil;

namespace
{

template < typename T >
concept SupportsCheckedArithmetic =
   requires( const T value )
   {
      { CheckedAdd( value, value, "checked addition" ) }
         -> std::same_as< T >;
      { CheckedMultiply( value, value, "checked multiplication" ) }
         -> std::same_as< T >;
   };

static_assert( SupportsCheckedArithmetic< unsigned int > );
static_assert( SupportsCheckedArithmetic< size_t > );
static_assert( !SupportsCheckedArithmetic< int > );
static_assert( !SupportsCheckedArithmetic< bool > );

template < typename T, T Lhs, T Rhs >
concept ConstantCheckedAdd = requires
{
   typename std::integral_constant<
      T, CheckedAdd( Lhs, Rhs, "constant addition" ) >;
};

template < typename T, T Lhs, T Rhs >
concept ConstantCheckedMultiply = requires
{
   typename std::integral_constant<
      T, CheckedMultiply( Lhs, Rhs, "constant multiplication" ) >;
};

constexpr size_t maximum = std::numeric_limits< size_t >::max();
static_assert( CheckedAdd( size_t{2}, size_t{3}, "constant addition" ) == 5 );
static_assert( CheckedMultiply( size_t{6}, size_t{7}, "constant multiplication" ) == 42 );
static_assert( ConstantCheckedAdd< size_t, maximum, 0 > );
static_assert( ConstantCheckedMultiply< size_t, maximum, 0 > );
static_assert( ConstantCheckedMultiply< size_t, 0, maximum > );
static_assert( ConstantCheckedMultiply< size_t, maximum, 1 > );
#ifndef GENDIL_DEVICE_CODE
static_assert( ( GENDIL_VERIFY( true ), true ) );
static_assert( !ConstantCheckedAdd< size_t, maximum, 1 > );
static_assert( !ConstantCheckedMultiply< size_t, maximum, 2 > );
#endif
#if !defined(NDEBUG) && !defined(GENDIL_DEVICE_CODE)
static_assert( ( GENDIL_ASSERT( true ), true ) );
#endif

} // namespace

int main( const int argc, const char * const * argv )
{
   using Value = size_t;
   constexpr Value maximum = std::numeric_limits< Value >::max();

   if ( argc == 2 )
   {
      const std::string_view mode = argv[1];
      if ( mode == "add-overflow" )
      {
         (void)CheckedAdd(
            maximum,
            Value{1},
            "CheckedAdd overflow test." );
      }
      if ( mode == "multiply-overflow" )
      {
         (void)CheckedMultiply(
            maximum,
            Value{2},
            "CheckedMultiply overflow test." );
      }
      if ( mode == "assert-evaluation-failure" )
      {
         // Function arguments need not evaluate the predicate before the message.
         [[maybe_unused]] int predicates = 0;
         [[maybe_unused]] int messages = 0;
         GENDIL_ASSERT( ++predicates == 0,
            ++messages == 1 ?
               "Assertion predicate and message evaluated once." :
               "Assertion evaluation count is incorrect." );
      }
      if ( mode == "verify-evaluation-failure" )
      {
         int predicates = 0;
         int messages = 0;
         GENDIL_VERIFY( ++predicates == 0,
            ++messages == 1 ?
               "Verification predicate and message evaluated once." :
               "Verification evaluation count is incorrect." );
      }
      return 2;
   }

   const bool success =
      CheckedAdd( Value{2}, Value{3}, "unexpected addition overflow" ) == 5 &&
      CheckedAdd( maximum, Value{0}, "unexpected addition overflow" ) ==
         maximum &&
      CheckedMultiply(
         Value{6},
         Value{7},
         "unexpected multiplication overflow" ) == 42 &&
      CheckedMultiply(
         maximum,
         Value{0},
         "unexpected multiplication overflow" ) == 0 &&
      CheckedMultiply(
         Value{0},
         maximum,
         "unexpected multiplication overflow" ) == 0 &&
      CheckedMultiply(
         maximum,
         Value{1},
         "unexpected multiplication overflow" ) == maximum;
   return success ? 0 : 1;
}
