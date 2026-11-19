// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/Interfaces/Caliper/caliperinterface.hpp>

#include <iostream>

#ifdef GENDIL_USE_CALIPER
#include <algorithm>
#include <array>
#include <initializer_list>
#include <string>
#include <vector>
#endif

namespace caliper_interface_test
{

#ifdef GENDIL_USE_CALIPER

int ReadRegion( void * data, cali_id_t attribute, cali_variant_t value )
{
   if ( cali_variant_get_type( value ) == CALI_TYPE_STRING &&
        ( cali_attribute_properties( attribute ) & CALI_ATTR_NESTED ) )
   {
      auto & regions = *static_cast< std::vector< std::string > * >( data );
      regions.emplace_back(
         static_cast< const char * >( cali_variant_get_data( &value ) ),
         cali_variant_get_size( value ) );
   }
   return 1;
}

bool CheckSnapshot(
   cali_id_t channel,
   std::initializer_list< const char * > expected,
   const char * stage )
{
   std::array< unsigned char, 4096 > buffer{};
   const auto size = cali_channel_pull_snapshot(
      channel, CALI_SCOPE_PROCESS | CALI_SCOPE_THREAD,
      buffer.size(), buffer.data() );
   if ( size == 0 || size > buffer.size() )
   {
      std::cerr << stage << ": Caliper snapshot is missing or truncated.\n";
      return false;
   }

   std::vector< std::string > actual;
   size_t bytes_read = 0;
   cali_unpack_snapshot( buffer.data(), &bytes_read, ReadRegion, &actual );
   std::vector< std::string > reference( expected.begin(), expected.end() );
   std::sort( actual.begin(), actual.end() );
   std::sort( reference.begin(), reference.end() );
   if ( bytes_read != size || actual != reference )
   {
      std::cerr << stage << ": unexpected Caliper region stack:";
      for ( const auto & region : actual )
      {
         std::cerr << ' ' << region;
      }
      std::cerr << '\n';
      return false;
   }
   return true;
}

bool CheckFunctionRegion( cali_id_t channel, bool early_return )
{
   GENDIL_BENCHMARK_FUNCTION;
   bool success = CheckSnapshot(
      channel, { "outer", __func__ }, "function scope" );
   if ( early_return )
   {
      return success;
   }

   GENDIL_BENCHMARK_REGION_BEGIN( "inner" );
   success = CheckSnapshot(
      channel, { "outer", __func__, "inner" }, "nested region" ) && success;
   GENDIL_BENCHMARK_REGION_END( "inner" );
   return CheckSnapshot(
      channel, { "outer", __func__ }, "nested region exit" ) && success;
}

bool Run()
{
   // Keep user profiling settings and output services out of this unit test.
   cali_config_allow_read_env( 0 );
   cali_config_set( "CALI_CONFIG", "" );
   cali_config_set( "CALI_SERVICES_ENABLE", "" );
   const char * configuration[][2] = {
      { "CALI_SERVICES_ENABLE", "" },
      { "CALI_CHANNEL_FLUSH_ON_EXIT", "false" },
      { nullptr, nullptr }
   };
   const auto config = cali_create_configset( configuration );
   const auto channel = cali_create_channel( "gendil-caliper-test", 0, config );
   cali_delete_configset( config );
   if ( channel == CALI_INV_ID )
   {
      std::cerr << "Could not create the Caliper test channel.\n";
      return false;
   }

   bool success = CheckSnapshot( channel, {}, "before annotations" );
   GENDIL_BENCHMARK_REGION_BEGIN( "outer" );
   success = CheckSnapshot( channel, { "outer" }, "outer region" ) && success;
   success = CheckFunctionRegion( channel, false ) && success;
   success = CheckSnapshot( channel, { "outer" }, "function exit" ) && success;
   success = CheckFunctionRegion( channel, true ) && success;
   success = CheckSnapshot( channel, { "outer" }, "early function exit" ) && success;
   GENDIL_BENCHMARK_REGION_END( "outer" );
   success = CheckSnapshot( channel, {}, "after annotations" ) && success;

   cali_delete_channel( channel );
   return success;
}

#else

bool Run()
{
   GENDIL_BENCHMARK_FUNCTION;
   int evaluations = 0;
   GENDIL_BENCHMARK_REGION_BEGIN( ( ++evaluations, "disabled" ) );
   GENDIL_BENCHMARK_REGION_END( ( ++evaluations, "disabled" ) );
   if ( evaluations != 0 )
   {
      std::cerr << "Disabled Caliper annotations evaluated their arguments.\n";
      return false;
   }
   return true;
}

#endif

} // namespace caliper_interface_test

int main()
{
   return caliper_interface_test::Run() ? 0 : 1;
}
