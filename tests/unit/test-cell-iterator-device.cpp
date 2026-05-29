// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-cell-iterator-device skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;

namespace
{
constexpr GlobalIndex num_cells = 10;
constexpr long long sentinel = -987654321;

struct TinyMesh
{
   GlobalIndex num_cells_;

   GlobalIndex GetNumberOfCells() const
   {
      return num_cells_;
   }

   GENDIL_HOST_DEVICE
   GlobalIndex GetCell( const GlobalIndex cell_index ) const
   {
      return cell_index;
   }
};

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

template < typename T >
struct DeviceBuffer
{
   HostDevicePointer< T > data;
   GlobalIndex size;

   DeviceBuffer( const GlobalIndex n, const T value )
      : size( n )
   {
      AllocateHostPointer( size, data );
      AllocateDevicePointer( size, data );
      Fill( value );
   }

   DeviceBuffer( const DeviceBuffer & ) = delete;
   DeviceBuffer & operator=( const DeviceBuffer & ) = delete;

   ~DeviceBuffer()
   {
      FreeDevicePointer( data );
      FreeHostPointer( data );
   }

   void Fill( const T value )
   {
      for ( GlobalIndex i = 0; i < size; ++i )
      {
         data.host_pointer[ i ] = value;
      }
      ToDevice( size, data );
   }

   void CopyToHost() const
   {
      ToHost( size, data );
   }
};

template < typename Config >
GENDIL_HOST_DEVICE
inline GlobalIndex CandidateCount()
{
   return Config::GetLaunchGeometry( num_cells ).grid_x *
      Config::batch_size;
}

template < typename Config >
bool RunCellIteratorDeviceCase( const char * name )
{
   const GlobalIndex candidate_count = CandidateCount< Config >();

   DeviceBuffer< long long > active_flag( candidate_count, sentinel );
   DeviceBuffer< long long > work_item_index( candidate_count, sentinel );
   DeviceBuffer< long long > guarded_output( candidate_count, sentinel );
   DeviceBuffer< long long > post_sync_status( candidate_count, sentinel );

   auto active_flag_data = active_flag.data;
   auto work_item_index_data = work_item_index.data;
   auto guarded_output_data = guarded_output.data;
   auto post_sync_status_data = post_sync_status.data;

   TinyMesh mesh{ num_cells };

   mesh::CellIterator< Config >(
      mesh,
      [=] GENDIL_DEVICE ( auto kernel ) mutable
      {
         const GlobalIndex cell_index = kernel.WorkItemIndex();
         const bool active = kernel.IsActive( num_cells );

         if ( kernel.GetLinearThreadIndex() == 0 )
         {
            active_flag_data[ cell_index ] = active ? 1 : 0;
            work_item_index_data[ cell_index ] =
               static_cast< long long >( cell_index );

            if ( active )
            {
               guarded_output_data[ cell_index ] =
                  static_cast< long long >( 1000 + cell_index );
            }
         }

         kernel.SyncWorkItem();

         if ( kernel.GetLinearThreadIndex() == 0 )
         {
            post_sync_status_data[ cell_index ] =
               static_cast< long long >( 2000 + cell_index );
         }
      } );

   GENDIL_DEVICE_SYNC;

   active_flag.CopyToHost();
   work_item_index.CopyToHost();
   guarded_output.CopyToHost();
   post_sync_status.CopyToHost();

   bool success = true;

   for ( GlobalIndex candidate = 0;
         candidate < candidate_count;
         ++candidate )
   {
      const bool expected_active = candidate < num_cells;

      success = Check(
         active_flag.data.host_pointer[ candidate ] ==
            ( expected_active ? 1 : 0 ),
         "CellIterator IsActive guard mismatch." ) && success;
      success = Check(
         work_item_index.data.host_pointer[ candidate ] ==
            static_cast< long long >( candidate ),
         "CellIterator WorkItemIndex mismatch." ) && success;
      success = Check(
         post_sync_status.data.host_pointer[ candidate ] ==
            static_cast< long long >( 2000 + candidate ),
         "Inactive CellIterator lane did not survive SyncWorkItem()." ) &&
         success;

      if ( expected_active )
      {
         success = Check(
            guarded_output.data.host_pointer[ candidate ] ==
               static_cast< long long >( 1000 + candidate ),
            "Active CellIterator lane did not write output." ) &&
            success;
      }
      else
      {
         success = Check(
            guarded_output.data.host_pointer[ candidate ] == sentinel,
            "Inactive CellIterator lane wrote guarded output." ) &&
            success;
      }
   }

   if ( !success )
   {
      std::cout << "Failure in " << name << ".\n";
   }
   return success;
}

bool TestWarpAlignedLayout()
{
   using Layout = ThreadBlockLayout< 4, 8 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static constexpr Integer BatchSize = 4;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   static_assert( Layout::GetNumberOfThreads() == 32 );

   return RunCellIteratorDeviceCase< Config >(
      "CellIterator ThreadBlockLayout<4, 8>" );
}

bool TestNonWarpAlignedLayout()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;
   static constexpr Integer BatchSize = 4;
   using Config =
      DeviceKernelConfiguration<
         Layout,
         MaxSharedDimensions,
         BatchSize >;

   static_assert( Layout::GetNumberOfThreads() == 15 );

   return RunCellIteratorDeviceCase< Config >(
      "CellIterator ThreadBlockLayout<3, 5>" );
}
} // namespace

int main()
{
   bool success = true;
   success = TestWarpAlignedLayout() && success;
   success = TestNonWarpAlignedLayout() && success;

   return success ? 0 : 1;
}

#endif
