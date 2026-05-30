// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include "batched-cell-test-helpers.hpp"

#include <cmath>
#include <iostream>

#if !defined(GENDIL_USE_DEVICE)

int main()
{
   std::cout
      << "test-batched-grad-grad-write-coverage skipped because "
      << "GENDIL_USE_DEVICE is not enabled.\n";
   return 0;
}

#else

using namespace gendil;
using namespace gendil::test;

namespace
{
constexpr Integer order = 3;
constexpr Integer dof_count = order + 1;
constexpr Real value_sentinel = -777777.25;
constexpr Real tolerance = 1.0e-12;

GENDIL_HOST_DEVICE
Real ExpectedWriteValue( const GlobalIndex element_index, const int writer_id )
{
   return 1000.0 +
      17.0 * static_cast< Real >( element_index ) +
      1.0e-4 * static_cast< Real >( writer_id );
}

GENDIL_HOST_DEVICE
int CurrentWriterId()
{
#ifdef GENDIL_DEVICE_CODE
   return static_cast< int >( threadIdx.x + 100000 * threadIdx.y );
#else
   return 0;
#endif
}

struct CountingAssignmentProxy
{
   Real * values;
   int * counts;
   int * first_writers;
   int * duplicates;
   GlobalIndex index;
   int writer_id;

   GENDIL_HOST_DEVICE
   const CountingAssignmentProxy & operator=( const Real value ) const
   {
      const int old_count = gendil::AtomicAdd( counts[ index ], 1 );
      if ( old_count == 0 )
      {
         first_writers[ index ] = writer_id;
      }
      else
      {
         duplicates[ index ] = 1;
      }
      values[ index ] = value;
      return *this;
   }
};

struct CountingL2OutputView
{
   Real * values;
   int * counts;
   int * first_writers;
   int * duplicates;

   GENDIL_HOST_DEVICE
   CountingAssignmentProxy operator()(
      const GlobalIndex dof,
      const GlobalIndex element_index ) const
   {
      const GlobalIndex index = element_index * dof_count + dof;
      return {
         values,
         counts,
         first_writers,
         duplicates,
         index,
         CurrentWriterId()
      };
   }
};

template <
   typename Config,
   typename FiniteElementSpace >
void RunWriteCoverageKernel(
   const FiniteElementSpace & fe_space,
   const GlobalIndex num_cells,
   CountingL2OutputView output )
{
   Config::BlockLoop(
      num_cells,
      [=] GENDIL_HOST_DEVICE ( const Config & kernel ) mutable
      {
         if ( !kernel.IsActive( num_cells ) )
         {
            return;
         }

         constexpr size_t required_shared_mem = 1;
         GENDIL_SHARED Real _shared_mem[
            KernelContext<
               Config,
               required_shared_mem >::shared_memory_block_size ];
         KernelContext< Config, required_shared_mem >
            kernel_conf( _shared_mem, kernel );

         using DofShape = orders_to_num_dofs<
            typename FiniteElementSpace::finite_element_type::
               shape_functions::orders >;
         using RegisterShape = subsequence_t<
            DofShape,
            typename Config::template register_dimensions<
               DofShape::size() > >;
         auto local = MakeStaticFIFOView< Real >( RegisterShape{} );
         const GlobalIndex element_index = kernel.WorkItemIndex();
         const int writer_id = CurrentWriterId();
         UnitLoop< RegisterShape >(
            [&] ( auto... k )
            {
               local( k... ) = ExpectedWriteValue( element_index, writer_id );
            } );

         WriteDofs( kernel_conf, fe_space, element_index, local, output );
      } );
   GENDIL_DEVICE_SYNC;
}

bool CheckCoverage(
   const char * label,
   const GlobalIndex num_cells,
   const GlobalIndex candidate_cells,
   DeviceBuffer< Real > & values,
   DeviceBuffer< int > & counts,
   DeviceBuffer< int > & first_writers,
   DeviceBuffer< int > & duplicates )
{
   values.CopyToHost();
   counts.CopyToHost();
   first_writers.CopyToHost();
   duplicates.CopyToHost();

   bool success = true;
   Integer printed = 0;
   Integer missing = 0;
   Integer duplicate = 0;
   Integer unexpected_inactive = 0;
   Integer wrong_value = 0;

   for ( GlobalIndex element = 0; element < candidate_cells; ++element )
   {
      for ( GlobalIndex dof = 0; dof < dof_count; ++dof )
      {
         const GlobalIndex index = element * dof_count + dof;
         const bool active = element < num_cells;
         const int count = counts.data.host_pointer[ index ];
         const int first_writer = first_writers.data.host_pointer[ index ];
         const int duplicate_marker = duplicates.data.host_pointer[ index ];
         const Real value = values.data.host_pointer[ index ];

         if ( active )
         {
            if ( count == 0 || std::abs( value - value_sentinel ) < tolerance )
            {
               ++missing;
               success = false;
               if ( printed < 16 )
               {
                  std::cout << "FAILED: expected write missing in "
                            << label << " at element " << element
                            << ", dof " << dof << ". count=" << count
                            << ", value=" << value << ".\n";
                  ++printed;
               }
            }
            else if ( count > 1 || duplicate_marker != 0 )
            {
               ++duplicate;
               success = false;
               if ( printed < 16 )
               {
                  std::cout << "FAILED: duplicate write in " << label
                            << " at element " << element << ", dof "
                            << dof << ". count=" << count
                            << ", first_writer=" << first_writer
                            << ", value=" << value << ".\n";
                  ++printed;
               }
            }
            else if (
               std::abs(
                  value - ExpectedWriteValue( element, first_writer ) ) >
               tolerance )
            {
               ++wrong_value;
               success = false;
               if ( printed < 16 )
               {
                  std::cout << "FAILED: correct write count but wrong value in "
                            << label << " at element " << element
                            << ", dof " << dof << ". observed=" << value
                            << ", expected="
                            << ExpectedWriteValue( element, first_writer )
                            << ", writer=" << first_writer << ".\n";
                  ++printed;
               }
            }
         }
         else if (
            count != 0 ||
            first_writer != -1 ||
            duplicate_marker != 0 ||
            std::abs( value - value_sentinel ) > tolerance )
         {
            ++unexpected_inactive;
            success = false;
            if ( printed < 16 )
            {
               std::cout << "FAILED: unexpected inactive write in "
                         << label << " at candidate element " << element
                         << ", dof " << dof << ". count=" << count
                         << ", first_writer=" << first_writer
                         << ", duplicate=" << duplicate_marker
                         << ", value=" << value << ".\n";
               ++printed;
            }
         }
      }
   }

   std::cout << label << " coverage summary: missing=" << missing
             << ", duplicate=" << duplicate
             << ", unexpected_inactive=" << unexpected_inactive
             << ", wrong_value=" << wrong_value << ".\n";
   return success;
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize >
bool RunWriteCoverageCase( const char * label, const GlobalIndex num_cells )
{
   if ( !LaunchConfigurationFits< Layout, BatchSize >( label ) )
   {
      return true;
   }

   using Config =
      DeviceKernelConfiguration< Layout, MaxSharedDimensions, BatchSize >;

   const GlobalIndex candidate_cells =
      ( ( num_cells + BatchSize - 1 ) / BatchSize ) * BatchSize;
   const GlobalIndex output_size = candidate_cells * dof_count;

   DeviceBuffer< Real > values( output_size, value_sentinel );
   DeviceBuffer< int > counts( output_size, 0 );
   DeviceBuffer< int > first_writers( output_size, -1 );
   DeviceBuffer< int > duplicates( output_size, 0 );

   const Real h = 1.0 / static_cast< Real >( num_cells );
   Cartesian1DMesh mesh( h, num_cells );

   FiniteElementOrders< order > orders;
   auto finite_element = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, finite_element );

   CountingL2OutputView output{
      values.data.device_pointer,
      counts.data.device_pointer,
      first_writers.data.device_pointer,
      duplicates.data.device_pointer
   };

   std::cout << label << ", num_cells = " << num_cells << '\n';
   RunWriteCoverageKernel< Config >( fe_space, num_cells, output );
   return CheckCoverage(
      label,
      num_cells,
      candidate_cells,
      values,
      counts,
      first_writers,
      duplicates );
}

template <
   typename Layout,
   Integer MaxSharedDimensions,
   Integer BatchSize >
bool RunWriteCoverageCases( const char * label )
{
   bool success = true;
   success =
      RunWriteCoverageCase< Layout, MaxSharedDimensions, BatchSize >(
         label,
         64 ) && success;
   success =
      RunWriteCoverageCase< Layout, MaxSharedDimensions, BatchSize >(
         label,
         65 ) && success;
   return success;
}

bool TestThreadedReferenceLayout()
{
   using Layout = ThreadBlockLayout< 4 >;
   static constexpr Integer MaxSharedDimensions = 1;

   bool success = true;
   success =
      RunWriteCoverageCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<4>, BatchSize=4" ) && success;
   success =
      RunWriteCoverageCases< Layout, MaxSharedDimensions, device_warp_size >(
         "ThreadBlockLayout<4>, BatchSize=device_warp_size" ) && success;
   return success;
}

bool TestIrregularDiagnosticLayout()
{
   using Layout = ThreadBlockLayout< 3, 5 >;
   static constexpr Integer MaxSharedDimensions = 2;

   std::cout
      << "Hypothesis under test: for a 1D path, logical dimension 0 "
      << "maps to the 1D DOF loop while logical dimension 1 is idle or "
      << "gating-only. This test verifies whether that extra dimension "
      << "causes missing, duplicate, or unexpected writers.\n";

   bool success = true;
   success =
      RunWriteCoverageCases< Layout, MaxSharedDimensions, 4 >(
         "ThreadBlockLayout<3,5>, BatchSize=4" ) && success;
   success =
      RunWriteCoverageCases< Layout, MaxSharedDimensions, device_warp_size >(
         "ThreadBlockLayout<3,5>, BatchSize=device_warp_size" ) && success;
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestThreadedReferenceLayout() && success;
   success = TestIrregularDiagnosticLayout() && success;
   return success ? 0 : 1;
}

#endif
