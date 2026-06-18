// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

bool Check( const bool condition, const char * message )
{
   if ( !condition )
   {
      std::cout << message << '\n';
   }
   return condition;
}

bool Near( const Real a, const Real b )
{
   return std::abs( a - b ) < tolerance;
}

bool CheckRawTripletRangesAndFinite(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer )
{
   bool success = true;
   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      success = Check(
         buffer.rows[i] < buffer.num_rows,
         "Raw COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         buffer.cols[i] < buffer.num_cols,
         "Raw COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( buffer.values[i] ),
         "Raw COO emitted a non-finite value." ) && success;
   }
   return success;
}

template < typename FESpace >
bool CheckScalar1DRawCellSlotCoordinates(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer,
   const FESpace & fe_space )
{
   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );

   bool success = true;
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      for ( GlobalIndex local_col = 0; local_col < local_dofs; ++local_col )
      {
         const std::array< GlobalIndex, 1 > col_indices{ local_col };
         const GlobalIndex expected_col =
            ElementToGlobalDofIndex( fe_space, element, col_indices );

         for ( GlobalIndex local_row = 0; local_row < local_dofs; ++local_row )
         {
            const std::array< GlobalIndex, 1 > row_indices{ local_row };
            const GlobalIndex expected_row =
               ElementToGlobalDofIndex( fe_space, element, row_indices );
            const GlobalIndex raw_index =
               element * local_dofs * local_dofs +
               local_col * local_dofs +
               local_row;

            success = Check(
               raw_index < buffer.nnz_raw,
               "Raw COO slot coordinate test exceeded the triplet buffer." ) && success;
            success = Check(
               buffer.rows[raw_index] == expected_row,
               "Raw COO row slot does not match element-local test DoF order." ) && success;
            success = Check(
               buffer.cols[raw_index] == expected_col,
               "Raw COO column slot does not match element-local trial DoF order." ) && success;
         }
      }
   }

   return success;
}

template < typename FESpace >
bool CheckVectorRawCellSlotCoordinates(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer,
   const FESpace & fe_space )
{
   using Space = std::remove_cvref_t< FESpace >;
   using ShapeFunctions =
      typename Space::finite_element_type::shape_functions;
   static_assert(
      is_vector_shape_functions_v< ShapeFunctions >,
      "CheckVectorRawCellSlotCoordinates requires a vector FE space." );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   const GlobalIndex block_entries = local_dofs * local_dofs;

   bool success = true;
   for ( GlobalIndex element = 0; element < num_elements; ++element )
   {
      std::vector< GlobalIndex > expected_globals(
         static_cast< size_t >( local_dofs ),
         0 );
      std::vector< bool > seen(
         static_cast< size_t >( local_dofs ),
         false );

      ForEachLocalDof(
         fe_space,
         [&] ( const auto component, const auto & indices )
         {
            const LocalIndex local_id =
               FlattenLocalDof( fe_space, component, indices );
            success = Check(
               local_id < local_dofs,
               "Vector Raw COO local DoF flattening exceeded the block size." ) && success;

            if ( local_id < local_dofs )
            {
               expected_globals[static_cast< size_t >( local_id )] =
                  ElementToGlobalDofIndex(
                     fe_space,
                     component,
                     element,
                     indices );
               seen[static_cast< size_t >( local_id )] = true;
            }
         });

      for ( const bool was_seen : seen )
      {
         success = Check(
            was_seen,
            "Vector Raw COO component-major local DoF traversal skipped a slot." ) && success;
      }

      for ( GlobalIndex local_col = 0; local_col < local_dofs; ++local_col )
      {
         for ( GlobalIndex local_row = 0; local_row < local_dofs; ++local_row )
         {
            const GlobalIndex raw_index =
               element * block_entries +
               local_col * local_dofs +
               local_row;

            success = Check(
               raw_index < buffer.nnz_raw,
               "Vector Raw COO slot coordinate test exceeded the triplet buffer." ) && success;
            success = Check(
               buffer.rows[raw_index] ==
                  expected_globals[static_cast< size_t >( local_row )],
               "Vector Raw COO row slot does not match component-major test DoF order." ) && success;
            success = Check(
               buffer.cols[raw_index] ==
                  expected_globals[static_cast< size_t >( local_col )],
               "Vector Raw COO column slot does not match component-major trial DoF order." ) && success;
         }
      }
   }

   return success;
}

bool HasDuplicateCoordinate(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer )
{
   std::vector< std::pair< GlobalIndex, GlobalIndex > > coordinates;
   coordinates.reserve( static_cast< size_t >( buffer.nnz_raw ) );

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      coordinates.emplace_back( buffer.rows[i], buffer.cols[i] );
   }

   std::sort( coordinates.begin(), coordinates.end() );
   return std::adjacent_find( coordinates.begin(), coordinates.end() ) !=
      coordinates.end();
}

template < typename Backend >
bool CheckCanonicalCOOSortedUnique(
   const COOMatrix< Real, GlobalIndex, Backend > & matrix )
{
   bool success = true;
   for ( GlobalIndex i = 0; i < matrix.nnz; ++i )
   {
      success = Check(
         matrix.rows[i] < matrix.num_rows,
         "Canonical COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         matrix.cols[i] < matrix.num_cols,
         "Canonical COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( matrix.values[i] ),
         "Canonical COO emitted a non-finite value." ) && success;
   }

   for ( GlobalIndex i = 1; i < matrix.nnz; ++i )
   {
      const bool ordered =
         matrix.rows[i - 1] < matrix.rows[i] ||
         ( matrix.rows[i - 1] == matrix.rows[i] &&
           matrix.cols[i - 1] < matrix.cols[i] );

      success = Check(
         ordered,
         "Canonical COO entries are not strictly sorted and unique." ) && success;
   }

   return success;
}

template < typename ActualBackend, typename ExpectedBackend >
bool CheckCOOMatricesEqual(
   const COOMatrix< Real, GlobalIndex, ActualBackend > & actual,
   const COOMatrix< Real, GlobalIndex, ExpectedBackend > & expected,
   const char * message )
{
   bool success = true;
   success = Check(
      actual.num_rows == expected.num_rows,
      message ) && success;
   success = Check(
      actual.num_cols == expected.num_cols,
      message ) && success;
   success = Check(
      actual.nnz == expected.nnz,
      message ) && success;

   const GlobalIndex nnz =
      actual.nnz < expected.nnz ? actual.nnz : expected.nnz;
   for ( GlobalIndex i = 0; i < nnz; ++i )
   {
      success = Check(
         actual.rows[i] == expected.rows[i],
         message ) && success;
      success = Check(
         actual.cols[i] == expected.cols[i],
         message ) && success;
      success = Check(
         Near( actual.values[i], expected.values[i] ),
         message ) && success;
   }

   return success;
}

template <
   bool IncludeCellTerms,
   bool IncludeBoundaryFaceTerms,
   bool IncludeInteriorFaceTerms,
   typename FESpace >
GlobalIndex ExpectedRawCOONNZ( const FESpace & fe_space )
{
   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex block_entries = local_dofs * local_dofs;
   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   std::vector< bool > active_diagonal(
      static_cast< size_t >( num_elements ),
      false );
   GlobalIndex offdiag_blocks = 0;

   if constexpr ( IncludeCellTerms )
   {
      std::fill(
         active_diagonal.begin(),
         active_diagonal.end(),
         true );
   }

   if constexpr ( IncludeBoundaryFaceTerms )
   {
      for ( GlobalIndex element = 0; element < num_elements; ++element )
      {
         BoundaryFaceLoop(
            fe_space,
            element,
            [&] ( const auto & )
            {
               active_diagonal[static_cast< size_t >( element )] = true;
            });
      }
   }

   if constexpr ( IncludeInteriorFaceTerms )
   {
      for ( GlobalIndex element = 0; element < num_elements; ++element )
      {
         InteriorFaceLoop(
            fe_space,
            element,
            [&] ( const auto & )
            {
               active_diagonal[static_cast< size_t >( element )] = true;
               ++offdiag_blocks;
            });
      }
   }

   GlobalIndex block_count = offdiag_blocks;
   for ( const bool is_active : active_diagonal )
   {
      if ( is_active )
      {
         ++block_count;
      }
   }

   return block_count * block_entries;
}

bool CheckVectorNear(
   const Vector & actual,
   const Vector & expected,
   const char * message )
{
   const Real * actual_data = actual.ReadHostData();
   const Real * expected_data = expected.ReadHostData();

   bool success = true;
   for ( GlobalIndex i = 0; i < actual.Size(); ++i )
   {
      success = Check(
         Near( actual_data[i], expected_data[i] ),
         message ) && success;
   }
   return success;
}

void FillDeterministicInput( Vector & x )
{
   Real * data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      data[i] = 0.75 + 0.125 * static_cast< Real >( i );
   }
}

void ApplyTwoCellVectorH1P1MassReference(
   const Vector & x,
   Vector & y )
{
   // Two uniform 1D p1 elements with h = 0.5 give local scalar mass
   // (h / 6) * [[2, 1], [1, 2]]. Assembling the shared middle node produces
   // the component-major dense reference below for each independent component.
   const Real h = 0.5;
   const Real scale = h / 6.0;
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = 0.0;
   }

   for ( GlobalIndex component = 0; component < 2; ++component )
   {
      const GlobalIndex offset = 3 * component;
      y_data[offset + 0] =
         scale * ( 2.0 * x_data[offset + 0] + x_data[offset + 1] );
      y_data[offset + 1] =
         scale * ( x_data[offset + 0] +
                   4.0 * x_data[offset + 1] +
                   x_data[offset + 2] );
      y_data[offset + 2] =
         scale * ( x_data[offset + 1] + 2.0 * x_data[offset + 2] );
   }
}

bool TestRawCOOBufferAllocation()
{
   auto buffer = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 4, 5, 6 );

   bool success = true;
   success = Check( buffer.num_rows == 4, "Raw COO row count is wrong." ) && success;
   success = Check( buffer.num_cols == 5, "Raw COO column count is wrong." ) && success;
   success = Check( buffer.nnz_raw == 6, "Raw COO triplet count is wrong." ) && success;
   success = Check( buffer.rows.host_pointer != nullptr, "Raw COO rows were not allocated on host." ) && success;
   success = Check( buffer.cols.host_pointer != nullptr, "Raw COO cols were not allocated on host." ) && success;
   success = Check( buffer.values.host_pointer != nullptr, "Raw COO values were not allocated on host." ) && success;

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      success = Check( buffer.rows[i] == 0, "Raw COO rows were not initialized." ) && success;
      success = Check( buffer.cols[i] == 0, "Raw COO cols were not initialized." ) && success;
      success = Check( Near( buffer.values[i], 0.0 ), "Raw COO values were not initialized." ) && success;
   }

   FreeRawCOOTripletBuffer( buffer );
   return success;
}

bool TestRawCOOToCOOFinalization()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 2, 3, 6 );

   raw.rows[0] = 1;
   raw.cols[0] = 2;
   raw.values[0] = 3.0;
   raw.rows[1] = 0;
   raw.cols[1] = 0;
   raw.values[1] = 1.0;
   raw.rows[2] = 1;
   raw.cols[2] = 2;
   raw.values[2] = 4.0;
   raw.rows[3] = 0;
   raw.cols[3] = 1;
   raw.values[3] = 5.0;
   raw.rows[4] = 1;
   raw.cols[4] = 1;
   raw.values[4] = 2.0;
   raw.rows[5] = 1;
   raw.cols[5] = 1;
   raw.values[5] = -2.0;

   auto coo =
      FinalizeRawCOOToCOO(
         raw,
         HostSortReduceRawCOOPolicy{} );

   bool success = true;
   success = Check( coo.num_rows == 2, "Canonical COO row count is wrong." ) && success;
   success = Check( coo.num_cols == 3, "Canonical COO column count is wrong." ) && success;
   success = Check( coo.nnz == 4, "Canonical COO reduced nnz is wrong." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   success = Check( coo.rows[0] == 0 && coo.cols[0] == 0, "Canonical COO entry 0 coordinate is wrong." ) && success;
   success = Check( Near( coo.values[0], 1.0 ), "Canonical COO entry 0 value is wrong." ) && success;
   success = Check( coo.rows[1] == 0 && coo.cols[1] == 1, "Canonical COO entry 1 coordinate is wrong." ) && success;
   success = Check( Near( coo.values[1], 5.0 ), "Canonical COO entry 1 value is wrong." ) && success;
   success = Check( coo.rows[2] == 1 && coo.cols[2] == 1, "Canonical COO entry 2 coordinate is wrong." ) && success;
   success = Check( Near( coo.values[2], 0.0 ), "Canonical COO should retain exact reduced zeros." ) && success;
   success = Check( coo.rows[3] == 1 && coo.cols[3] == 2, "Canonical COO entry 3 coordinate is wrong." ) && success;
   success = Check( Near( coo.values[3], 7.0 ), "Canonical COO entry 3 value is wrong." ) && success;

   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw );
   return success;
}

bool TestScalarL2CellMassRawCOOAgainstBSR()
{
   const Integer n = 4;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 2;
   FiniteElementOrders< order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto rho =
      MakeCoefficient< "density", PhysicalCoordinate >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + x_phys[0] * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      coo.nnz == raw_coo.nnz_raw,
      "Scalar L2 cell-only RawCOO should not create duplicate triplets." ) && success;
   success = Check(
      direct_coo.nnz == coo.nnz,
      "Direct scalar L2 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.25 + 0.5 * static_cast< Real >( i );
   }

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_bsr( fe_space.GetNumberOfFiniteElementDofs() );
   direct_coo( x, y_coo );
   y_bsr = 0.0;
   bsr( x, y_bsr );

   const Real * coo_data = y_coo.ReadHostData();
   const Real * bsr_data = y_bsr.ReadHostData();
   for ( GlobalIndex i = 0; i < y_coo.Size(); ++i )
   {
      success = Check(
         Near( coo_data[i], bsr_data[i] ),
         "Raw COO action disagrees with BSR action." ) && success;
   }

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestVectorL2CellMassRawCOOAgainstSGBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   Cells< "mesh" > domain;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( domain, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Vector L2 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Vector L2 Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Vector L2 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 8 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 }
   };

   success = Check(
      raw_coo.nnz_raw == raw_coords.size(),
      "Vector p0 RawCOO test expected exactly two 2x2 cell blocks." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         raw_coo.rows[i] == raw_coords[i].first &&
            raw_coo.cols[i] == raw_coords[i].second,
         "Vector p0 RawCOO coordinates are not component-major." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      coo.nnz == raw_coo.nnz_raw,
      "Vector L2 cell-only RawCOO should not create duplicate triplets." ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector L2 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_sgbsr( fe_space.GetNumberOfFiniteElementDofs() );
   direct_coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector L2 COO action disagrees with SGBSR action." ) && success;

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestVectorBoundaryFaceMassCOOAgainstSGBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   BoundaryFacets< "mesh" > boundary_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( boundary_facets, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, true, false >( fe_space ),
      "Vector boundary face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector boundary COO assembly disagrees with explicit RawCOO finalization." ) && success;

   const GlobalIndex num_elements =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() );
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      const bool cross_component =
         ( raw_coo.rows[i] < num_elements ) !=
         ( raw_coo.cols[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( raw_coo.values[i], 0.0 ),
            "Vector boundary RawCOO should retain zero cross-component entries." ) && success;
      }
   }

   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      const bool cross_component =
         ( coo.rows[i] < num_elements ) !=
         ( coo.cols[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( coo.values[i], 0.0 ),
            "Vector boundary canonical COO should retain zero cross-component entries." ) && success;
      }
   }

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_sgbsr( fe_space.GetNumberOfFiniteElementDofs() );
   direct_coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector boundary COO action disagrees with SGBSR action." ) && success;

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestScalarH1CellMassRawCOOPreservesDuplicatesAgainstSGBSR()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto fe = MakeLobattoFiniteElement( orders );

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction restriction{ restriction_indices, 3 };
   auto fe_space = MakeFiniteElementSpace( mesh, fe, restriction );

   Cells< "mesh" > domain;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto rho =
      MakeCoefficient< "density", PhysicalCoordinate >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + 0.25 * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "H1 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "H1 Raw COO column dimension is wrong." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "H1 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = Check(
      HasDuplicateCoordinate( raw_coo ),
      "Scalar H1 RawCOO should preserve duplicate raw triplet coordinates." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      raw_coo.nnz_raw > coo.nnz,
      "Scalar H1 RawCOO should have fewer canonical entries after reduction." ) && success;
   success = Check(
      direct_coo.nnz == coo.nnz,
      "Direct scalar H1 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   for ( Integer vector_case = 0; vector_case < 2; ++vector_case )
   {
      Vector x( fe_space.GetNumberOfFiniteElementDofs() );
      Real * x_data = x.WriteHostData();
      for ( GlobalIndex i = 0; i < x.Size(); ++i )
      {
         x_data[i] =
            vector_case == 0
               ? 0.5 + 0.25 * static_cast< Real >( i )
               : 1.0 - 0.2 * static_cast< Real >( i );
      }

      Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
      Vector y_sgbsr( fe_space.GetNumberOfFiniteElementDofs() );
      direct_coo( x, y_coo );
      y_sgbsr = 0.0;
      sgbsr( x, y_sgbsr );

      const Real * coo_data = y_coo.ReadHostData();
      const Real * sgbsr_data = y_sgbsr.ReadHostData();
      for ( GlobalIndex i = 0; i < y_coo.Size(); ++i )
      {
         success = Check(
            Near( coo_data[i], sgbsr_data[i] ),
            "H1 Raw COO action disagrees with SGBSR action." ) && success;
      }
   }

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestVectorH1CellMassRawCOOAgainstDenseReference()
{
   const Integer n = 2;
   const Real h = 1.0 / static_cast< Real >( n );
   Cartesian1DMesh mesh( h, n );

   constexpr Integer order = 1;
   FiniteElementOrders< order > orders;
   auto scalar_fe = MakeLobattoFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );

   const std::vector< int > restriction_map{ 0, 1, 1, 2 };
   HostDevicePointer< const int > restriction_indices{};
   restriction_indices.host_pointer = restriction_map.data();
   H1Restriction scalar_restriction{ restriction_indices, 3 };
   auto restriction = MakeVectorH1Restriction< 2 >( scalar_restriction );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe, restriction );

   Cells< "mesh" > domain;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( domain, dot( u, v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   const GlobalIndex local_dofs =
      static_cast< GlobalIndex >( fe_space.finite_element.GetNumDofs() );
   const GlobalIndex expected_nnz =
      static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElements() ) *
      local_dofs *
      local_dofs;

   bool success = true;
   success = Check(
      raw_coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Vector H1 Raw COO row dimension is wrong." ) && success;
   success = Check(
      raw_coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Vector H1 Raw COO column dimension is wrong." ) && success;
   success = Check(
      fe_space.GetNumberOfFiniteElementDofs() == 6,
      "Vector H1 test expected two component-major copies of three scalar true DoFs." ) && success;
   success = Check(
      raw_coo.nnz_raw == expected_nnz,
      "Vector H1 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckVectorRawCellSlotCoordinates( raw_coo, fe_space ) && success;
   success = Check(
      HasDuplicateCoordinate( raw_coo ),
      "Vector H1 RawCOO should preserve duplicate coordinates from shared true DoFs." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = Check(
      raw_coo.nnz_raw > coo.nnz,
      "Vector H1 RawCOO should have fewer canonical entries after reduction." ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector H1 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_expected( fe_space.GetNumberOfFiniteElementDofs() );
   direct_coo( x, y_coo );
   ApplyTwoCellVectorH1P1MassReference( x, y_expected );

   success = CheckVectorNear(
      y_coo,
      y_expected,
      "Vector H1 COO action disagrees with the dense p1 mass reference." ) && success;

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestScalarP0InteriorJumpAnalyticRawCOO()
{
   // Analytic p0 assumptions for this mesh/form:
   // - each element basis value is 1 everywhere on the element;
   // - the single interior vertical face has physical measure 1.
   // Therefore integrate(interior_facets, jump(u) * jump(v)) on two cells is
   // [[1, -1], [-1, 1]].
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw == 4,
      "Analytic p0 interior jump RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 }
   };
   const std::array< Real, 4 > raw_values{ 1.0, -1.0, 1.0, -1.0 };

   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         raw_coo.rows[i] == raw_coords[i].first &&
            raw_coo.cols[i] == raw_coords[i].second,
         "Analytic p0 RawCOO did not preserve the directed interior face traversal." ) && success;
      success = Check(
         Near( raw_coo.values[i], raw_values[i] ),
         "Analytic p0 RawCOO value is wrong." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check( coo.nnz == 4, "Analytic p0 canonical COO nnz is wrong." ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 }
   };
   const std::array< Real, 4 > coo_values{ 1.0, -1.0, -1.0, 1.0 };

   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         coo.rows[i] == coo_coords[i].first &&
            coo.cols[i] == coo_coords[i].second,
         "Analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( coo.values[i], coo_values[i] ),
         "Analytic p0 canonical COO value is wrong." ) && success;
   }

   Vector x( 2 );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;

   Vector y( 2 );
   coo( x, y );
   const Real * y_data = y.ReadHostData();
   success = Check( Near( y_data[0], -3.0 ), "Analytic p0 COO action row 0 is wrong." ) && success;
   success = Check( Near( y_data[1], 3.0 ), "Analytic p0 COO action row 1 is wrong." ) && success;

   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestVectorP0InteriorJumpAnalyticRawCOO()
{
   // Analytic p0 assumptions for this mesh/form:
   // - each component basis value is 1 on its owning element;
   // - the single interior vertical face has physical measure 1;
   // - directed traversal emits current-row self and neighbor vector blocks.
   // Therefore dot(jump(u), jump(v)) gives the scalar jump matrix independently
   // for each component, with exact zero cross-component entries retained.
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto scalar_fe = MakeLegendreFiniteElement( orders );
   auto vector_fe =
      MakeVectorFiniteElement(
         scalar_fe,
         scalar_fe );
   auto fe_space = MakeFiniteElementSpace( mesh, vector_fe );

   InteriorFacets< "mesh" > interior_facets;
   VectorTrialSpace< "u" > u;
   VectorTestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, dot( jump( u ), jump( v ) ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOO(
         raw_coo,
         HostSortReduceRawCOOPolicy{} );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, false, true >( fe_space ),
      "Vector analytic p0 interior jump RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 16 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 2 }
   };
   const std::array< Real, 16 > raw_values{
      1.0, 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0, -1.0,
      1.0, 0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0, -1.0
   };

   success = Check(
      raw_coo.nnz_raw == raw_values.size(),
      "Vector analytic p0 RawCOO should have four compact 2x2 blocks." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         raw_coo.rows[i] == raw_coords[i].first &&
            raw_coo.cols[i] == raw_coords[i].second,
         "Vector analytic p0 RawCOO compact directed coordinate is wrong." ) && success;
      success = Check(
         Near( raw_coo.values[i], raw_values[i] ),
         "Vector analytic p0 RawCOO value is wrong." ) && success;
   }

   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCanonicalCOOSortedUnique( direct_coo ) && success;
   success = CheckCOOMatricesEqual(
      direct_coo,
      coo,
      "Direct vector analytic p0 COO assembly disagrees with explicit RawCOO finalization." ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 16 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 2, 3 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 2 },
      std::pair< GlobalIndex, GlobalIndex >{ 3, 3 }
   };
   const std::array< Real, 16 > coo_values{
      1.0, -1.0, 0.0, 0.0,
      -1.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, -1.0,
      0.0, 0.0, -1.0, 1.0
   };

   success = Check(
      coo.nnz == coo_values.size(),
      "Vector analytic p0 canonical COO nnz is wrong." ) && success;
   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         coo.rows[i] == coo_coords[i].first &&
            coo.cols[i] == coo_coords[i].second,
         "Vector analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( coo.values[i], coo_values[i] ),
         "Vector analytic p0 canonical COO value is wrong." ) && success;
   }

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   Real * x_data = x.WriteHostData();
   x_data[0] = 2.0;
   x_data[1] = 5.0;
   x_data[2] = 7.0;
   x_data[3] = 11.0;

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_sgbsr( fe_space.GetNumberOfFiniteElementDofs() );
   coo( x, y_coo );
   y_sgbsr = 0.0;
   sgbsr( x, y_sgbsr );

   Vector y_expected( fe_space.GetNumberOfFiniteElementDofs() );
   Real * expected_data = y_expected.WriteHostData();
   expected_data[0] = -3.0;
   expected_data[1] = 3.0;
   expected_data[2] = -4.0;
   expected_data[3] = 4.0;

   success = CheckVectorNear(
      y_coo,
      y_expected,
      "Vector analytic p0 COO action is wrong." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_sgbsr,
      "Vector analytic p0 COO action disagrees with SGBSR action." ) && success;

   FreeCOOMatrix( direct_coo );
   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestScalarBoundaryFaceMassCOOAgainstGenericAndBSR()
{
   Cartesian2DMesh mesh( 0.5, 2, 2 );

   constexpr Integer order = 1;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   BoundaryFacets< "mesh" > boundary_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( boundary_facets, u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< order + 2, order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, true, false >( fe_space ),
      "Boundary face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_bsr( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_generic( fe_space.GetNumberOfFiniteElementDofs() );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Boundary face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Boundary face COO action disagrees with matrix-free action." ) && success;

   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestScalarInteriorJumpCOOAgainstGenericAndBSR()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 1;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;
   auto weak_form = integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< order + 2, order + 2 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< false, false, true >( fe_space ),
      "Interior face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_bsr( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_generic( fe_space.GetNumberOfFiniteElementDofs() );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Interior face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Interior face COO action disagrees with matrix-free action." ) && success;

   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

bool TestScalarCombinedFaceCOOOffsetsAndAccumulation()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );

   constexpr Integer order = 0;
   FiniteElementOrders< order, order > orders;
   auto fe = MakeLegendreFiniteElement( orders );
   auto fe_space = MakeFiniteElementSpace( mesh, fe );

   Cells< "mesh" > cells;
   BoundaryFacets< "mesh" > boundary_facets;
   InteriorFacets< "mesh" > interior_facets;
   TrialSpace< "u" > u;
   TestSpace< "u" > v;

   auto weak_form =
      integrate( cells, u * v ) +
      integrate( boundary_facets, u * v ) +
      integrate( boundary_facets, 2.0 * u * v ) +
      integrate( interior_facets, jump( u ) * jump( v ) );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( fe_space ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto bsr =
      GenericAssembly< MatrixAssemblyType::BSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto generic_operator =
      MakeGenericOperator< KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );

   bool success = true;
   success = Check(
      raw_coo.nnz_raw ==
         ExpectedRawCOONNZ< true, true, true >( fe_space ),
      "Combined face RawCOO count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( raw_coo ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > raw_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 }
   };
   const std::array< Real, 4 > raw_values{ 11.0, -1.0, 11.0, -1.0 };

   success = Check(
      raw_coo.nnz_raw == raw_values.size(),
      "Combined p0 RawCOO should have compact diagonal/offdiag blocks only." ) && success;
   for ( GlobalIndex i = 0; i < raw_coo.nnz_raw; ++i )
   {
      success = Check(
         raw_coo.rows[i] == raw_coords[i].first &&
            raw_coo.cols[i] == raw_coords[i].second,
         "Combined p0 RawCOO compact block coordinate is wrong." ) && success;
      success = Check(
         Near( raw_coo.values[i], raw_values[i] ),
         "Combined p0 RawCOO did not accumulate self contributions into diagonal slots." ) && success;
   }

   const std::array< std::pair< GlobalIndex, GlobalIndex >, 4 > coo_coords{
      std::pair< GlobalIndex, GlobalIndex >{ 0, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 0, 1 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 0 },
      std::pair< GlobalIndex, GlobalIndex >{ 1, 1 }
   };
   const std::array< Real, 4 > coo_values{ 11.0, -1.0, -1.0, 11.0 };

   success = Check(
      coo.nnz == coo_values.size(),
      "Combined p0 canonical COO nnz is wrong." ) && success;
   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      success = Check(
         coo.rows[i] == coo_coords[i].first &&
            coo.cols[i] == coo_coords[i].second,
         "Combined p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( coo.values[i], coo_values[i] ),
         "Combined p0 canonical COO value is wrong." ) && success;
   }

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   FillDeterministicInput( x );

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_bsr( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_generic( fe_space.GetNumberOfFiniteElementDofs() );
   coo( x, y_coo );
   y_bsr = 0.0;
   y_generic = 0.0;
   bsr( x, y_bsr );
   generic_operator( x, y_generic );

   success = CheckVectorNear(
      y_coo,
      y_bsr,
      "Combined face COO action disagrees with BSR action." ) && success;
   success = CheckVectorNear(
      y_coo,
      y_generic,
      "Combined face COO action disagrees with matrix-free action." ) && success;

   FreeCOOMatrix( coo );
   FreeRawCOOTripletBuffer( raw_coo );
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRawCOOBufferAllocation() && success;
   success = TestRawCOOToCOOFinalization() && success;
   success = TestScalarL2CellMassRawCOOAgainstBSR() && success;
   success = TestVectorL2CellMassRawCOOAgainstSGBSR() && success;
   success = TestVectorBoundaryFaceMassCOOAgainstSGBSR() && success;
   success = TestScalarH1CellMassRawCOOPreservesDuplicatesAgainstSGBSR() && success;
   success = TestVectorH1CellMassRawCOOAgainstDenseReference() && success;
   success = TestScalarP0InteriorJumpAnalyticRawCOO() && success;
   success = TestVectorP0InteriorJumpAnalyticRawCOO() && success;
   success = TestScalarBoundaryFaceMassCOOAgainstGenericAndBSR() && success;
   success = TestScalarInteriorJumpCOOAgainstGenericAndBSR() && success;
   success = TestScalarCombinedFaceCOOOffsetsAndAccumulation() && success;

   return success ? 0 : 1;
}
