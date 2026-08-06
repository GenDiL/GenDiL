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
         ReadHost( buffer.rows )[i] < buffer.num_rows,
         "Raw COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         ReadHost( buffer.cols )[i] < buffer.num_cols,
         "Raw COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( ReadHost( buffer.values )[i] ),
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
            GlobalDofIndex( fe_space, element, col_indices );

         for ( GlobalIndex local_row = 0; local_row < local_dofs; ++local_row )
         {
            const std::array< GlobalIndex, 1 > row_indices{ local_row };
            const GlobalIndex expected_row =
               GlobalDofIndex( fe_space, element, row_indices );
            const GlobalIndex raw_index =
               element * local_dofs * local_dofs +
               local_col * local_dofs +
               local_row;

            success = Check(
               raw_index < buffer.nnz_raw,
               "Raw COO slot coordinate test exceeded the triplet buffer." ) && success;
            success = Check(
               ReadHost( buffer.rows )[raw_index] == expected_row,
               "Raw COO row slot does not match element-local test DoF order." ) && success;
            success = Check(
               ReadHost( buffer.cols )[raw_index] == expected_col,
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
            const GlobalIndex local_id =
               FlattenLocalDof( fe_space, component, indices );
            success = Check(
               local_id < local_dofs,
               "Vector Raw COO local DoF flattening exceeded the block size." ) && success;

            if ( local_id < local_dofs )
            {
               expected_globals[static_cast< size_t >( local_id )] =
                  GlobalDofIndex(
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
               ReadHost( buffer.rows )[raw_index] ==
                  expected_globals[static_cast< size_t >( local_row )],
               "Vector Raw COO row slot does not match component-major test DoF order." ) && success;
            success = Check(
               ReadHost( buffer.cols )[raw_index] ==
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
      coordinates.emplace_back( ReadHost( buffer.rows )[i], ReadHost( buffer.cols )[i] );
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
         ReadHost( matrix.rows )[i] < matrix.num_rows,
         "Canonical COO emitted a row outside the matrix dimensions." ) && success;
      success = Check(
         ReadHost( matrix.cols )[i] < matrix.num_cols,
         "Canonical COO emitted a column outside the matrix dimensions." ) && success;
      success = Check(
         std::isfinite( ReadHost( matrix.values )[i] ),
         "Canonical COO emitted a non-finite value." ) && success;
   }

   for ( GlobalIndex i = 1; i < matrix.nnz; ++i )
   {
      const bool ordered =
         ReadHost( matrix.rows )[i - 1] < ReadHost( matrix.rows )[i] ||
         ( ReadHost( matrix.rows )[i - 1] == ReadHost( matrix.rows )[i] &&
           ReadHost( matrix.cols )[i - 1] < ReadHost( matrix.cols )[i] );

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
         ReadHost( actual.rows )[i] == ReadHost( expected.rows )[i],
         message ) && success;
      success = Check(
         ReadHost( actual.cols )[i] == ReadHost( expected.cols )[i],
         message ) && success;
      success = Check(
         Near( ReadHost( actual.values )[i], ReadHost( expected.values )[i] ),
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
   success = Check( buffer.rows.data.host_pointer != nullptr, "Raw COO rows were not allocated on host." ) && success;
   success = Check( buffer.cols.data.host_pointer != nullptr, "Raw COO cols were not allocated on host." ) && success;
   success = Check( buffer.values.data.host_pointer != nullptr, "Raw COO values were not allocated on host." ) && success;

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      success = Check( ReadHost( buffer.rows )[i] == 0, "Raw COO rows were not initialized." ) && success;
      success = Check( ReadHost( buffer.cols )[i] == 0, "Raw COO cols were not initialized." ) && success;
      success = Check( Near( ReadHost( buffer.values )[i], 0.0 ), "Raw COO values were not initialized." ) && success;
   }

   return success;
}

bool TestRawCOOToCOOFinalization()
{
   auto raw = MakeRawCOOTripletBuffer< Real, GlobalIndex >( 2, 3, 6 );
   auto raw_data = GetHostReadWriteView( raw );

   raw_data.rows[0] = 1;
   raw_data.cols[0] = 2;
   raw_data.values[0] = 3.0;
   raw_data.rows[1] = 0;
   raw_data.cols[1] = 0;
   raw_data.values[1] = 1.0;
   raw_data.rows[2] = 1;
   raw_data.cols[2] = 2;
   raw_data.values[2] = 4.0;
   raw_data.rows[3] = 0;
   raw_data.cols[3] = 1;
   raw_data.values[3] = 5.0;
   raw_data.rows[4] = 1;
   raw_data.cols[4] = 1;
   raw_data.values[4] = 2.0;
   raw_data.rows[5] = 1;
   raw_data.cols[5] = 1;
   raw_data.values[5] = -2.0;

   auto coo =
      FinalizeRawCOOToCOOHost( raw );

   bool success = true;
   success = Check( coo.num_rows == 2, "Canonical COO row count is wrong." ) && success;
   success = Check( coo.num_cols == 3, "Canonical COO column count is wrong." ) && success;
   success = Check( coo.nnz == 4, "Canonical COO reduced nnz is wrong." ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;

   success = Check( ReadHost( coo.rows )[0] == 0 && ReadHost( coo.cols )[0] == 0, "Canonical COO entry 0 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[0], 1.0 ), "Canonical COO entry 0 value is wrong." ) && success;
   success = Check( ReadHost( coo.rows )[1] == 0 && ReadHost( coo.cols )[1] == 1, "Canonical COO entry 1 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[1], 5.0 ), "Canonical COO entry 1 value is wrong." ) && success;
   success = Check( ReadHost( coo.rows )[2] == 1 && ReadHost( coo.cols )[2] == 1, "Canonical COO entry 2 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[2], 0.0 ), "Canonical COO should retain exact reduced zeros." ) && success;
   success = Check( ReadHost( coo.rows )[3] == 1 && ReadHost( coo.cols )[3] == 2, "Canonical COO entry 3 coordinate is wrong." ) && success;
   success = Check( Near( ReadHost( coo.values )[3], 7.0 ), "Canonical COO entry 3 value is wrong." ) && success;

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
      MakeCoefficient< "density", PhysicalCoordinates >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + x_phys[0] * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
      FinalizeRawCOOToCOOHost( raw_coo );
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
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
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
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
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
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
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
         ( ReadHost( raw_coo.rows )[i] < num_elements ) !=
         ( ReadHost( raw_coo.cols )[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( ReadHost( raw_coo.values )[i], 0.0 ),
            "Vector boundary RawCOO should retain zero cross-component entries." ) && success;
      }
   }

   for ( GlobalIndex i = 0; i < coo.nnz; ++i )
   {
      const bool cross_component =
         ( ReadHost( coo.rows )[i] < num_elements ) !=
         ( ReadHost( coo.cols )[i] < num_elements );
      if ( cross_component )
      {
         success = Check(
            Near( ReadHost( coo.values )[i], 0.0 ),
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
      MakeCoefficient< "density", PhysicalCoordinates >(
         [] GENDIL_HOST_DEVICE ( const auto & x_phys )
         {
            return 1.0 + 0.25 * x_phys[0];
         } );
   auto weak_form = integrate( domain, rho * u * v );
   auto wf_context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( fe_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
      FinalizeRawCOOToCOOHost( raw_coo );
   auto direct_coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto sgbsr =
      GenericAssembly< MatrixAssemblyType::SGBSR, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule,
         HostBSRBackend<>{} );

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
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
      FinalizeRawCOOToCOOHost( raw_coo );
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
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );

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
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Analytic p0 RawCOO did not preserve the directed interior face traversal." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
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
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
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
         MakeIntegrationDomain< "mesh" >( mesh ) );

   IntegrationRuleNumPoints< 1, 1 > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto raw_coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         weak_form,
         wf_context,
         integration_rule );
   auto coo =
      FinalizeRawCOOToCOOHost( raw_coo );
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
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Vector analytic p0 RawCOO compact directed coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
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
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Vector analytic p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
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
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
         MakeIntegrationDomain< "mesh" >( mesh ) );

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
         ReadHost( raw_coo.rows )[i] == raw_coords[i].first &&
            ReadHost( raw_coo.cols )[i] == raw_coords[i].second,
         "Combined p0 RawCOO compact block coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( raw_coo.values )[i], raw_values[i] ),
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
         ReadHost( coo.rows )[i] == coo_coords[i].first &&
            ReadHost( coo.cols )[i] == coo_coords[i].second,
         "Combined p0 canonical COO coordinate is wrong." ) && success;
      success = Check(
         Near( ReadHost( coo.values )[i], coo_values[i] ),
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

   return success;
}

template < typename Operator >
Vector ApplyRectangularOperator(
   const Operator & op,
   const Vector & x,
   const GlobalIndex output_size )
{
   Vector y( output_size );
   y = 0.0;
   op( x, y );
   return y;
}

template < typename Operator >
bool CheckRectangularAction(
   const Operator & op,
   const Vector & x,
   const Vector & expected,
   const char * message )
{
   const auto actual =
      ApplyRectangularOperator( op, x, expected.Size() );
   return CheckVectorNear( actual, expected, message );
}

template < size_t NumNodes >
Real IndependentLagrangeValue(
   const std::array< Real, NumNodes > & nodes,
   const size_t basis,
   const Real point )
{
   Real value = 1.0;
   for ( size_t other = 0; other < NumNodes; ++other )
   {
      if ( other != basis )
      {
         value *=
            ( point - nodes[other] ) /
            ( nodes[basis] - nodes[other] );
      }
   }
   return value;
}

std::array< Real, 36 > IndependentP2TestP1TrialCellMassBlock()
{
   // Test-owned values on [0, 1]. These deliberately do not use GenDiL's
   // point, weight, interpolation, qdata, or DofToQuad implementations.
   constexpr std::array< Real, 2 > trial_nodes{
      0.21132486540518711775,
      0.78867513459481288225 };
   constexpr std::array< Real, 3 > test_nodes{
      0.11270166537925831148,
      0.5,
      0.88729833462074168852 };
   constexpr std::array< Real, 5 > points{
      0.0469100770306680,
      0.230765344947158,
      0.500000000000000,
      0.769234655052842,
      0.953089922969332 };
   constexpr std::array< Real, 5 > weights{
      0.118463442528095,
      0.239314335249683,
      0.284444444444444,
      0.239314335249683,
      0.118463442528095 };

   std::array< Real, 36 > block{};
   for ( size_t trial_y = 0; trial_y < trial_nodes.size(); ++trial_y )
   {
      for ( size_t trial_x = 0; trial_x < trial_nodes.size(); ++trial_x )
      {
         const size_t trial = trial_x + 2 * trial_y;
         for ( size_t test_y = 0; test_y < test_nodes.size(); ++test_y )
         {
            for ( size_t test_x = 0; test_x < test_nodes.size(); ++test_x )
            {
               const size_t test = test_x + 3 * test_y;
               Real value = 0.0;
               for ( size_t qy = 0; qy < points.size(); ++qy )
               {
                  for ( size_t qx = 0; qx < points.size(); ++qx )
                  {
                     value +=
                        weights[qx] * weights[qy] *
                        IndependentLagrangeValue(
                           test_nodes,
                           test_x,
                           points[qx] ) *
                        IndependentLagrangeValue(
                           test_nodes,
                           test_y,
                           points[qy] ) *
                        IndependentLagrangeValue(
                           trial_nodes,
                           trial_x,
                           points[qx] ) *
                        IndependentLagrangeValue(
                           trial_nodes,
                           trial_y,
                           points[qy] );
                  }
               }
               // Cartesian2DMesh(1.0, ...) has unit affine-cell Jacobian.
               block[trial * 9 + test] = value;
            }
         }
      }
   }
   return block;
}

bool TestRectangularP1TrialP2TestCellAssembly()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form = integrate( Cells< "mesh" >{}, u * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      raw.num_rows == 18 && raw.num_cols == 8,
      "Rectangular cell RawCOO dimensions are not 18 x 8." ) && success;
   success = Check(
      raw.nnz_raw == 72,
      "Rectangular cell RawCOO must contain two 9 x 4 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct rectangular COO disagrees with finalized RawCOO." ) && success;
   success = Check(
      coo.num_rows == 18 && coo.num_cols == 8 &&
      csr.num_rows == 18 && csr.num_cols == 8 &&
      csc.num_rows == 18 && csc.num_cols == 8,
      "A derived cell format lost the 18 x 8 dimensions." ) && success;

   const auto independent_block =
      IndependentP2TestP1TrialCellMassBlock();
   const auto raw_data = GetHostReadView( raw );
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      for ( GlobalIndex trial = 0; trial < 4; ++trial )
      {
         const std::array< GlobalIndex, 2 > trial_indices{
            trial % 2,
            trial / 2 };
         const GlobalIndex expected_col =
            GlobalDofIndex( trial_space, element, trial_indices );
         for ( GlobalIndex test = 0; test < 9; ++test )
         {
            const std::array< GlobalIndex, 2 > test_indices{
               test % 3,
               test / 3 };
            const GlobalIndex expected_row =
               GlobalDofIndex( test_space, element, test_indices );
            const GlobalIndex entry =
               element * 36 + trial * 9 + test;
            success = Check(
               raw_data.rows[entry] == expected_row,
               "Rectangular cell RawCOO row is not derived from the p2 test space." ) && success;
            success = Check(
               raw_data.cols[entry] == expected_col,
               "Rectangular cell RawCOO column is not derived from the p1 trial space." ) && success;
         }
      }
   }
   for ( GlobalIndex entry = 0; entry < 36; ++entry )
   {
      success = Check(
         Near( raw_data.values[entry], independent_block[entry] ),
         "Rectangular 9 x 4 cell block disagrees with independent nodal integration." ) && success;
   }

   Vector x( 8 );
   FillDeterministicInput( x );
   const auto expected = ApplyRectangularOperator( generic, x, 18 );
   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Finalized rectangular RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Rectangular COO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Rectangular CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Rectangular CSC action disagrees with MakeGenericOperator." ) && success;

   return success;
}

bool TestRectangularP1TrialP2TestAffineBoundaryAssembly()
{
   Cartesian2DMesh mesh( 1.0, 2, 1 );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto source = MakeCoefficient< "source" >(
      [] GENDIL_HOST_DEVICE () -> Real { return 1.375; } );
   auto form =
      integrate(
         BoundaryFacets< "mesh" >{},
         u * v + source * v );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      raw.num_rows == 18 && raw.num_cols == 8 && raw.nnz_raw == 72,
      "Rectangular affine-boundary RawCOO does not contain two 9 x 4 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct affine-boundary COO disagrees with finalized RawCOO." ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check(
      coo.num_rows == 18 && coo.num_cols == 8 &&
      csr.num_rows == 18 && csr.num_cols == 8 &&
      csc.num_rows == 18 && csc.num_cols == 8,
      "A derived affine-boundary format lost the 18 x 8 dimensions." ) && success;

   Vector x( 8 );
   FillDeterministicInput( x );
   Vector zero( 8 );
   zero = 0.0;
   const auto fx = ApplyRectangularOperator( generic, x, 18 );
   const auto fzero = ApplyRectangularOperator( generic, zero, 18 );
   Vector expected( 18 );
   Real * expected_data = expected.WriteHostData();
   const Real * fx_data = fx.ReadHostData();
   const Real * fzero_data = fzero.ReadHostData();
   for ( GlobalIndex i = 0; i < expected.Size(); ++i )
   {
      expected_data[i] = fx_data[i] - fzero_data[i];
   }

   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Affine-boundary RawCOO action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Affine-boundary COO action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Affine-boundary CSR action disagrees with F(x) - F(0)." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Affine-boundary CSC action disagrees with F(x) - F(0)." ) && success;

   return success;
}

struct OrientedTwoQuadFixture
{
   // Cell 0 is axis-aligned. Cell 1 is rotated by 180 degrees, so both cells
   // are positively oriented but their shared-face tangential coordinates run
   // in opposite directions.
   std::array< Real, 12 > node_data{
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      1.0, 1.0,
      2.0, 0.0,
      2.0, 1.0 };
   std::array< int, 8 > restriction_data{
      0, 1, 2, 3,
      5, 3, 4, 1 };
   HostDevicePointer< const int > restriction_pointer{};
   UnstructuredConformingConnectivity< HyperCube< 2 > > connectivity{ 2 };

   OrientedTwoQuadFixture()
   {
      restriction_pointer.host_pointer = restriction_data.data();
      for ( GlobalIndex element = 0; element < 2; ++element )
      {
         for ( Integer face = 0; face < HyperCube< 2 >::num_faces; ++face )
         {
            connectivity[element].faces[face] =
               { 0,
                 {},
                 MakeReferencePermutation< 2 >(),
                 {},
                 {},
                 true };
         }
      }

      const Permutation< 2 > rotated_neighbor{ { -1, -2 } };
      connectivity[0].faces[2] =
         { 1, {}, rotated_neighbor, {}, {}, false };
      connectivity[1].faces[2] =
         { 0, {}, rotated_neighbor, {}, {}, false };
   }

   auto MakeMesh() const
   {
      return QuadMesh< 1 >{
         MakeFIFOView( node_data.data(), GlobalIndex( 2 ), GlobalIndex( 6 ) ),
         MakeFIFOView(
            restriction_pointer,
            GlobalIndex( 2 ),
            GlobalIndex( 2 ),
            GlobalIndex( 2 ) ),
         connectivity,
         2 };
   }
};

Real IndependentP2P1TangentialMass(
   const size_t trial,
   const size_t test )
{
   constexpr std::array< Real, 3 > trial_nodes{
      0.11270166537925831148,
      0.5,
      0.88729833462074168852 };
   constexpr std::array< Real, 2 > test_nodes{
      0.21132486540518711775,
      0.78867513459481288225 };
   constexpr std::array< Real, 5 > points{
      0.0469100770306680,
      0.230765344947158,
      0.500000000000000,
      0.769234655052842,
      0.953089922969332 };
   constexpr std::array< Real, 5 > weights{
      0.118463442528095,
      0.239314335249683,
      0.284444444444444,
      0.239314335249683,
      0.118463442528095 };

   Real value = 0.0;
   for ( size_t q = 0; q < points.size(); ++q )
   {
      value +=
         weights[q] *
         IndependentLagrangeValue( trial_nodes, trial, points[q] ) *
         IndependentLagrangeValue( test_nodes, test, points[q] );
   }
   return value;
}

bool TestRectangularP2TrialP1TestOrientedInteriorAssembly()
{
   OrientedTwoQuadFixture fixture;
   auto mesh = fixture.MakeMesh();
   const auto fixture_face =
      mesh.GetLocalFaceInfo(
         0,
         std::integral_constant< Integer, 2 >{} );
   auto trial_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 2, 2 >{} );
   auto test_fe =
      MakeLegendreFiniteElement( FiniteElementOrders< 1, 1 >{} );
   auto trial_space = MakeFiniteElementSpace( mesh, trial_fe );
   auto test_space = MakeFiniteElementSpace( mesh, test_fe );

   TrialSpace< "u" > u;
   TestSpace< "v" > v;
   auto form =
      integrate(
         InteriorFacets< "mesh" >{},
         jump( u ) * jump( v ) );
   auto context =
      MakeWeakFormContext(
         MakeTrialField< "u" >( trial_space ),
         MakeTestField< "v" >( test_space ),
         MakeIntegrationDomain< "mesh" >( mesh ) );
   auto integration_rule =
      MakeIntegrationRule( IntegrationRuleNumPoints< 4, 4 >{} );
   using KernelPolicy = SerialKernelConfiguration;

   auto raw =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto finalized = FinalizeRawCOOToCOOHost( raw );
   auto coo =
      GenericAssembly< MatrixAssemblyType::COO, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csr =
      GenericAssembly< MatrixAssemblyType::CSR, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto csc =
      GenericAssembly< MatrixAssemblyType::CSC, KernelPolicy >(
         form,
         context,
         integration_rule );
   auto generic =
      MakeGenericOperator< KernelPolicy >(
         form,
         context,
         integration_rule );

   bool success = true;
   success = Check(
      fixture_face.PlusSide().orientation ==
         Permutation< 2 >{ { -1, -2 } },
      "Two-quad fixture did not retain its nonidentity face orientation." ) && success;
   for ( GlobalIndex element = 0; element < 2; ++element )
   {
      const int node00 = fixture.restriction_data[4 * element];
      const int node10 = fixture.restriction_data[4 * element + 1];
      const int node01 = fixture.restriction_data[4 * element + 2];
      const Real dx_x =
         fixture.node_data[2 * node10] - fixture.node_data[2 * node00];
      const Real dx_y =
         fixture.node_data[2 * node10 + 1] -
         fixture.node_data[2 * node00 + 1];
      const Real dy_x =
         fixture.node_data[2 * node01] - fixture.node_data[2 * node00];
      const Real dy_y =
         fixture.node_data[2 * node01 + 1] -
         fixture.node_data[2 * node00 + 1];
      const Real determinant =
         dx_x * dy_y - dx_y * dy_x;
      success = Check(
         determinant > 0.0,
         "Two-quad orientation fixture contains a non-positive cell." ) && success;
   }
   success = Check(
      raw.num_rows == 8 && raw.num_cols == 18,
      "Reverse rectangular interior RawCOO dimensions are not 8 x 18." ) && success;
   success = Check(
      raw.nnz_raw == 144,
      "Reverse rectangular interior RawCOO must contain four 4 x 9 blocks." ) && success;
   success = CheckRawTripletRangesAndFinite( raw ) && success;
   success = CheckCOOMatricesEqual(
      coo,
      finalized,
      "Direct reverse-interior COO disagrees with finalized RawCOO." ) && success;
   success = CheckCanonicalCOOSortedUnique( finalized ) && success;
   success = CheckCanonicalCOOSortedUnique( coo ) && success;
   success = Check(
      coo.num_rows == 8 && coo.num_cols == 18 &&
      csr.num_rows == 8 && csr.num_cols == 18 &&
      csc.num_rows == 8 && csc.num_cols == 18,
      "A derived interior format lost the 8 x 18 dimensions." ) && success;

   const auto raw_data = GetHostReadView( raw );
   constexpr GlobalIndex first_neighbor_offset = 36;
   constexpr std::array< GlobalIndex, 4 > selected_trial_dofs{
      0, 1, 5, 8 };
   for ( const GlobalIndex trial : selected_trial_dofs )
   {
      const GlobalIndex trial_x = trial % 3;
      const GlobalIndex trial_y = trial / 3;
      const std::array< GlobalIndex, 2 > native_trial_indices{
         trial_x,
         trial_y };
      const GlobalIndex expected_col =
         GlobalDofIndex( trial_space, 1, native_trial_indices );

      for ( GlobalIndex test = 0; test < 4; ++test )
      {
         const GlobalIndex test_x = test % 2;
         const GlobalIndex test_y = test / 2;
         const std::array< GlobalIndex, 2 > test_indices{
            test_x,
            test_y };
         const GlobalIndex expected_row =
            GlobalDofIndex( test_space, 0, test_indices );
         const GlobalIndex entry =
            first_neighbor_offset + trial * 4 + test;
         success = Check(
            raw_data.rows[entry] == expected_row,
            "Oriented neighbor RawCOO row is not from the p1 minus-side test space." ) && success;
         success = Check(
            raw_data.cols[entry] == expected_col,
            "Oriented neighbor RawCOO column does not apply the p2 plus-side reversal." ) && success;

         constexpr std::array< Real, 2 > p1_nodes{
            0.21132486540518711775,
            0.78867513459481288225 };
         constexpr std::array< Real, 3 > p2_nodes{
            0.11270166537925831148,
            0.5,
            0.88729833462074168852 };
         // RawCOO slots are indexed by native neighbor columns. The {-1,-2}
         // plus orientation maps the reversed reference basis into this slot.
         const GlobalIndex reference_trial_x = 2 - trial_x;
         const GlobalIndex reference_trial_y = 2 - trial_y;
         const Real expected_value =
            -IndependentLagrangeValue(
               p1_nodes,
               static_cast<size_t>( test_x ),
               1.0 ) *
            IndependentLagrangeValue(
               p2_nodes,
               static_cast<size_t>( reference_trial_x ),
               0.0 ) *
            IndependentP2P1TangentialMass(
               static_cast<size_t>( reference_trial_y ),
               static_cast<size_t>( test_y ) );
         success = Check(
            Near( raw_data.values[entry], expected_value ),
            "Oriented neighbor RawCOO value disagrees with the independent face block." ) && success;
      }
   }

   Vector x( 18 );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] =
         0.35 + 0.11 * static_cast< Real >( i ) +
         0.017 * static_cast< Real >( i * i );
   }
   const auto expected = ApplyRectangularOperator( generic, x, 8 );
   success = CheckRectangularAction(
      finalized,
      x,
      expected,
      "Reverse-interior RawCOO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      coo,
      x,
      expected,
      "Reverse-interior COO action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csr,
      x,
      expected,
      "Reverse-interior CSR action disagrees with MakeGenericOperator." ) && success;
   success = CheckRectangularAction(
      csc,
      x,
      expected,
      "Reverse-interior CSC action disagrees with MakeGenericOperator." ) && success;

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
   success = TestRectangularP1TrialP2TestCellAssembly() && success;
   success = TestRectangularP1TrialP2TestAffineBoundaryAssembly() && success;
   success = TestRectangularP2TrialP1TestOrientedInteriorAssembly() && success;

   return success ? 0 : 1;
}
