// Copyright GenDiL Project Developers. See COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gendil/gendil.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace gendil;

namespace
{

constexpr Real tolerance = 1.0e-12;

struct Triplet
{
   GlobalIndex row;
   GlobalIndex col;
   Real value;
};

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

std::vector< Triplet > SortReduceTriplets(
   const RawCOOTripletBuffer< Real, GlobalIndex > & buffer )
{
   std::vector< Triplet > triplets;
   triplets.reserve( static_cast< size_t >( buffer.nnz_raw ) );

   for ( GlobalIndex i = 0; i < buffer.nnz_raw; ++i )
   {
      triplets.push_back(
         Triplet{ buffer.rows[i], buffer.cols[i], buffer.values[i] } );
   }

   std::sort(
      triplets.begin(),
      triplets.end(),
      [] ( const Triplet & lhs, const Triplet & rhs )
      {
         if ( lhs.row != rhs.row )
         {
            return lhs.row < rhs.row;
         }
         return lhs.col < rhs.col;
      });

   std::vector< Triplet > reduced;
   for ( const auto & triplet : triplets )
   {
      if ( !reduced.empty() &&
           reduced.back().row == triplet.row &&
           reduced.back().col == triplet.col )
      {
         reduced.back().value += triplet.value;
      }
      else
      {
         reduced.push_back( triplet );
      }
   }

   return reduced;
}

void ApplyReducedCOO(
   const std::vector< Triplet > & triplets,
   const Vector & x,
   Vector & y )
{
   const Real * x_data = x.ReadHostData();
   Real * y_data = y.WriteHostData();

   for ( GlobalIndex i = 0; i < y.Size(); ++i )
   {
      y_data[i] = 0.0;
   }

   for ( const auto & triplet : triplets )
   {
      y_data[triplet.row] += triplet.value * x_data[triplet.col];
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
         MakeDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
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
      coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Raw COO row dimension is wrong." ) && success;
   success = Check(
      coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "Raw COO column dimension is wrong." ) && success;
   success = Check(
      coo.nnz_raw == expected_nnz,
      "Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( coo, fe_space ) && success;

   auto reduced = SortReduceTriplets( coo );
   success = Check(
      reduced.size() == static_cast< size_t >( coo.nnz_raw ),
      "Scalar L2 cell-only RawCOO should not create duplicate triplets." ) && success;

   Vector x( fe_space.GetNumberOfFiniteElementDofs() );
   Real * x_data = x.WriteHostData();
   for ( GlobalIndex i = 0; i < x.Size(); ++i )
   {
      x_data[i] = 0.25 + 0.5 * static_cast< Real >( i );
   }

   Vector y_coo( fe_space.GetNumberOfFiniteElementDofs() );
   Vector y_bsr( fe_space.GetNumberOfFiniteElementDofs() );
   ApplyReducedCOO( reduced, x, y_coo );
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

   FreeRawCOOTripletBuffer( coo );
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
         MakeDomain< "mesh" >( mesh ) );

   constexpr Integer num_quad_1d = order + 2;
   IntegrationRuleNumPoints< num_quad_1d > nq;
   auto integration_rule = MakeIntegrationRule( nq );

   using KernelPolicy = SerialKernelConfiguration;

   auto coo =
      GenericAssembly< MatrixAssemblyType::RawCOO, KernelPolicy >(
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
      coo.num_rows ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "H1 Raw COO row dimension is wrong." ) && success;
   success = Check(
      coo.num_cols ==
         static_cast< GlobalIndex >( fe_space.GetNumberOfFiniteElementDofs() ),
      "H1 Raw COO column dimension is wrong." ) && success;
   success = Check(
      coo.nnz_raw == expected_nnz,
      "H1 Raw COO cell-mass triplet count is wrong." ) && success;
   success = CheckRawTripletRangesAndFinite( coo ) && success;
   success = CheckScalar1DRawCellSlotCoordinates( coo, fe_space ) && success;
   success = Check(
      HasDuplicateCoordinate( coo ),
      "Scalar H1 RawCOO should preserve duplicate raw triplet coordinates." ) && success;

   auto reduced = SortReduceTriplets( coo );
   success = Check(
      coo.nnz_raw > static_cast< GlobalIndex >( reduced.size() ),
      "Scalar H1 RawCOO should have fewer canonical entries after reduction." ) && success;

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
      ApplyReducedCOO( reduced, x, y_coo );
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

   FreeRawCOOTripletBuffer( coo );
   return success;
}

} // namespace

int main()
{
   bool success = true;
   success = TestRawCOOBufferAllocation() && success;
   success = TestScalarL2CellMassRawCOOAgainstBSR() && success;
   success = TestScalarH1CellMassRawCOOPreservesDuplicatesAgainstSGBSR() && success;

   return success ? 0 : 1;
}
